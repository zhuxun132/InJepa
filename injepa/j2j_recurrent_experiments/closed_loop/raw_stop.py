"""Portable RAW distance measurements with independent statistic recomputation."""
from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path

from j2j.authority import git_head
from j2j.data.keys import frame_key as canonical_frame_key
from j2j.data.source import open_released_streamvln_source
from j2j.encoding.cache import open_cache_store
from j2j.receipts import canonical_json_bytes
from j2j.encoding.vjepa2 import OFFICIAL_SOURCE_COMMIT, OFFICIAL_SOURCE_TREE
from j2j.evaluation.vjepa_grid_encoder import (
    VJEPA_NATIVE_POOL_IDENTITY_SHA256, VJEPA_PREPROCESS_IDENTITY_SHA256,
)
from j2j_iclr_experiments.online import stop as legacy


def _coordinate(value):
    fields = legacy._COORDINATE_FIELDS | {"representation", "spatial_shape"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("RAW coordinate fields differ")
    result = dict(value)
    shape = result["spatial_shape"]
    if (not isinstance(shape, (tuple, list)) or len(shape) != 2
            or any(type(x) is not int or x <= 0 for x in shape)
            or shape[1] != 768 or math.isqrt(shape[0]) ** 2 != shape[0]):
        raise ValueError("RAW coordinate requires a square spatial grid")
    result["spatial_shape"] = list(shape)
    fixed = dict(vjepa_source_commit=OFFICIAL_SOURCE_COMMIT,
                 vjepa_source_tree=OFFICIAL_SOURCE_TREE,
                 preprocess_sha256=VJEPA_PREPROCESS_IDENTITY_SHA256,
                 pool_sha256=VJEPA_NATIVE_POOL_IDENTITY_SHA256,
                 whitening_sha256=None, representation="raw")
    if any(result[k] != v for k, v in fixed.items()):
        raise ValueError("RAW native coordinate identity differs")
    legacy._receipt_sha(result["checkpoint_sha256"], name="RAW checkpoint")
    return result


def close_raw_visual_coordinate_identity(*, training_expected_identities,
        encoder_provenance, stop_coordinate_identity):
    coordinate = _coordinate(dict(encoder_provenance["coordinate_identity"]))
    if coordinate != _coordinate(dict(stop_coordinate_identity)):
        raise ValueError("STOP and runtime coordinates differ")
    if encoder_provenance.get("whitening", "missing") is not None:
        raise ValueError("RAW encoder cannot whiten")
    ledger = encoder_provenance.get("load_ledger", {})
    if (ledger.get("load_status") != "STRICT_FROZEN_VJEPA2_LOADED"
            or ledger.get("source_commit") != coordinate["vjepa_source_commit"]
            or ledger.get("source_tree") != coordinate["vjepa_source_tree"]
            or ledger.get("checkpoint_sha256") != coordinate["checkpoint_sha256"]
            or encoder_provenance.get("vjepa_checkpoint", {}).get("sha256") != coordinate["checkpoint_sha256"]):
        raise ValueError("RAW encoder strict load identity differs")
    for field in legacy._CACHE_IDENTITY_FIELDS:
        if field not in training_expected_identities or training_expected_identities[field] != coordinate[field]:
            raise ValueError(f"RAW training coordinate differs: {field}")
    return coordinate


def _identity(path):
    path = Path(path).resolve()
    content = path.read_bytes()
    return {"path": str(path), "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest()}


def _read_bound(path, digest, size=None):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != digest or (size is not None and len(content) != size):
        raise ValueError("RAW STOP artifact bytes differ")
    return content


def _portable_identity(path, *, relative_path=None):
    identity = _identity(path)
    identity.pop("path")
    if relative_path is not None:
        identity["relative_path"] = relative_path
    return identity


def _source_cache_locators(source):
    locators = {}
    used_rows = set()
    for item in source:
        trajectory = item.canonical_trajectory
        for step, global_row in enumerate(item.cache_global_rows):
            frame = canonical_frame_key(
                trajectory.canonical_trajectory_key, step
            ).hex()
            if frame in locators:
                raise ValueError("RAW STOP source duplicates a canonical frame key")
            if global_row in used_rows:
                raise ValueError("RAW STOP source reuses a cache row")
            used_rows.add(global_row)
            locators[frame] = (
                global_row,
                {
                    "frame_key": frame,
                    "source_dataset": trajectory.source_id,
                    "partition": item.projection_partition,
                    "building": trajectory.scan_id,
                    "compressed_jpeg_sha256": trajectory.compressed_jpeg_sha256s[
                        step
                    ],
                    "decoded_rgb_sha256": trajectory.decoded_rgb_sha256s[step],
                },
            )
    return locators


def _record_identity(record):
    frame = getattr(record, "frame_key", None)
    return {
        "frame_key": frame.hex() if isinstance(frame, bytes) else None,
        "source_dataset": getattr(record, "source_dataset", None),
        "partition": getattr(record, "partition", None),
        "building": getattr(record, "building", None),
        "compressed_jpeg_sha256": getattr(
            record, "compressed_jpeg_sha256", None
        ),
        "decoded_rgb_sha256": getattr(record, "decoded_rgb_sha256", None),
    }


def _measure_source_rows(population, *, cache_store, locators):
    measured = []
    for row in population:
        try:
            current_locator, current_expected = locators[row["current_frame_key"]]
            goal_locator, goal_expected = locators[row["goal_frame_key"]]
        except KeyError as exc:
            raise ValueError(
                "RAW STOP pair frame is absent from the source catalog"
            ) from exc
        try:
            current_record, goal_record = cache_store.read_global_rows(
                (current_locator, goal_locator)
            )
        except Exception as exc:
            raise ValueError("RAW STOP cache pair read failed") from exc
        for role, record, expected in (
            ("current", current_record, current_expected),
            ("goal", goal_record, goal_expected),
        ):
            if _record_identity(record) != expected:
                raise ValueError(
                    f"RAW STOP {role} source/cache frame identity differs"
                )
        result = dict(row)
        result["distance"] = legacy.canonical_stop_distance(
            current_record.grid, goal_record.grid
        )
        measured.append(result)
    return tuple(measured)


def produce_raw_stop_bundle(*, canonical_source_manifest_path,
        canonical_source_manifest_sha256, source_catalog_path,
        source_catalog_sha256, raw_cache_manifest_path,
        raw_cache_manifest_sha256, expected_parent_manifest_sha256,
        expected_identities, protocol_sha256, output_dir, source_revision=None):
    """Measure the source-derived RAW STOP population and write a portable bundle."""
    legacy._receipt_sha(protocol_sha256, name="RAW STOP protocol")
    if not isinstance(expected_identities, Mapping):
        raise ValueError("RAW STOP expected cache identities must be a mapping")

    source_manifest = Path(canonical_source_manifest_path)
    source_catalog = Path(source_catalog_path)
    cache_manifest_path = Path(raw_cache_manifest_path)
    source_manifest_bytes = _read_bound(
        source_manifest, canonical_source_manifest_sha256
    )
    source_catalog_bytes = _read_bound(source_catalog, source_catalog_sha256)
    cache_manifest_bytes = _read_bound(
        cache_manifest_path, raw_cache_manifest_sha256
    )
    if cache_manifest_path.name != "manifest.json":
        raise ValueError("RAW STOP cache artifact must be manifest.json")
    try:
        cache_manifest = json.loads(cache_manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("RAW STOP cache manifest is unreadable") from exc
    if not isinstance(cache_manifest, dict):
        raise ValueError("RAW STOP cache manifest must be a mapping")

    if "spatial_shape" not in cache_manifest:
        raise ValueError("RAW cache must declare its spatial shape")
    shape = cache_manifest["spatial_shape"]
    coordinate = _coordinate({
        **dict(expected_identities),
        "vjepa_source_tree": OFFICIAL_SOURCE_TREE,
        "representation": "raw",
        "spatial_shape": shape,
    })
    try:
        cache_store = open_cache_store(
            cache_manifest_path.parent,
            expected_parent_manifest_sha256=expected_parent_manifest_sha256,
            expected_identities=expected_identities,
            require_stage="RAW32",
            require_training_eligible=True,
            require_production_eligible=True,
        )
        source = open_released_streamvln_source(
            canonical_manifest=source_manifest,
            source_catalog=source_catalog,
            cache_store=cache_store,
            expected_manifest_sha256=canonical_source_manifest_sha256,
            partition="project-dev",
            require_production_eligible=True,
        )
    except Exception as exc:
        raise ValueError(
            f"RAW STOP source/catalog/cache authority could not be admitted: {exc}"
        ) from exc

    population = legacy._source_derived_population(
        canonical_source_manifest_path=source_manifest,
        canonical_source_manifest_sha256=canonical_source_manifest_sha256,
    )
    rows = _measure_source_rows(
        population,
        cache_store=cache_store,
        locators=_source_cache_locators(source),
    )
    census = dict(legacy.validate_stop_pair_ledger(
        rows,
        canonical_source_manifest_path=source_manifest,
        canonical_source_manifest_sha256=canonical_source_manifest_sha256,
    ))
    census["class_counts"] = dict(
        positive=sum(bool(row["label"]) for row in rows),
        negative=sum(not bool(row["label"]) for row in rows),
    )
    census["building_count"] = len({row["building"] for row in rows})

    ledger_bytes = b"".join(canonical_json_bytes(row) for row in rows)
    ledger_identity = {
        "bytes": len(ledger_bytes),
        "sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "basename": "pair_ledger.jsonl",
        "row_count": len(rows),
        "ordered_pair_key_sha256": hashlib.sha256(canonical_json_bytes(
            [row["pair_key"] for row in rows]
        )).hexdigest(),
    }
    producer_path = Path(__file__).resolve()
    legacy_path = Path(legacy.__file__).resolve()
    repository = producer_path.parents[2]
    revision = git_head(repository) if source_revision is None else source_revision
    if not isinstance(revision, str) or legacy._GIT_IDENTITY.fullmatch(revision) is None:
        raise ValueError("RAW producer source revision must be a Git commit identity")
    receipt = {
        "schema": "J2J_RAW_STOP_MEASUREMENT_V1",
        "status": "MEASURED_SOURCE_BOUND",
        "protocol_sha256": protocol_sha256,
        "partition": "project-dev",
        "simulator_data_used": False,
        "pair_contracts": {
            **legacy._PAIR_CONTRACTS,
            "source_binding": "J2J_RAW_STOP_MEASUREMENT_LEDGER_V1",
        },
        "distance_contract": {
            "name": "cpu_c_order_float32_mean_absolute",
            "spatial_shape": coordinate["spatial_shape"],
            "source": _portable_identity(
                legacy_path,
                relative_path="j2j_iclr_experiments/online/stop.py",
            ),
        },
        "ledger": ledger_identity,
        "population_census": census,
        "source_artifacts": {
            "canonical_source_manifest": {
                "bytes": len(source_manifest_bytes),
                "sha256": canonical_source_manifest_sha256,
            },
            "source_catalog": {
                "bytes": len(source_catalog_bytes),
                "sha256": source_catalog_sha256,
            },
        },
        "cache_artifact": {
            "bytes": len(cache_manifest_bytes),
            "sha256": raw_cache_manifest_sha256,
            "parent_manifest_sha256": cache_manifest.get(
                "parent_manifest_sha256"
            ),
            "frame_ledger_sha256": cache_manifest.get(
                "frame_ledger_sha256"
            ),
            "stage": cache_manifest.get("stage"),
            "dtype": cache_manifest.get("dtype"),
            "spatial_shape": coordinate["spatial_shape"],
            "training_eligible": cache_manifest.get("training_eligible"),
            "production_eligible": cache_manifest.get("production_eligible"),
        },
        "visual_coordinate_identity": coordinate,
        "producer_code": {
            "repository_revision": revision,
            "producer": _portable_identity(
                producer_path,
                relative_path=(
                    "j2j_recurrent_experiments/closed_loop/raw_stop.py"
                ),
            ),
        },
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "pair_ledger.jsonl").write_bytes(ledger_bytes)
    (directory / "producer_receipt.json").write_bytes(
        canonical_json_bytes(receipt)
    )
    return receipt


def _numeric_census(rows, source_sha):
    """Check portable pair relationships; source completeness was measured upstream."""
    seen = set()
    positives = []
    if not rows or len(rows) % 2:
        raise ValueError("RAW STOP requires paired positive/negative rows")
    for index, row in enumerate(rows):
        if set(row) != legacy._LEDGER_FIELDS:
            raise ValueError("RAW STOP pair fields differ")
        kind = row["pair_type"]
        if type(row["label"]) is not bool or row["label"] != (index % 2 == 0):
            raise ValueError("RAW STOP label/order differs")
        if kind not in (("SELF", "PURE_TURN") if row["label"] else ("NEGATIVE",)):
            raise ValueError("RAW STOP pair type differs")
        if row["partition"] != "project-dev" or row["goal_partition"] != "project-dev":
            raise ValueError("RAW STOP pairs must be released dev")
        if row["source_manifest_sha256"] != source_sha or row["source_id"] != row["goal_source_id"]:
            raise ValueError("RAW STOP source identity differs")
        distance = legacy._finite_number(row["distance"], name="RAW distance")
        if distance < 0:
            raise ValueError("RAW distance is negative")
        for prefix, trajectory_field, step_field in (
                ("current", "trajectory_key", "current_step"),
                ("goal", "goal_trajectory_key", "goal_step")):
            if row[prefix + "_frame_key"] != legacy._frame_key(row[trajectory_field], row[step_field]):
                raise ValueError("RAW STOP frame/time identity differs")
            legacy._receipt_sha(row[prefix + "_rgb_sha256"], name="RGB")
        pair = legacy._pair_key(kind, row["current_frame_key"], row["goal_frame_key"])
        if pair != row["pair_key"] or pair in seen:
            raise ValueError("RAW STOP pair key differs or repeats")
        seen.add(pair)
        if row["building"] != row["current_scan"] or row["goal_building"] != row["goal_scan"]:
            raise ValueError("RAW STOP building identity differs")
        if row["label"]:
            positives.append(pair)
            if row["positive_pair_key"] != pair or row["trajectory_key"] != row["goal_trajectory_key"]:
                raise ValueError("RAW STOP positive relation differs")
            if kind == "SELF":
                if distance != 0 or row["current_step"] != row["goal_step"] or row["current_step"] != row["terminal_step"]:
                    raise ValueError("RAW STOP SELF relation differs")
            elif row["goal_step"] != row["current_step"] + 1 or row["transition_action_id"] not in (2, 3):
                raise ValueError("RAW STOP turn relation differs")
        else:
            positive = rows[index - 1]
            if (row["positive_pair_key"] != positive["pair_key"]
                    or row["current_frame_key"] != positive["current_frame_key"]
                    or row["current_scan"] == row["goal_scan"]):
                raise ValueError("RAW STOP negative relation differs")
    turns = [r for r in rows if r["pair_type"] == "PURE_TURN"]
    if positives != sorted(positives) or len({r["building"] for r in turns}) < 2 or not any(r["distance"] > 0 for r in turns):
        raise ValueError("RAW STOP population order/turn coverage differs")
    return dict(positive_by_type=dict(Counter(r["pair_type"] for r in rows if r["label"])),
        negative_count=len(rows)//2, pure_turn_building_count=len({r["building"] for r in turns}),
        partition="project-dev", formal_eligible=True,
        class_counts={"positive": len(rows)//2, "negative": len(rows)//2},
        building_count=len({r["building"] for r in rows}))


def _load_bundle(path, digest):
    path = Path(path).resolve()
    content = _read_bound(path, digest)
    producer = json.loads(content)
    if content != canonical_json_bytes(producer):
        raise ValueError("RAW STOP producer JSON is noncanonical")
    if (producer.get("schema") != "J2J_RAW_STOP_MEASUREMENT_V1"
            or producer.get("status") != "MEASURED_SOURCE_BOUND"
            or producer.get("partition") != "project-dev" or producer.get("simulator_data_used") is not False):
        raise ValueError("RAW STOP producer identity differs")
    legacy._receipt_sha(
        producer.get("protocol_sha256"), name="RAW STOP producer protocol"
    )
    coordinate = _coordinate(producer["visual_coordinate_identity"])
    cache = producer["cache_artifact"]
    if (cache.get("stage") != "RAW32" or cache.get("dtype") != "<f4"
            or cache.get("spatial_shape") != coordinate["spatial_shape"]
            or cache.get("training_eligible") is not True or cache.get("production_eligible") is not True):
        raise ValueError("RAW STOP cache identity differs")
    expected_contract = {**legacy._PAIR_CONTRACTS, "source_binding": "J2J_RAW_STOP_MEASUREMENT_LEDGER_V1"}
    distance_contract = producer["distance_contract"]
    if (producer["pair_contracts"] != expected_contract
            or distance_contract["name"] != "cpu_c_order_float32_mean_absolute"
            or distance_contract.get("spatial_shape") != coordinate["spatial_shape"]):
        raise ValueError("RAW STOP distance/coordinate contract differs")
    ledger = producer["ledger"]
    if ledger["basename"] != "pair_ledger.jsonl":
        raise ValueError("RAW STOP ledger basename differs")
    ledger_path = path.parent / ledger["basename"]
    blob = _read_bound(ledger_path, ledger["sha256"], ledger["bytes"])
    rows = [json.loads(line) for line in blob.splitlines()]
    if blob != b"".join(canonical_json_bytes(row) for row in rows):
        raise ValueError("RAW STOP ledger is noncanonical")
    keys_sha = hashlib.sha256(canonical_json_bytes([r["pair_key"] for r in rows])).hexdigest()
    if len(rows) != ledger["row_count"] or keys_sha != ledger["ordered_pair_key_sha256"]:
        raise ValueError("RAW STOP ledger population differs")
    census = _numeric_census(rows, producer["source_artifacts"]["canonical_source_manifest"]["sha256"])
    if census != producer["population_census"]:
        raise ValueError("RAW STOP census differs")
    return producer, rows, census, ledger_path


def consume_raw_stop_bundle(*, producer_receipt_path, producer_receipt_sha256,
        expected_protocol_sha256, encoder_provenance, bootstrap_parameters,
        output_dir):
    legacy._receipt_sha(
        expected_protocol_sha256, name="expected RAW STOP protocol"
    )
    producer, rows, census, ledger_path = _load_bundle(producer_receipt_path, producer_receipt_sha256)
    if producer["protocol_sha256"] != expected_protocol_sha256:
        raise ValueError("RAW STOP producer protocol differs")
    coordinate = close_raw_visual_coordinate_identity(
        training_expected_identities=producer["visual_coordinate_identity"],
        encoder_provenance=encoder_provenance,
        stop_coordinate_identity=producer["visual_coordinate_identity"])
    receipt = dict(schema="J2J_RAW_STOP_CALIBRATION_V1", status="FORMAL",
        protocol_sha256=expected_protocol_sha256,
        interpretation="released-data location-match transfer proxy",
        producer_receipt=_identity(producer_receipt_path), ledger=_identity(ledger_path),
        producer_provenance=producer, visual_coordinate_identity=coordinate,
        **legacy.derive_stop_calibration(rows, census=census, bootstrap_parameters=bootstrap_parameters))
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "RAW_STOP_CALIBRATION_V1.json").write_bytes(canonical_json_bytes(receipt))
    return receipt


def validate_raw_stop_calibration_receipt(receipt):
    if receipt.get("schema") != "J2J_RAW_STOP_CALIBRATION_V1" or receipt.get("status") != "FORMAL":
        raise ValueError("RAW STOP receipt is not formal")
    identity = receipt["producer_receipt"]
    producer, rows, census, ledger_path = _load_bundle(identity["path"], identity["sha256"])
    protocol_sha256 = receipt.get("protocol_sha256")
    legacy._receipt_sha(protocol_sha256, name="RAW STOP protocol")
    if protocol_sha256 != producer["protocol_sha256"]:
        raise ValueError("RAW STOP protocol differs from producer")
    if (identity != _identity(identity["path"]) or receipt["ledger"] != _identity(ledger_path)
            or receipt["producer_provenance"] != producer
            or _coordinate(receipt["visual_coordinate_identity"]) != _coordinate(producer["visual_coordinate_identity"])):
        raise ValueError("RAW STOP receipt provenance differs")
    derived = legacy.derive_stop_calibration(rows, census=census,
        bootstrap_parameters=receipt["bootstrap"]["parameters"])
    for field, value in derived.items():
        if not legacy._scientific_fields_equal(receipt.get(field), value):
            raise ValueError(f"RAW STOP {field} differs from recomputed value")
    return receipt
