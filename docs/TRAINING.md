# Train InJepa

Activate the [training environment](ENVIRONMENT.md) and start from the repository root:

```bash
export REPO="$PWD"
export PYTHONPATH="$REPO/injepa${PYTHONPATH:+:$PYTHONPATH}"
```

## Prepare the released trajectories

Follow [Data access and use](DATA_LICENSES.md), then download the published [StreamVLN R2R/RxR annotations and RGB archives](https://huggingface.co/datasets/cywan/StreamVLN-Trajectory-Data) at revision `dc61ee9b4e90aa7ba63c1163b2134df5610dccb9`. `data/training_assets.json` gives the required filenames, sizes and hashes. Place the files under these roots:

```text
ANNOTATIONS/
  R2R/annotations_v1-3.json
  RxR/annotations.json
RGB_ARCHIVES/
  R2R_images_v1-3.tar.gz
  RxR_images.tar.gz.part0
  RxR_images.tar.gz.part1
POINTNAV_EPISODES/
  mp3d/pointnav_mp3d_v1.zip
  gibson/pointnav_gibson_v1.zip
```

Keep the two RxR parts in this order and pass their directory to the census command. Its reader consumes the multipart archive. Download the original MP3D and Gibson PointNav episode archives through the [official Habitat dataset links](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md). These archives supply scene exclusion identities. Training images and actions come from the released StreamVLN trajectories.

```bash
export ANNOTATIONS=/absolute/path/to/streamvln/annotations
export RGB_ARCHIVES=/absolute/path/to/streamvln/rgb_archives
export POINTNAV_EPISODES=/absolute/path/to/pointnav_episode_archives
mkdir -p configs runs/data/census runs/data/payload_spool runs/data/receipts
python injepa/scripts/j2j/build_census.py \
  --config injepa/configs/census.yaml --asset-manifest data/training_assets.json \
  --annotation-root "$ANNOTATIONS" --rgb-root "$RGB_ARCHIVES" \
  --episode-root "$POINTNAV_EPISODES" --receipt-root runs/data/receipts \
  --canonical-manifest-root runs/data/census \
  --payload-spool-root runs/data/payload_spool --dev-count 11
```

The census verifies bytes and temporal/action alignment, deduplicates physical trajectories, creates the fixed building split, and retains verified original JPEG payloads. Its outputs feed the following steps:

| Output | Location |
|---|---|
| Source manifest and canonical trajectory/alias JSONL files | `runs/data/census/`, headed by `J2J_CANONICAL_SOURCE_MANIFEST_V1.json` |
| Verified JPEG bytes and frame index | `runs/data/payload_spool/J2J_VERIFIED_RGB_PAYLOADS_V1.spool` and `J2J_VERIFIED_RGB_PAYLOAD_INDEX_V1.jsonl` |
| Payload identity receipt used by the encoder | `runs/data/payload_spool/J2J_VERIFIED_RGB_PAYLOAD_SPOOL_V1.json` |
| Census receipt including building assignments | `runs/data/receipts/J2J_BUILD_CENSUS_RECEIPT_V1.json` |

The split is determined before feature encoding. Eligible buildings belong to both the released trajectories and MP3D PointNav train, after excluding the final MP3D/Gibson evaluation buildings. A fixed hash ordering assigns 11 eligible buildings to `project-dev` and the remaining 50 to `project-train`. The exact sets are in `data/scene_splits.json` (`validation` is the dev set), and trajectory identities are in `data/trajectory_index.jsonl`. The source manifest retains the split ledger and per-trajectory partitions. Keep `--dev-count 11` and these assignments for the reported setup. Clean150's 11 scenes are separate from both training and dev.

## Cache frozen V-JEPA features

Copy `injepa/configs/cache_vjepa.json` to `configs/cache_vjepa.local.json`. Set:

| Field | Local value |
|---|---|
| `canonical_manifest`, `canonical_manifest_sha256` | Census output manifest and its SHA-256 |
| `verified_payload_spool_receipt` | Receipt produced in `runs/data/payload_spool` |
| `vjepa_source_root`, `checkpoint` | V-JEPA source checkout and encoder checkpoint |
| `raw_output`, `workspace` | New cache and encoding work directories |
| `device`, `devices`, `encoding_devices` | The encoder device and recorded device list. Set optional `encoding_devices` to the same list when encoding on multiple GPUs |
| `workers`, `cache_batch` | RGB loading workers and encoding batch size |
| `reconstruction_command` | The local command used to build this cache |

Keep `representation: raw`, `spatial_shape: [576, 768]`, encoder identities and horizon 4. The supplied `expected_rows` is 849555 for the admitted canonical data. The driver verifies this count against the generated frame plan. The same cache contains both `project-train` and `project-dev` rows with their original partition tags. Then run:

```bash
python injepa/scripts/j2j/build_released_cache.py --config configs/cache_vjepa.local.json
```

The encoder writes `<raw_output>/manifest.json`, feature shards and frame indexes. Copy `injepa/configs/catalog_vjepa.template.json` to `configs/catalog_vjepa.local.json` and set:

| Field | Value from the preceding stages |
|---|---|
| `canonical_manifest`, `expected_manifest_sha256` | Census source manifest path and SHA-256 of that file |
| `z32_cache_dir` | The chosen `raw_output` directory |
| `u_cache_manifest` | `<raw_output>/manifest.json`, the same RAW32 cache manifest |
| `expected_parent_manifest_sha256` | The `parent_manifest_sha256` **field inside** that cache manifest |
| `expected_identities` | Preserve the V-JEPA identities from the cache configuration |
| `output_dir` | A new source-catalog directory, for example `runs/data/source_catalog` |

`z32_cache_dir` and `u_cache_manifest` are compatibility field names. `require_stage: RAW32` selects native, unwhitened V-JEPA features. Run:

```bash
python injepa/scripts/j2j/build_source_catalog.py --config configs/catalog_vjepa.local.json
```

The result is `<output_dir>/J2J_RELEASED_SOURCE_CATALOG_V1.json`. It joins the same train/dev trajectory partitions to cache frame rows, preserving their separation.

## Train Q/A/G/F

Copy `injepa/configs/train_vjepa.json` to `configs/train_vjepa.local.json`. Point `data.canonical_manifest`, `data.source_catalog` and `data.z32_cache_dir` to the generated source manifest, source catalog and RAW32 directory. Set `data.expected_parent_manifest_sha256` to the cache manifest's `parent_manifest_sha256` field. Set `identity.source_manifest_sha256` and `identity.cache_manifest_sha256` to the SHA-256 of the source manifest file and `<raw_output>/manifest.json`, respectively. Preserve the visual feature identities in `data.expected_identities`.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --standalone --nproc_per_node=4 injepa/scripts/train_j2j_context4.py \
  --config configs/train_vjepa.local.json --output-root runs/train_vjepa
```

The configuration uses global batch 128 (4 processes × 32 examples), AdamW, separate Q/AGF learning rates and clipping, and a 30-epoch cosine schedule. Adjust the runtime resource settings together with `world_size`, microbatch and accumulation while retaining the desired global batch. Keep output directories unique.

The trainer opens the catalog separately with `partition="project-train"` and `partition="project-dev"`. Only `project-train` batches update the model. Dev evaluation runs after each epoch and records objective losses, numerators/denominators, bounded model diagnostics and the train/dev gap through `write_validation`. `training.validation_enabled` defaults to `true` when absent, as in the supplied configuration. Setting it to `false` skips dev loading and evaluation. These are offline trajectory diagnostics. Run Clean150 separately with the evaluation commands in the README.

E12 is epoch 12 of this 30-epoch schedule. Keep `epochs: 30` when reproducing it. Q is trained on one next state with goal sampling horizon 4. The deployment horizon is separately configured. QG uses a detached posterior proposal. The visual encoder remains frozen.

The released checkpoint's `weights/injepa/e12_training_config.json` is its immutable training identity used for strict loading. Edit the local training configuration under `configs/` to start a new run.

## Source map

| Component | File |
|---|---|
| Proposal Q | `injepa/j2j/proposal/stochastic.py` |
| A/G/F | `injepa/module.py`, `injepa/j2j/spatial_intact.py` |
| Model assembly | `injepa/j2j/context4.py` |
| Training losses | `injepa/j2j/context4_objective.py` |
| Training loop | `injepa/j2j/context4_training.py` |
| Candidate rollout | `injepa/j2j/context4_rollout.py` |
| Closed-loop evaluation | `injepa/j2j_recurrent_experiments/closed_loop/` |
