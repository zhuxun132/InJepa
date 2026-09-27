"""Content-bound pooled STOP admission without repeated cache reconstruction.

Attestation establishes full cache/statistics validation at production time.
Consumers rehash all directly declared files, never claim current raw-cache
row integrity, and never recompute calibration or bootstrap statistics.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import time
from weakref import WeakKeyDictionary
from j2j_iclr_experiments.common.artifacts import (
    canonical_mapping_sha256, create_once_json, deep_freeze, file_identity, require_sha256, to_plain_json,
)
from . import stop

_SCHEMA = 'J2J_POOLED_STOP_ATTESTATION_V1'
_SCOPE = 'full_source_cache_statistics_at_production;direct_file_content_hashes_at_admission;raw_cache_rows_not_rehashed_at_admission'
_SEAL = object()
_ISSUED = WeakKeyDictionary()


def _digest(value):
    return canonical_mapping_sha256(value)


def stop_admission_config_sha256(config):
    value = to_plain_json(config)
    section = value.get('context4', {})
    section.pop('decision', None)
    section.get('stop', {}).pop('attestation', None)
    return _digest(value)


@dataclass(frozen=True, eq=False)
class ValidatedStopReceipt:
    receipt: object
    receipt_sha256: str
    _seal: object


def _token(receipt):
    token = ValidatedStopReceipt(deep_freeze(to_plain_json(receipt)), _digest(receipt), _SEAL)
    _ISSUED[token] = token.receipt_sha256
    return token


def validate_stop_token(token, receipt):
    if type(token) is not ValidatedStopReceipt or token._seal is not _SEAL:
        raise ValueError('STOP admission requires an internally issued token')
    if _ISSUED.get(token) != token.receipt_sha256:
        raise ValueError('STOP admission token was not issued by validation')
    if token.receipt_sha256 != _digest(receipt) or token.receipt_sha256 != _digest(token.receipt):
        raise ValueError('STOP admission token receipt mismatch')
    return token.receipt


def validate_stop_once(receipt):
    started = time.monotonic()
    print(json.dumps({'event': 'stop_full_validation_start'}), flush=True)
    validated = stop.validate_stop_calibration_receipt(receipt)
    token = _token(validated)
    print(json.dumps({'event': 'stop_full_validation_complete', 'elapsed_s': time.monotonic() - started}), flush=True)
    return token


def _validators():
    return {'stop': file_identity(stop.__file__), 'admission': file_identity(__file__)}


def _inputs(receipt):
    return to_plain_json({'ledger': receipt['ledger'], 'calibration_provenance': receipt['calibration_provenance'], 'artifacts': receipt['provenance']['artifacts']})


def _rehash_inputs(identities):
    direct = {'ledger': identities['ledger'], 'calibration_provenance': identities['calibration_provenance'], **identities['artifacts']}
    for name, expected in direct.items():
        if file_identity(expected['path'], name=name) != expected:
            raise ValueError('STOP direct artifact content changed: ' + name)


def produce_stop_attestation(receipt, *, config_sha256, output_path):
    require_sha256(config_sha256, name='evaluation config')
    if Path(output_path).exists():
        raise FileExistsError(str(output_path))
    if receipt.get('schema') != 'STOP_CALIBRATION_V1':
        raise ValueError('attestation supports the pooled STOP protocol only')
    validators_before = _validators()
    token = validate_stop_once(receipt)
    if _validators() != validators_before:
        raise ValueError('STOP validator code changed during full validation')
    inputs = _inputs(token.receipt)
    _rehash_inputs(inputs)
    payload = {'schema': _SCHEMA, 'status': 'FULL_VALIDATION_PASSED', 'config_sha256': config_sha256,
               'receipt_sha256': token.receipt_sha256, 'protocol_sha256': stop._PROTOCOL_SHA256,
               'validator_identity': validators_before, 'input_identities': inputs, 'validation_scope': _SCOPE}
    create_once_json(output_path, payload)
    return payload


def admit_stop_attestation(receipt, attestation, *, config_sha256):
    started = time.monotonic()
    require_sha256(config_sha256, name='evaluation config')
    if receipt.get('schema') != 'STOP_CALIBRATION_V1':
        raise ValueError('attestation supports the pooled STOP protocol only')
    expected = {'schema': _SCHEMA, 'status': 'FULL_VALIDATION_PASSED', 'config_sha256': config_sha256,
                'receipt_sha256': _digest(receipt), 'protocol_sha256': stop._PROTOCOL_SHA256,
                'validator_identity': _validators(), 'input_identities': _inputs(receipt), 'validation_scope': _SCOPE}
    if to_plain_json(attestation) != expected:
        raise ValueError('STOP attestation scientific identity mismatch')
    if receipt["provenance"].get("numpy_version") != stop.np.__version__:
        raise ValueError("STOP NumPy version differs from attested producer")
    _rehash_inputs(expected['input_identities'])
    token = _token(receipt)
    print(json.dumps({'event': 'stop_attestation_admitted', 'elapsed_s': time.monotonic() - started, 'validation_scope': _SCOPE}), flush=True)
    return token
