"""File binding utilities for the public command-line entrypoints."""
from pathlib import Path
import gzip
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('injepa', 'lwm_croco', 'lwm_vjepa', 'nomad', 'rae_nwm')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve(path):
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)}


def checked(path, expected):
    path = resolve(path)
    if sha256(path) != expected:
        raise ValueError(f'SHA-256 mismatch: {path}')
    return path


def checked_ledger(path, expected):
    path = checked(path, expected)
    content = path.read_bytes()
    if path.suffix == '.gz':
        content = gzip.decompress(content)
    episodes = json.loads(content)['episodes']
    if len(episodes) != 150:
        raise ValueError('Clean150 requires 150 recorded episodes')
    return path, episodes


def resolve_fields(config, fields):
    for key in fields:
        if key in config and config[key] is not None:
            config[key] = str(resolve(config[key]))

