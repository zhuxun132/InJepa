"""Separate diagnostic inference YAML; never an admitted training overlay."""
import copy
from pathlib import Path
from .config_guard import load_yaml_mapping, sha256_file


def write_inference_config(original, destination, steps):
    if type(steps) is not int or steps <= 0:
        raise ValueError('diagnostic integration steps must be a positive integer')
    import yaml
    original=Path(original);destination=Path(destination)
    config=copy.deepcopy(load_yaml_mapping(original))
    config.setdefault('transport',{})['num_steps']=steps
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x') as handle:yaml.safe_dump(config,handle,sort_keys=False)
    return dict(path=str(destination.resolve()),sha256=sha256_file(destination),
        original_path=str(original.resolve()),original_sha256=sha256_file(original),
        num_steps=steps,scope='diagnostic_inference_only',sampling_method='euler')
