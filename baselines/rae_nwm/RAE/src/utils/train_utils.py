from omegaconf import OmegaConf, DictConfig
from typing import List, Tuple



def parse_configs(config_path: str) -> Tuple[DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig]:
    """Load a config file and return component sections as DictConfigs."""
    config = OmegaConf.load(config_path)
    rae_config = config.get("stage_1", None)
    stage2_config = config.get("stage_2", None)
    transport_config = config.get("transport", None)
    sampler_config = config.get("sampler", None)
    guidance_config = config.get("guidance", None)
    misc = config.get("misc", None)
    training_config = config.get("training", None)
    return rae_config, stage2_config, transport_config, sampler_config, guidance_config, misc, training_config

def none_or_str(value):
    if value == 'None':
        return None
    return value