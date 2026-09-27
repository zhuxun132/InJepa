import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('rae_continuous_cli',Path(__file__).resolve().parents[2]/'scripts/run_rae_stream_habitat.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)

def args(tmp_path):return ['--runner-root',str(tmp_path),'--habitat-config',str(tmp_path/'h.yaml'),'--episodes-path',str(tmp_path/'e.json'),'--rae-root',str(tmp_path),'--dry-run']

def test_continuous_cli_requires_diagnostic(tmp_path):
    assert cli.main(args(tmp_path)+['--control-mode','continuous'])==2

def test_continuous_dry_run_records_controller_parameters(tmp_path,capsys):
    extra=['--diagnostic','--control-mode','continuous','--control-dt','.5','--control-max-translation','.2','--control-max-rotation','.3','--control-stop-translation-speed','.01','--control-stop-rotation-speed','.02']
    assert cli.main(args(tmp_path)+extra)==0
    record=json.loads(capsys.readouterr().out)
    assert record['control_mode']=='continuous'
    assert record['control_parameters']==dict(dt=.5,max_translation=.2,max_rotation=.3,translation_stop_speed=.01,rotation_stop_speed=.02)
    assert record['diagnostic'] is True
    assert record['decoder'] is None


def test_continuous_config_injects_official_typed_action_preserving_base(monkeypatch):
    from dataclasses import dataclass, field
    from typing import Dict
    import sys
    from types import ModuleType
    from omegaconf import OmegaConf
    @dataclass
    class ActionConfig:
        type: str = ''
    @dataclass
    class Task:
        actions: Dict[str, ActionConfig] = field(default_factory=lambda:{'stop':ActionConfig(type='StopAction')})
    @dataclass
    class Habitat:
        task: Task = field(default_factory=Task)
        seed: int = 17
    @dataclass
    class Config:
        habitat: Habitat = field(default_factory=Habitat)
    original=OmegaConf.structured(Config)
    OmegaConf.set_readonly(original,True)
    seen=[]
    def get_config(path):seen.append(path);return original
    for name,attributes in [
        ('habitat',{}),('habitat.config',{}),
        ('habitat.config.default',{'get_config':get_config}),
        ('habitat.config.default_structured_configs',{'ActionConfig':ActionConfig}),
    ]:
        module=ModuleType(name);module.__dict__.update(attributes);monkeypatch.setitem(sys.modules,name,module)
    config=cli._continuous_habitat_config('original.yaml')
    assert seen==['original.yaml']
    assert config.habitat.seed==17
    assert config.habitat.task.actions.stop.type=='StopAction'
    assert config.habitat.task.actions.rae_continuous.type=='RAEContinuousVelocityAction'
    assert OmegaConf.get_type(config.habitat.task.actions.rae_continuous) is ActionConfig
    assert OmegaConf.is_readonly(config)


def test_encoder_batch_size_is_explicit_runtime_identity(tmp_path,capsys):
    assert cli.main(args(tmp_path)+['--encoder-batch-size','7'])==0
    assert json.loads(capsys.readouterr().out)['encoder_batch_size']==7

@pytest.mark.parametrize('size',['0','-1'])
def test_encoder_batch_size_cli_rejects_nonpositive(tmp_path,size):
    assert cli.main(args(tmp_path)+['--encoder-batch-size',size])==2


def test_model_batch_size_is_explicit_runtime_identity(tmp_path,capsys):
    assert cli.main(args(tmp_path)+['--model-batch-size','7'])==0
    assert json.loads(capsys.readouterr().out)['model_batch_size']==7
