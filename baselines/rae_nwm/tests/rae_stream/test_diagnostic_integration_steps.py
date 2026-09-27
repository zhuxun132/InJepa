"""RED feasibility gate: execute the actual unmodified official sampler call."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('requested',[50,10])
def test_official_rollout_reads_inference_config_steps(requested):
    root=Path(__file__).resolve().parents[2]
    tree=ast.parse((root/'planning_eval.py').read_text())
    evaluator=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WM_Planning_Evaluator')
    rollout=next(n for n in evaluator.body if isinstance(n,ast.FunctionDef) and n.name=='autoregressive_rollout_latent')
    calls=[n for n in ast.walk(rollout) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='sample_ode']
    assert len(calls)==1
    received=[]
    def sample_ode(**kwargs):received.append(kwargs)
    bound=SimpleNamespace(config={'transport':{'num_steps':requested,'sampling_method':'euler'}},
                          sampler=SimpleNamespace(sample_ode=sample_ode))
    expression=ast.Expression(body=calls[0])
    namespace={'self':bound}
    setup=next(n for n in ast.walk(rollout) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='integration_num_steps' for t in n.targets))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[setup],type_ignores=[])),'official_setup','exec'),namespace)
    eval(compile(ast.fix_missing_locations(expression),'official_rollout_sample_call','eval'),namespace)
    assert received[0]['num_steps']==requested
    assert received[0]['sampling_method']=='euler'


def test_cli_override_requires_diagnostic_and_records_request(tmp_path,capsys):
    import json
    from scripts.run_rae_stream_habitat import main
    args=['--runner-root',str(tmp_path),'--habitat-config','h.yaml','--episodes-path','e.json','--rae-root',str(tmp_path),'--dry-run']
    assert main(args+['--diagnostic-integration-steps','10'])==2
    assert main(args+['--diagnostic','--diagnostic-integration-steps','10'])==0
    assert json.loads(capsys.readouterr().out)['diagnostic_integration_steps']==10
    assert main(args)==0
    assert json.loads(capsys.readouterr().out)['diagnostic_integration_steps'] is None
    assert main(args+['--diagnostic','--diagnostic-integration-steps','0'])==2


@pytest.mark.parametrize('value',[True,0,-1,1.5,'10'])
def test_inference_config_bad_values(tmp_path,value):
    from rae_stream.diagnostic_inference import write_inference_config
    with pytest.raises(ValueError):write_inference_config(tmp_path/'missing',tmp_path/'out',value)


def test_inference_config_separate_and_only_num_steps_changed(tmp_path):
    import yaml
    from rae_stream.diagnostic_inference import write_inference_config
    original={'transport':{'num_steps':50,'sampling_method':'euler'},'model':'untouched','lr':.0002}
    path=tmp_path/'original.yaml';path.write_text(yaml.safe_dump(original));before=path.read_bytes()
    record=write_inference_config(path,tmp_path/'inference.yaml',10)
    actual=yaml.safe_load(Path(record['path']).read_text())
    assert actual['transport']['num_steps']==10
    actual['transport']['num_steps']=50
    assert actual==original and path.read_bytes()==before
    with pytest.raises(FileExistsError):write_inference_config(path,tmp_path/'inference.yaml',10)


def test_source_variant_must_be_explicit_and_exact(tmp_path,monkeypatch):
    from rae_stream import config_guard as guard
    root=Path(__file__).resolve().parents[2]
    # Real local source gate: diagnostic flag admits only the frozen planner bytes.
    with pytest.raises(RuntimeError):guard.verify_upstream_source(root)
    result=guard.verify_upstream_source(root,diagnostic_inference=True)
    assert result['diagnostic_source_variant']=='diagnostic_configurable_euler_v1'


def test_sampler_call_records_actual_steps_after_return():
    root=Path(__file__).resolve().parents[2]
    tree=ast.parse((root/'planning_eval.py').read_text())
    # Execute only the real scalar assignment/call/receipt sequence, no model reimplementation.
    assignments=[n for n in ast.walk(tree) if isinstance(n,ast.Assign)]
    call_assign=next(n for n in assignments if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='sample_ode')
    relevant=[n for n in assignments if any((isinstance(t,ast.Name) and t.id=='integration_num_steps') or (isinstance(t,ast.Attribute) and t.attr=='last_ode_num_steps') for t in n.targets)]
    assert len(relevant)==2
    received=[]
    self=SimpleNamespace(config={'transport':{'num_steps':10}},sampler=SimpleNamespace(sample_ode=lambda **kw:received.append(kw)))
    code=ast.Module(body=sorted(relevant+[call_assign],key=lambda n:n.lineno),type_ignores=[])
    exec(compile(ast.fix_missing_locations(code),'official_sampler_scalar_path','exec'),{'self':self})
    assert self.last_ode_num_steps==received[0]['num_steps']==10


def test_real_cli_factory_keeps_original_guard_and_binds_runtime_receipt(tmp_path,monkeypatch):
    import json,yaml
    from scripts import run_rae_stream_habitat as cli
    original=tmp_path/'config'/'rae_stream.yaml';original.parent.mkdir()
    original.write_text('transport:\n  num_steps: 50\nmodel: keep\n')
    checkpoint=tmp_path/'checkpoint';checkpoint.write_bytes(b'checkpoint')
    order=[]
    def guard(**kw):
        assert kw['config_path']==original and kw['diagnostic_inference'] is True
        assert yaml.safe_load(original.read_text())['transport']['num_steps']==50
        order.append('guard');return {'source':'explicit diagnostic'}
    def factory(**kw):
        order.append('factory')
        assert kw['config']!=original
        assert yaml.safe_load(kw['config'].read_text())['transport']['num_steps']==10
        return SimpleNamespace(plan=lambda *a:(0,0,0),evaluator=SimpleNamespace(last_ode_num_steps=10))
    monkeypatch.setattr(cli,'resolved_training_identity',guard)
    monkeypatch.setattr(cli,'_load_symbol',lambda *a,**kw:factory)
    runner=SimpleNamespace(load_habitat_environment=lambda **kw:SimpleNamespace(close=lambda:None),
                          run_imagegoal_episode=lambda *a,**kw:{'success':False,'num_steps':1})
    monkeypatch.setattr(cli,'_load_runner',lambda *a:runner)
    receipt=tmp_path/'receipt.json'
    args=['--runner-root',str(tmp_path),'--habitat-config','h.yaml','--episodes-path','e.json','--rae-root',str(tmp_path),
          '--checkpoint',str(checkpoint),'--diagnostic','--diagnostic-integration-steps','10','--receipt',str(receipt)]
    assert cli.main(args)==0
    assert order==['guard','factory']
    assert json.loads(receipt.read_text())['identity']['observed_ode_num_steps']==10
