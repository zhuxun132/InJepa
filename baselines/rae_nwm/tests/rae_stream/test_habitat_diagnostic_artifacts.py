"""Evidence-only additions to existing RAE Habitat CLI."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("rae_diagnostic_cli",ROOT/"scripts/run_rae_stream_habitat.py")
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)


def setup(tmp_path,monkeypatch,fail_second=False):
    ledger=tmp_path/"episodes.json";ledger.write_text(json.dumps({"episodes":[
        {"episode_id":str(i),"scene_id":"scene/scene.glb"} for i in range(4)]}))
    ckpt=tmp_path/"checkpoint.pt";ckpt.write_bytes(b"real-factory-fixture")
    calls={"load":[],"run":[],"factory":[],"videos":[]}
    class Env:
        def __init__(self):self.i=0
        def get_metrics(self):return {"distance_to_goal":.75}
        def close(self):pass
    env=Env()
    def load(**kwargs):calls["load"].append(kwargs);return env
    def run(env,policy,**kwargs):
        calls["run"].append(kwargs)
        if fail_second and len(calls["run"])==2:raise RuntimeError("later episode failed")
        goal=np.zeros((8,8,3),dtype=np.uint8);current=np.ones_like(goal)
        obs=kwargs.get("frame_observer")
        if obs:obs({"rgb":current,"imagegoal":goal,"step":0,"action":None,"phase":"reset"})
        policy.reset(goal);action=policy.act(current,goal,[])
        if obs:obs({"rgb":current,"imagegoal":goal,"step":1,"action":"FWD","phase":"step"})
        assert action==1
        return {"scene_id":"scene","episode_id":kwargs.get("expected_episode_id","0"),
            "success":0.,"spl":0.,"distance_to_goal":.75,"num_steps":1,"termination_reason":"max_steps"}
    monkeypatch.setattr(cli,"_load_runner",lambda path:SimpleNamespace(load_habitat_environment=load,run_imagegoal_episode=run))
    monkeypatch.setattr(cli,"resolved_training_identity",lambda **kwargs:{"fixture":True})
    def factory(**kwargs):calls["factory"].append(kwargs);return SimpleNamespace(plan=lambda context,goal:(1/64,0.,0.))
    monkeypatch.setattr(cli,"_load_symbol",lambda *args,**kwargs:factory)
    args=["--runner-root",str(tmp_path),"--habitat-config",str(tmp_path/"habitat.yaml"),
        "--episodes-path",str(ledger),"--rae-root",str(tmp_path),"--checkpoint",str(ckpt),
        "--receipt",str(tmp_path/"result.json"),"--success-distance","1"]
    return args,calls


def test_real_diagnostic_binds_global_episode_and_is_not_formal(tmp_path,monkeypatch):
    args,calls=setup(tmp_path,monkeypatch)
    assert cli.main(args+["--diagnostic","--episode-indices","2","--planner-output-dir",str(tmp_path/"planner")])==0
    assert calls["load"][0]["episode_indices"]==[2]
    assert calls["run"][0]["expected_episode_id"]=="2"
    assert calls["run"][0]["expected_scene_id"]=="scene"
    assert calls["factory"][0]["output_dir"]==(tmp_path/"planner").resolve()
    assert json.loads((tmp_path/"result.json").read_text())["formal_evaluation_eligible"] is False


def test_post_action_command_trace_does_not_change_policy_inputs(tmp_path,monkeypatch):
    args,calls=setup(tmp_path,monkeypatch)
    trace_dir=tmp_path/"traces"
    assert cli.main(args+["--diagnostic","--episode-indices","2","--step-trace-dir",str(trace_dir)])==0
    rows=[json.loads(x) for x in (trace_dir/"episode_000002.steps.jsonl").read_text().splitlines()]
    assert len(rows)==1
    assert rows[0]["step"]==1 and rows[0]["action"]=="FWD"
    assert rows[0]["continuous_command"]==[1/64,0.,0.]
    assert rows[0]["metric_delta"]==[.25,0.,0.]
    assert rows[0]["distance_to_goal"]==.75
    assert rows[0]["timing"]=="post_action"
    assert "decision_observer" not in calls["run"][0]


def test_completed_episode_json_survives_next_episode_failure(tmp_path,monkeypatch):
    args,_=setup(tmp_path,monkeypatch,fail_second=True)
    out=tmp_path/"completed"
    assert cli.main(args+["--diagnostic","--episodes","2","--episode-indices","1","2","--episode-results-dir",str(out)])==2
    row=json.loads((out/"episode_000001.json").read_text())
    assert row["distance_to_goal"]==.75 and row["num_steps"]==1
    assert not (tmp_path/"result.json").exists()


def test_trace_never_overwrites_prior_evidence(tmp_path,monkeypatch):
    args,_=setup(tmp_path,monkeypatch)
    trace_dir=tmp_path/"traces";trace_dir.mkdir()
    path=trace_dir/"episode_000002.steps.jsonl";path.write_text("preserved\n")
    assert cli.main(args+["--diagnostic","--episode-indices","2","--step-trace-dir",str(trace_dir)])==2
    assert path.read_text()=="preserved\n"


def test_only_first_selected_episode_records_reset_and_post_step_frames(tmp_path,monkeypatch):
    args,calls=setup(tmp_path,monkeypatch)
    class Recorder:
        def __init__(self,path,**kwargs):self.path=Path(path);self.events=[];self.closed=False;calls["videos"].append(self)
        def __call__(self,event):self.events.append(event)
        def close(self):self.closed=True;return {"path":str(self.path)}
    monkeypatch.setattr(cli,"_video_recorder",Recorder,raising=False)
    assert cli.main(args+["--diagnostic","--episodes","2","--episode-indices","1","2", "--video-dir",str(tmp_path/"videos"),"--video-episodes","1","--video-fps","4"])==0
    assert len(calls["videos"])==1
    recorder=calls["videos"][0]
    assert recorder.path.name=="episode_000001.mp4" and recorder.closed
    assert [event["phase"] for event in recorder.events]==["reset","step"]
    assert [event["action"] for event in recorder.events]==[None,"FWD"]


def test_encoder_batch_size_reaches_real_factory(tmp_path,monkeypatch):
    args,calls=setup(tmp_path,monkeypatch)
    assert cli.main(args+['--diagnostic','--encoder-batch-size','7'])==0
    assert calls['factory'][0]['encoder_batch_size']==7


def test_model_batch_size_reaches_real_factory(tmp_path,monkeypatch):
    args,calls=setup(tmp_path,monkeypatch)
    assert cli.main(args+['--diagnostic','--model-batch-size','7'])==0
    assert calls['factory'][0]['model_batch_size']==7


def test_diagnostic_error_preserves_traceback(tmp_path,monkeypatch,capsys):
    args,calls=setup(tmp_path,monkeypatch,fail_second=True)
    assert cli.main(args+['--diagnostic','--episodes','2'])==2
    stderr=capsys.readouterr().err
    assert 'Traceback (most recent call last)' in stderr
    assert 'later episode failed' in stderr
