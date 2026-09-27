"""Completed replay receipt must cover every expected job exactly once."""
import copy
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'source'))
spec=importlib.util.spec_from_file_location('lwm_prepare_replay_test_target',HERE/'scripts/prepare_replay.py')
prepare=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)

@pytest.mark.parametrize('mutation',['valid','missing_entry','duplicate_filename','missing_file','frames','aliases','split','scene','status'])
def test_completed_scene_receipt_requires_exact_coverage_and_metadata(tmp_path,monkeypatch,mutation):
    scene='sceneA'; root=tmp_path/scene; root.mkdir()
    jobs=[]; files=[]
    for index,char in enumerate(('a','b')):
        key=char*64; aliases=[{'source':'r2r','video':f'sceneA_r2r_{index}'}]
        job={'key':key,'scene':scene,'split':'train','aliases':aliases,'row':{'source':'r2r','record':{'actions':[-1,1,2]}}}
        jobs.append(job)
        path=root/(key+'.npz')
        np.savez(path,positions=np.zeros((3,3),dtype=np.float64),quaternions_xyzw=np.tile([0.,0.,0.,1.],(3,1)),state_index=np.arange(3,dtype=np.int64))
        files.append({'path':path.name,'sha256':prepare.file_sha(path),'bytes':path.stat().st_size,'frames':3,'aliases':copy.deepcopy(aliases),'split':'train'})
    receipt={'scene':scene,'identity':prepare.digest([j['key'] for j in jobs]),'files':files,'seconds':1.,'status':'POSE_ONLY_RGB_JOIN_PENDING'}
    if mutation=='missing_entry': receipt['files']=[] # reported P1: nonempty admitted jobs falsely complete
    elif mutation=='duplicate_filename': receipt['files']=[copy.deepcopy(files[0]),copy.deepcopy(files[0])]
    elif mutation=='missing_file': (root/files[1]['path']).unlink()
    elif mutation=='frames': files[0]['frames']=4
    elif mutation=='aliases': files[0]['aliases']=[{'source':'rxr','video':'unrelated'}]
    elif mutation=='split': files[0]['split']='dev'
    elif mutation=='scene': receipt['scene']='another_scene'
    elif mutation=='status': receipt['status']='DATA_READY'
    (root/'DONE.json').write_text(json.dumps(receipt))
    def forbidden(*args,**kwargs): pytest.fail('completed receipt validation must not create Habitat')
    monkeypatch.setattr(prepare,'NativeReplayBackend',forbidden)
    if mutation=='valid':
        assert prepare.replay_scene(scene,jobs,'unused-assets',{},tmp_path)==receipt
    else:
        with pytest.raises(ValueError): prepare.replay_scene(scene,jobs,'unused-assets',{},tmp_path)
