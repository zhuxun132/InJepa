"""Existing numeric RGB files join exact replay state indices."""
import copy
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'source'))
from lwm_stream.dataset import index_trajectory


def fixture(tmp_path,base=1):
    record={'id':4,'video':'sceneA_r2r_4','actions':[-1,1,2,1]}
    rgb=tmp_path/'rgb'; rgb.mkdir()
    for i in reversed(range(base,base+4)): (rgb/f'{i:03}.jpg').write_bytes(f'original-byte-{i}'.encode())
    pose=tmp_path/'poses.npz'
    arrays={'positions':np.array([[0,0,0],[.1,0,0],[.25,0,0],[.5,0,0]],dtype=np.float64),'quaternions_xyzw':np.tile([0.,0.,0.,1.],(4,1)),'state_index':np.arange(4,dtype=np.int64)}
    np.savez(pose,**arrays)
    return record,pose,rgb,arrays

@pytest.mark.parametrize('base',[0,1])
def test_actual_rgb_sort_pose_alignment_keyframes_hash_and_unchanged(tmp_path,base):
    record,pose,rgb,_=fixture(tmp_path,base)
    before=copy.deepcopy(record); pose_bytes=pose.read_bytes(); rgb_bytes={p.name:p.read_bytes() for p in rgb.iterdir()}
    result=index_trajectory(record,pose,rgb)
    assert result['video']==record['video']
    assert Path(result['rgb_dir'])==rgb and Path(result['pose_path'])==pose
    assert result['frame_names']==[f'{i:03}.jpg' for i in range(base,base+4)]
    assert result['keyframe_indices']==[0,2,3]
    assert result['pose_sha256']==hashlib.sha256(pose_bytes).hexdigest()
    json.dumps(result,allow_nan=False)
    assert record==before and pose.read_bytes()==pose_bytes
    assert {p.name:p.read_bytes() for p in rgb.iterdir()}==rgb_bytes

@pytest.mark.parametrize('mutation',['missing_rgb','pose_count','state_order','nan','zero_quaternion'])
def test_index_rejects_misalignment_or_invalid_pose(tmp_path,mutation):
    record,pose,rgb,arrays=fixture(tmp_path)
    if mutation=='missing_rgb': (rgb/'004.jpg').unlink()
    elif mutation=='pose_count': arrays['positions']=arrays['positions'][:-1]
    elif mutation=='state_order': arrays['state_index']=np.array([0,1,3,2])
    elif mutation=='nan': arrays['positions'][1,0]=np.nan
    else: arrays['quaternions_xyzw'][1]=0
    np.savez(pose,**arrays)
    with pytest.raises(ValueError): index_trajectory(record,pose,rgb)


def test_no_future_keeps_original_index_for_caller_exclusion(tmp_path):
    record,pose,rgb,arrays=fixture(tmp_path)
    arrays['positions'][:]=0; np.savez(pose,**arrays)
    result=index_trajectory(record,pose,rgb)
    assert result['keyframe_indices']==[0] and len(result['frame_names'])==4
