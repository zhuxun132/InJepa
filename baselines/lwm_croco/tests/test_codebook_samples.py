"""Weighted physical sample geometry for the two original codebooks."""
import copy
import hashlib
import sys
from pathlib import Path
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(HERE/'source'))
from lwm_stream.dataset import codebook_samples

@pytest.fixture
def rows(tmp_path):
    path=tmp_path/'pose.npz'; p=np.zeros((6,3)); p[:,2]=[0,-.4,-1,-2,-2.7,-4]
    np.savez(path,positions=p,quaternions_xyzw=np.tile([0.,0.,0.,1.],(6,1)),state_index=np.arange(6))
    row={'source':'r2r','video':'sceneA_r2r_1','split':'train','pose_key':'physical-a','pose_path':str(path),'pose_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'keyframe_indices':[0,2,3,5],'rgb_dir':str(tmp_path/'does-not-exist')}
    return [row]


def test_full_trajectories_only_all_valid_deltas_no_rgb_and_no_input_change(rows):
    before=copy.deepcopy(rows); payload=Path(rows[0]['pose_path']).read_bytes()
    got=codebook_samples(rows,horizon=2)
    expected=np.array([[[1,0,0],[2,0,0]],[[1,0,0],[3,0,0]]],dtype=np.float32)
    np.testing.assert_array_equal(got['trajectories'],expected)
    np.testing.assert_array_equal(got['deltas'],[[1,0],[1,0],[1,0],[2,0],[2,0]])
    np.testing.assert_array_equal(got['delta_weights'],np.ones(5))
    np.testing.assert_array_equal(got['trajectory_weights'],np.ones(2))
    assert got['trajectories'].dtype==got['deltas'].dtype==np.float32
    assert got['delta_weights'].dtype==got['trajectory_weights'].dtype==np.float64
    assert rows==before and Path(rows[0]['pose_path']).read_bytes()==payload


def test_alias_weight_matches_expanded_duplicate_objective(rows):
    single=codebook_samples(rows,horizon=2)
    duplicate=copy.deepcopy(rows[0]); duplicate['source']='rxr'; duplicate['video']='sceneA_rxr_2'
    weighted=codebook_samples(rows+[duplicate],horizon=2)
    for sample_name,weight_name in [('deltas','delta_weights'),('trajectories','trajectory_weights')]:
        np.testing.assert_array_equal(weighted[sample_name],single[sample_name])
        np.testing.assert_array_equal(weighted[weight_name],single[weight_name]*2)
        x=single[sample_name].reshape(len(single[sample_name]),-1)
        centers=np.stack([np.full(x.shape[1],.3),np.full(x.shape[1],1.7)])
        squared=((x[:,None,:]-centers[None,:,:])**2).sum(-1).min(-1)
        weighted_objective=(squared*weighted[weight_name]).sum()
        expanded=np.repeat(x,2,axis=0)
        expanded_objective=((expanded[:,None,:]-centers[None,:,:])**2).sum(-1).min(-1).sum()
        np.testing.assert_allclose(weighted_objective,expanded_objective,rtol=1e-7,atol=1e-7)

@pytest.mark.parametrize('invalid',['dev','same_key_pose','same_key_frames','duplicate_alias','no_full_window'])
def test_codebook_rejects_leakage_identity_drift_or_no_full_window(rows,invalid):
    horizon=2
    if invalid=='dev': rows[0]['split']='dev'
    elif invalid=='no_full_window': horizon=4
    else:
        other=copy.deepcopy(rows[0]); rows.append(other)
        if invalid!='duplicate_alias': other['video']='sceneA_r2r_2'
        if invalid=='same_key_pose': other['pose_sha256']='0'*64
        elif invalid=='same_key_frames': other['keyframe_indices']=[0,2,5]
    with pytest.raises(ValueError): codebook_samples(rows,horizon=horizon)
