"""Admitted physical splits and actual image/frame loader geometry."""
import copy
import hashlib
import random
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

HERE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(HERE/'source')); sys.path.insert(0,str(HERE/'source/official'))
from lwm.preprocess import get_image_transform
from lwm_stream.dataset import partition_rows,ReplayDataset


@pytest.fixture(autouse=True)
def cpu_threads():
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)

def rows_for_split():
    return [{'source':'r2r','video':f'scene_r2r_{i}','split':'train','pose_key':key,'keyframe_indices':[0,2]} for i,key in enumerate(['a','b','c','d','a'])]+[{'source':'rxr','video':'other_rxr_0','split':'dev','pose_key':'e','keyframe_indices':[0,1]}]


def test_partition_physical_aliases_half_groups_stable_and_input_immutable():
    rows=rows_for_split(); before=copy.deepcopy(rows)
    got=partition_rows(rows,seed=12); reordered=partition_rows(list(reversed(rows)),seed=12)
    assert got==reordered and rows==before
    groupkeys={name:{r['pose_key'] for r in got[name]} for name in got}
    assert len(groupkeys['wm_il'])==len(groupkeys['rl'])==2
    assert not groupkeys['wm_il']&groupkeys['rl'] and groupkeys['dev']=={'e'}
    assert sum(r['pose_key']=='a' for r in got['wm_il']) in (0,2)
    for values in got.values(): assert [(r['source'],r['video']) for r in values]==sorted((r['source'],r['video']) for r in values)
    got['wm_il'][0]['keyframe_indices'][0]=99
    assert rows==before

@pytest.mark.parametrize('failure',['cross_split_key','duplicate_alias'])
def test_partition_rejects_leakage_or_duplicate_identity(failure):
    rows=rows_for_split()
    if failure=='cross_split_key': rows[-1]['pose_key']='a'
    else: rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError): partition_rows(rows,seed=12)

@pytest.fixture
def actual_rows(tmp_path):
    result=[]
    for key,z,keys,color in [('a',[0,-.1,-1,-1.3,-3],[0,2,4],0),('b',[0,-10,-30],[0,1,2],30)]:
        rgb=tmp_path/key; rgb.mkdir()
        names=[]
        for i in range(len(z)):
            name=f'{i+1:03}.jpg'; names.append(name)
            Image.new('RGB',(32,24),(color+i*35,20+i*10,90)).save(rgb/name,quality=95)
        path=tmp_path/f'{key}.npz'; positions=np.zeros((len(z),3)); positions[:,2]=z
        np.savez(path,positions=positions,quaternions_xyzw=np.tile([0.,0.,0.,1.],(len(z),1)),state_index=np.arange(len(z),dtype=np.int64))
        result.append({'source':'r2r','video':f'scene_{key}_r2r_0','split':'train','pose_key':key,'rgb_dir':str(rgb),'frame_names':names,'pose_path':str(path),'pose_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'keyframe_indices':keys})
    alias=copy.deepcopy(result[0]); alias['source']='rxr'; alias['video']='scene_a_rxr_1'
    return [result[0],alias,result[1]]


def transformed(row,index):
    with Image.open(Path(row['rgb_dir'])/row['frame_names'][index]) as image:
        return get_image_transform()(image.convert('RGB'))


def test_wm_actual_frames_factual0_unscaled_donors_and_independent_masks(actual_rows):
    before=copy.deepcopy(actual_rows)
    dataset=ReplayDataset(actual_rows,mode='wm',seed=18,num_candidates=3,max_future=4,pose_cache_size=1)
    assert len(dataset)==6
    sample=dataset[0]
    assert set(sample)=={'now','goal','actions_m','valid_mask','goal_xy_m'}
    torch.testing.assert_close(sample['now'],transformed(actual_rows[0],0),rtol=0,atol=0)
    actions=np.asarray(sample['actions_m']); mask=np.asarray(sample['valid_mask'])
    assert actions.shape==(3,4,3) and actions.dtype==np.float32 and mask.dtype==bool
    np.testing.assert_array_equal(actions[0],[[1,0,0],[3,0,0],[0,0,0],[0,0,0]])
    np.testing.assert_array_equal(mask[0],[True,True,False,False])
    for m in (1,2):
        valid=actions[m,mask[m],0].tolist()
        assert valid in ([10.,30.],[20.]) # excludes same-physical alias and .1-scaled geometry
        np.testing.assert_array_equal(mask[m],np.arange(4)<len(valid))
        np.testing.assert_array_equal(actions[m,~mask[m]],np.zeros((4-len(valid),3)))
    matching=[(frame,xy) for frame,xy in [(2,[1,0]),(4,[3,0])] if torch.equal(sample['goal'],transformed(actual_rows[0],frame))]
    assert len(matching)==1
    np.testing.assert_array_equal(sample['goal_xy_m'],matching[0][1])
    assert actual_rows==before


def test_flat_alias_anchor_maps_original_now_frame_and_pair_outputs(actual_rows):
    dataset=ReplayDataset(actual_rows,mode='pair',seed=18,max_future=4)
    sample=dataset[3] # row1 alias, second anchor: original frame2 -> frame4
    assert set(sample)=={'now','goal'}
    torch.testing.assert_close(sample['now'],transformed(actual_rows[1],2),rtol=0,atol=0)
    torch.testing.assert_close(sample['goal'],transformed(actual_rows[1],4),rtol=0,atol=0)


def test_epoch_index_rng_is_local_and_exactly_repeatable(actual_rows):
    dataset=ReplayDataset(actual_rows,mode='wm',seed=7,num_candidates=3,max_future=4)
    py=random.getstate(); npstate=np.random.get_state(); ts=torch.get_rng_state().clone()
    a=dataset[(2,0)]; _=dataset[(5,4)]; b=dataset[(2,0)]
    for key in a: np.testing.assert_array_equal(a[key],b[key])
    for key in dataset[0]: np.testing.assert_array_equal(dataset[0][key],dataset[(0,0)][key])
    assert random.getstate()==py and torch.equal(torch.get_rng_state(),ts)
    after=np.random.get_state(); assert after[0]==npstate[0] and after[2:]==npstate[2:]
    np.testing.assert_array_equal(after[1],npstate[1])


def test_pose_hash_mismatch_rejected_on_first_use(actual_rows):
    actual_rows[0]['pose_sha256']='0'*64
    with pytest.raises(ValueError):
        dataset=ReplayDataset(actual_rows,mode='pair',seed=18)
        dataset[0]
