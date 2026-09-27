"""Physical aliases are provenance; stratified cycles set explicit sample budget."""
import copy
import hashlib
import pickle
import random
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(HERE/'source'))
from test_dataset import actual_rows
from lwm_stream.dataset import physical_rows,StratifiedReplayDataset,ReplayDataset,PseudoDataset,codebook_samples

@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def same_source_rows(actual_rows):
    rows=copy.deepcopy(actual_rows); rows[1]['source']='r2r'; rows[1]['video']='scene_a_r2r_1'
    return rows


def test_physical_representative_aliases_order_stable_and_independent(actual_rows):
    rows=same_source_rows(actual_rows); before=copy.deepcopy(rows)
    got=physical_rows(rows); reverse=physical_rows(list(reversed(rows)))
    assert got==reverse and [r['pose_key'] for r in got]==['a','b']
    assert got[0]['video']=='scene_a_r2r_0'
    assert got[0]['source_aliases']==[{'source':'r2r','video':'scene_a_r2r_0'},{'source':'r2r','video':'scene_a_r2r_1'}]
    got[0]['keyframe_indices'][0]=99; got[0]['source_aliases'][0]['video']='changed'
    assert rows==before


def test_physical_conflicting_source_split_pose_frames_and_duplicate_alias_rejected(actual_rows):
    for change in ('source','split','pose_sha256','frame_names','keyframe_indices','duplicate_alias'):
        rows=same_source_rows(actual_rows)
        if change=='source': rows[1][change]='rxr'
        elif change=='split': rows[1][change]='dev'
        elif change=='pose_sha256': rows[1][change]='0'*64
        elif change=='frame_names': rows[1][change]=rows[1][change][:-1]
        elif change=='keyframe_indices': rows[1][change]=[0,4]
        else: rows[1]['video']=rows[0]['video']
        with pytest.raises(ValueError): physical_rows(rows)


def test_dedup_codebook_weights_one_not_source_alias_multiplicity(actual_rows):
    dedup=physical_rows(same_source_rows(actual_rows))
    assert len(dedup[0]['source_aliases'])==2
    samples=codebook_samples(dedup,horizon=2)
    np.testing.assert_array_equal(samples['delta_weights'],np.ones(len(samples['deltas'])))
    np.testing.assert_array_equal(samples['trajectory_weights'],np.ones(len(samples['trajectories'])))

class EchoReplay(ReplayDataset):
    def __getitem__(self,item):
        epoch,index=item
        return {'now':torch.tensor([epoch,index]),'goal':torch.tensor([epoch,index+1])}


def echo(lengths,order=None):
    rows=[]
    for key,length in lengths:
        rows.append({'source':'r2r','video':f'{key}_r2r_0','split':'train','pose_key':key,
                     'frame_names':[f'{i:03}.jpg' for i in range(length+1)],'keyframe_indices':list(range(length+1))})
    if order is not None: rows=[rows[i] for i in order]
    return EchoReplay(rows,mode='pair',seed=5)


def test_bins_short_tail_and_cycles_exact_anchor_coverage():
    base=echo([('long',8),('short',2)])
    dataset=StratifiedReplayDataset(base,anchor_bin_size=3,seed=17)
    assert len(dataset)==4
    for index,(start,width) in enumerate([(0,3),(3,3),(6,2),(8,2)]):
        for cycle in range(2):
            picks=[dataset.anchor_index(cycle*width+e,index) for e in range(width)]
            assert sorted(picks)==list(range(start,start+width))
            for epoch,pick in enumerate(picks,start=cycle*width):
                torch.testing.assert_close(dataset[(epoch,index)]['now'],torch.tensor([epoch,pick]))


def test_width63_first50_are_distinct_and_full_cycle_covers_all():
    dataset=StratifiedReplayDataset(echo([('long',63)]),anchor_bin_size=63,seed=9)
    chosen=[dataset.anchor_index(epoch,0) for epoch in range(63)]
    assert len(set(chosen[:50]))==50 and sorted(chosen)==list(range(63))


def physical_choices(dataset,base,epoch):
    result={}
    for sample in range(len(dataset)):
        flat=dataset.anchor_index(epoch,sample)
        row=int(np.searchsorted(base._offsets,flat,side='right')-1)
        local=flat-base._offsets[row]
        result[(base.rows[row]['pose_key'],local//3)]=local
    return result


def test_same_physical_bin_draw_does_not_change_with_row_order():
    first=echo([('a',8),('b',5)]); second=echo([('a',8),('b',5)],order=[1,0])
    ds1=StratifiedReplayDataset(first,anchor_bin_size=3,seed=7)
    ds2=StratifiedReplayDataset(second,anchor_bin_size=3,seed=7)
    for epoch in range(7): assert physical_choices(ds1,first,epoch)==physical_choices(ds2,second,epoch)


def test_cycle_rng_is_local_and_pickle_retains_identity():
    py=random.getstate(); ns=np.random.get_state(); ts=torch.get_rng_state().clone()
    ds=StratifiedReplayDataset(echo([('a',8),('b',2)]),anchor_bin_size=3,seed=13)
    expected=[ds.anchor_index(e,i) for e in range(9) for i in range(len(ds))]
    restored=pickle.loads(pickle.dumps(ds))
    assert expected==[restored.anchor_index(e,i) for e in range(9) for i in range(len(restored))]
    assert py==random.getstate() and torch.equal(ts,torch.get_rng_state())
    now=np.random.get_state(); assert ns[0]==now[0] and ns[2:]==now[2:]; np.testing.assert_array_equal(ns[1],now[1])


def test_il_pseudo_pairs_remain_fixed_epoch0_with_rotating_bins(actual_rows,tmp_path):
    base=ReplayDataset(physical_rows(same_source_rows(actual_rows)),mode='pair',seed=21)
    rotating=StratifiedReplayDataset(base,anchor_bin_size=2,seed=17)
    values=np.full((len(rotating),65),66,dtype=np.int16); values[:,:3]=[64,1,65]
    path=tmp_path/'labels.npy'; np.save(path,values)
    il=PseudoDataset(rotating,path,expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for index in range(len(rotating)):
        expected=rotating[(0,index)]
        for epoch in (0,1,7):
            actual=il[(epoch,index)]
            torch.testing.assert_close(actual['now'],expected['now'],rtol=0,atol=0)
            torch.testing.assert_close(actual['goal'],expected['goal'],rtol=0,atol=0)
