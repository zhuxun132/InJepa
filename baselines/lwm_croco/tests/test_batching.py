"""Actual global-batch sharding and deterministic DataLoader worker inputs."""
import os
import multiprocessing
import sys
from pathlib import Path
import pytest
import torch

HERE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(HERE/'source'))
from lwm_stream.batching import EpochBatches

class IndexedDataset(torch.utils.data.Dataset):
    def __len__(self): return 13
    def __getitem__(self,key):
        epoch,index=key
        return {'item':torch.tensor([epoch,index]),'pid':torch.tensor(os.getpid()),'start_method':multiprocessing.get_start_method()}


def make(rank=0,workers=0,shuffle=False,**overrides):
    cfg={'global_batch':5,'micro_batch':2,'rank':rank,'world_size':2,'seed':17,'workers':workers,'device':torch.device('cpu'),'shuffle':shuffle}
    cfg.update(overrides)
    return EpochBatches(IndexedDataset(),**cfg)


def flatten_update(update): return torch.cat([micro['item'] for micro in update],dim=0)


def test_rank_striding_unequal_micro_counts_disjoint_global_coverage_and_tail_drop():
    allranks=[]
    for rank in range(2):
        total,iterator=make(rank)(3,0); updates=list(iterator)
        assert total==2 and len(updates)==2
        assert [len(update) for update in updates]==([2,2] if rank==0 else [1,1])
        for u,update in enumerate(updates):
            expected=torch.tensor([[3,i] for i in range(5*u+rank,5*u+5,2)])
            torch.testing.assert_close(flatten_update(update),expected,rtol=0,atol=0)
        allranks.append(updates)
    for u in range(2):
        ids=[flatten_update(allranks[r][u])[:,1].tolist() for r in range(2)]
        assert not set(ids[0])&set(ids[1]) and sorted(ids[0]+ids[1])==list(range(u*5,u*5+5))


def test_offset_skips_full_global_updates_and_keeps_total_count():
    batches=make(rank=0)
    total,full=batches(4,0); full=list(full)
    resumed_total,resumed=batches(4,1); resumed=list(resumed)
    assert total==resumed_total==2 and len(resumed)==1
    torch.testing.assert_close(flatten_update(resumed[0]),flatten_update(full[1]),rtol=0,atol=0)
    empty_total,empty=batches(4,2)
    assert empty_total==2 and list(empty)==[]

@pytest.mark.parametrize('workers',[0,1])
def test_shuffle_seed_epoch_worker_indices_repeatable_and_global_rng_unchanged(workers):
    state=torch.get_rng_state().clone()
    parent_method=multiprocessing.get_start_method()
    batches=make(rank=0,workers=workers,shuffle=True)
    total,first=batches(2,0); first=list(first)
    _,second=batches(2,0); second=list(second)
    assert total==2 and torch.equal(torch.get_rng_state(),state)
    expected_order=torch.randperm(13,generator=torch.Generator().manual_seed(19))[:10].reshape(2,5)
    for i in range(2):
        got=flatten_update(first[i])
        torch.testing.assert_close(got[:,0],torch.full((3,),2),rtol=0,atol=0)
        torch.testing.assert_close(got[:,1],expected_order[i,::2],rtol=0,atol=0)
        torch.testing.assert_close(got,flatten_update(second[i]),rtol=0,atol=0)
    assert multiprocessing.get_start_method()==parent_method
    if workers:
        assert {method for update in first for micro in update for method in micro['start_method']}=={'spawn'}
    pids=torch.cat([micro['pid'] for update in first for micro in update])
    assert bool((pids!=os.getpid()).all()) if workers else bool((pids==os.getpid()).all())


def test_reject_world_size_larger_than_global_batch():
    with pytest.raises(ValueError): make(global_batch=1,world_size=2)
