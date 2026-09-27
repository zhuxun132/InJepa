"""Physical sampling is admitted, identity-bound and reflected in update budgets."""
import copy
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(HERE/'source'))
from test_entry import chain
from test_physical_sampling import EchoReplay
import lwm_stream.entry as entry
from lwm_stream.dataset import StratifiedReplayDataset,PseudoDataset


def configuration(cfg):
    cfg=copy.deepcopy(cfg); cfg['sampling']={'strategy':'physical_stratified_cycle_v1','anchor_bin_size':3}
    cfg['wm'].update(global_batch=3,epochs=2,warmup_epochs=1)
    cfg['il'].update(global_batch=2,epochs=3,warmup_epochs=2)
    cfg['rl'].update(global_batch=2,epochs=4,warmup_epochs=1)
    return cfg


def partitions():
    def row(key,length,split='train'):
        return {'source':'r2r','video':f'{key}_r2r_0','split':split,'pose_key':key,
                'keyframe_indices':list(range(length+1)),'frame_names':[f'{i:03}.jpg' for i in range(length+1)]}
    return {'wm_il':[row('a',7),row('b',2)],'rl':[row('c',4)],'dev':[row('d',2,'dev')]}


def admitted_rows(cfg,values,representation=True):
    root=Path(cfg['codebooks']); status=json.loads((root/'STATUS.json').read_text())
    if representation: status['representation']='physical_trajectories_v2'
    for name,rows in values.items():
        path=root/f'{name}.jsonl'; path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        status['partitions'][name].update(rows=len(rows),physical_trajectories=len({r['pose_key'] for r in rows}),windows=sum(len(r['keyframe_indices'])-1 for r in rows),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (root/'STATUS.json').write_text(json.dumps(status))


def test_budget_counts_bins_global_updates_drop_last_and_real_warmup(chain):
    cfg=configuration(chain[0]); got=entry.sampling_budget(cfg,partitions())
    assert got['partitions']=={'wm_il':{'physical_trajectories':2,'all_anchors':9,'samples_per_epoch':4},'rl':{'physical_trajectories':1,'all_anchors':4,'samples_per_epoch':2},'dev':{'physical_trajectories':1,'all_anchors':2,'samples_per_epoch':1}}
    assert got['stages']=={'wm':{'samples_per_epoch':4,'updates_per_epoch':1,'total_updates':2,'warmup_updates':1,'dropped_samples_per_epoch':1},'il':{'samples_per_epoch':4,'updates_per_epoch':2,'total_updates':6,'warmup_updates':4,'dropped_samples_per_epoch':0},'rl':{'samples_per_epoch':2,'updates_per_epoch':1,'total_updates':4,'warmup_updates':1,'dropped_samples_per_epoch':0}}


def test_budget_rejects_unrecognized_strategy_or_invalid_bin(chain):
    for field,value in [('strategy','other'),('anchor_bin_size',0),('anchor_bin_size',True)]:
        cfg=configuration(chain[0]); cfg['sampling'][field]=value
        with pytest.raises(ValueError): entry.sampling_budget(cfg,partitions())

@pytest.mark.parametrize('mutation',['valid','representation','duplicate_physical','cross_half'])
def test_preflight_physical_representation_unique_halves_and_budget(chain,mutation):
    cfg=configuration(chain[0]); values=partitions()
    if mutation=='duplicate_physical': values['wm_il'][1]['pose_key']='a'
    elif mutation=='cross_half': values['rl'][0]['pose_key']='a'
    admitted_rows(cfg,values,representation=mutation!='representation')
    if mutation=='valid':
        result=entry.preflight(cfg,'wm')
        assert result['data_budget']==entry.sampling_budget(cfg,values)
    else:
        with pytest.raises(ValueError): entry.preflight(cfg,'wm')


def test_sampling_rules_bound_to_scientific_and_pseudo_identities(chain):
    cfg,hashes,complete,ready=chain; cfg=configuration(cfg)
    admission={'hashes':hashes,'predecessors':{'wm':{'sha256':complete['wm']['sha256']}},'pseudo':ready}
    scientific=entry.scientific_identity(cfg,'il',admission); pseudo=entry.pseudo_identity(cfg,admission)
    assert scientific['sampling']==pseudo['sampling']==cfg['sampling']
    changed=copy.deepcopy(cfg); changed['sampling']['anchor_bin_size']=2
    assert entry.scientific_identity(changed,'il',admission)!=scientific
    assert entry.pseudo_identity(changed,admission)!=pseudo


def test_build_dataset_wraps_pair_bins_before_fixed_il_cache(chain,monkeypatch):
    cfg,hashes,_,_=chain; cfg=configuration(cfg); values=partitions(); admitted_rows(cfg,values)
    monkeypatch.setattr(entry,'ReplayDataset',EchoReplay)
    admission={'hashes':hashes,'predecessors':{},'pseudo':None}
    pseudo=entry.build_dataset(cfg,'pseudo',admission)
    assert isinstance(pseudo,StratifiedReplayDataset) and len(pseudo)==4
    tokens=np.full((4,65),66,dtype=np.int16); tokens[:,:3]=[64,1,65]
    token_path=Path(cfg['output'])/'pseudo/tokens.npy'; np.save(token_path,tokens)
    admission['pseudo']={'sha256':hashlib.sha256(token_path.read_bytes()).hexdigest()}
    il=entry.build_dataset(cfg,'il',admission)
    assert isinstance(il,PseudoDataset) and isinstance(il.pairs,StratifiedReplayDataset) and len(il)==4
    for index in range(4):
        np.testing.assert_array_equal(il[(8,index)]['now'],pseudo[(0,index)]['now'])
