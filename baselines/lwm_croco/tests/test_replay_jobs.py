"""Exact physical replay deduplication, never approximate geometry grouping."""
import copy
import sys
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'source'))
from lwm_stream.replay import group_replay_jobs


def row(eid=1,source='r2r',scene='sceneA'):
    return {'source':source,'record':{'id':eid,'video':f'{scene}_{source}_{eid}','actions':[-1,1,2],'instructions':['different text']},'episode':{'episode_id':str(eid),'scene_id':f'mp3d/{scene}/{scene}.glb','start_position':[1.,2.,3.],'start_rotation':[0.,0.,0.,1.]}}


def group(rows,**changes):
    kwargs={'runtime_identity':'runtime-config-sha','scene_identities':{'sceneA':'asset-a','sceneB':'asset-b','sceneFinal':'asset-f'},'splits':{'train':['sceneA'],'dev':['sceneB'],'final_mp3d':['sceneFinal']}}
    kwargs.update(changes)
    return group_replay_jobs(rows,**kwargs)


def signatures(jobs):
    return [(j['key'],[(a['source'],a['video']) for a in j['aliases']]) for j in jobs]


def test_exact_physical_duplicates_share_cross_source_without_losing_aliases():
    rows=[row(2,'rxr'),row(1,'r2r')]; rows[0]['record']['instructions']=['unrelated instruction']
    before=copy.deepcopy(rows)
    jobs=group(rows)
    assert len(jobs)==1 and jobs[0]['scene']=='sceneA' and jobs[0]['split']=='train'
    assert [(a['source'],a['video']) for a in jobs[0]['aliases']]==[('r2r','sceneA_r2r_1'),('rxr','sceneA_rxr_2')]
    assert jobs[0]['row']==rows[0] and jobs[0]['row'] is not rows[0]
    assert rows==before
    reverse=group(list(reversed(rows)))
    assert signatures(jobs)==signatures(reverse)
    assert len(jobs[0]['key'])==64 and int(jobs[0]['key'],16)>=0
    jobs[0]['row']['record']['actions'][1]=3
    assert rows==before

@pytest.mark.parametrize('change',['action','position','quaternion_sign','scene'])
def test_physical_identity_differences_never_share(change):
    first,second=row(1),row(2)
    if change=='action': second['record']['actions'][2]=3
    elif change=='position': second['episode']['start_position'][0]+=1e-10
    elif change=='quaternion_sign': second['episode']['start_rotation']=[0.,0.,0.,-1.]
    else: second=row(2,scene='sceneB')
    jobs=group([first,second])
    assert len(jobs)==2 and len({j['key'] for j in jobs})==2
    assert [j['key'] for j in jobs]==sorted(j['key'] for j in jobs)
    if change=='scene': assert {j['scene']:j['split'] for j in jobs}=={'sceneA':'train','sceneB':'dev'}


def test_runtime_and_scene_asset_each_bind_job_key():
    original=group([row()])[0]['key']
    runtime=group([row()],runtime_identity='changed-runtime')[0]['key']
    asset=group([row()],scene_identities={'sceneA':'changed-asset'})[0]['key']
    assert len({original,runtime,asset})==3

@pytest.mark.parametrize('failure',['duplicate_alias','source_mismatch','final_scene','unknown_scene','split_overlap'])
def test_invalid_alias_source_or_split_rejected(failure):
    rows=[row()]; extra={}
    if failure=='duplicate_alias': rows.append(copy.deepcopy(rows[0]))
    elif failure=='source_mismatch': rows[0]['source']='rxr'
    elif failure=='final_scene': rows=[row(scene='sceneFinal')]
    elif failure=='unknown_scene': rows=[row(scene='unknown')]
    else: extra['splits']={'train':['sceneA'],'dev':['sceneA'],'final_mp3d':['sceneFinal']}
    before=copy.deepcopy(rows)
    with pytest.raises(ValueError): group(rows,**extra)
    assert rows==before
