"""Original annotation/archive and complete replay admission before RGB READY."""
import copy
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(HERE/'source'))
spec=importlib.util.spec_from_file_location('prepare_rgb_test_target',HERE/'scripts/prepare_rgb.py')
prepare=importlib.util.module_from_spec(spec); spec.loader.exec_module(prepare)

def pose(path):
    np.savez(path,positions=np.array([[0.,0.,0.],[.25,0.,0.]]),quaternions_xyzw=np.tile([0.,0.,0.,1.],(2,1)),state_index=np.arange(2,dtype=np.int64))

def record(source): return {'id':1,'video':f'sceneA_{source}_1','actions':[-1,1]}

@pytest.mark.parametrize('mutation',['valid','key','aliases','split','annotation_sha','frames','scene_alias'])
def test_validate_replay_index_exact_identity_and_coverage(tmp_path,mutation):
    replay=tmp_path/'replay'; scene=replay/'sceneA'; scene.mkdir(parents=True)
    cfg={}; hashes={}; aliases=[]
    for source in ('r2r','rxr'):
        annotation=tmp_path/f'{source}.json'; annotation.write_text(json.dumps([record(source)]))
        cfg[source]={'annotations':str(annotation),'expected_annotation_sha256':prepare.sha(annotation),'expected_archive_sha256':'a'*64}
        hashes[str(annotation)]=prepare.sha(annotation)
        aliases.append({'source':source,'video':record(source)['video'],'id':1})
    key='c'*64; path=scene/(key+'.npz'); pose(path)
    admission={'job_keys':[key],'aliases':2,'input_hashes':hashes,'splits':{'train':['sceneA'],'dev':[],'final_mp3d':[]}}
    file={'path':path.name,'sha256':prepare.sha(path),'bytes':path.stat().st_size,'frames':2,'aliases':aliases,'split':'train'}
    receipt={'scene':'sceneA','status':'POSE_ONLY_RGB_JOIN_PENDING','files':[file],'identity':hashlib.sha256(json.dumps([key],separators=(',',':')).encode()).hexdigest()}
    if mutation=='key': admission['job_keys']=['d'*64]
    elif mutation=='aliases': file['aliases'][1]=copy.deepcopy(file['aliases'][0])
    elif mutation=='split': file['split']='dev'
    elif mutation=='annotation_sha':
        annotation=Path(cfg['r2r']['annotations']); annotation.write_text(annotation.read_text()+'\n')
    elif mutation=='frames': file['frames']=3
    elif mutation=='scene_alias':
        moved=replay/'sceneB'; scene.rename(moved); scene=moved
        receipt['scene']='sceneB'; admission['splits']['train']=['sceneB']
    (replay/'ADMISSION.json').write_text(json.dumps(admission)); (scene/'DONE.json').write_text(json.dumps(receipt))
    if mutation=='valid':
        jobs=prepare.validate_replay_index(replay,cfg)
        assert set(jobs)=={'r2r','rxr'}
        for source in jobs:
            assert len(jobs[source])==1
            job=jobs[source][0]
            assert job['video']==record(source)['video'] and job['split']=='train'
            assert job['pose_key']==key and Path(job['pose_path'])==path and job['keyframe_count']==2
    else:
        with pytest.raises(ValueError): prepare.validate_replay_index(replay,cfg)

@pytest.mark.parametrize('mutation',['valid','annotation_sha','archive_sha','split_valid','split_bad_part'])
def test_prepare_source_verifies_published_hashes_before_index_ready(tmp_path,mutation):
    annotation=tmp_path/'annotation.json'; annotation.write_text(json.dumps([record('r2r')]))
    trajectory=record('r2r')['video']; archive=tmp_path/'original.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for i in (1,2):
            data=f'original JPEG bytes {i}'.encode(); member=tarfile.TarInfo(f'images/{trajectory}/rgb/{i:03}.jpg'); member.size=len(data)
            tar.addfile(member,io.BytesIO(data))
    path=tmp_path/'pose.npz'; pose(path)
    parts=[str(archive)]; expected=prepare.sha(archive)
    if mutation.startswith('split_'):
        raw=archive.read_bytes(); cut=len(raw)//2; parts=[]; expected=[]
        for i,chunk in enumerate((raw[:cut],raw[cut:])):
            part=tmp_path/f'part{i}'; part.write_bytes(chunk); parts.append(str(part)); expected.append(prepare.sha(part))
        if mutation=='split_bad_part': expected[1]='0'*64
    cfg={'annotations':str(annotation),'existing_images':str(tmp_path/'absent'),'archive_parts':parts,'expected_annotation_sha256':prepare.sha(annotation),'expected_archive_sha256':expected}
    if mutation=='annotation_sha': cfg['expected_annotation_sha256']='0'*64
    elif mutation=='archive_sha': cfg['expected_archive_sha256']='0'*64
    jobs=[{'video':trajectory,'keyframe_count':2,'pose_path':str(path),'split':'train','pose_key':'c'*64}]
    output=tmp_path/'output'; output.mkdir()
    if mutation in ('valid','split_valid'):
        receipt=prepare.prepare_source('r2r',cfg,jobs,output)
        assert receipt['status']=='RGB_POSE_INDEX_READY' and receipt['trajectories']==1
        assert (output/'r2r/INDEX.jsonl').is_file()
    else:
        with pytest.raises(ValueError): prepare.prepare_source('r2r',cfg,jobs,output)
        assert not (output/'r2r/INDEX.jsonl').exists()
        assert not (output/'r2r/STATUS.json').exists()
