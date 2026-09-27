"""CPU-only original trajectory replay. Invoke with existing Habitat Python."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
import numpy as np
from lwm_stream.replay import NativeReplayBackend, replay_episode, group_replay_jobs


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def replay_scene(scene, jobs, assets, configuration, output):
    root=Path(output)/scene
    identity=digest([job['key'] for job in jobs])
    done=root/'DONE.json'
    if done.exists():
        receipt=json.loads(done.read_text())
        if receipt['identity']!=identity: raise ValueError('existing scene identity differs')
        if receipt.get('scene')!=scene or receipt.get('status')!='POSE_ONLY_RGB_JOIN_PENDING':
            raise ValueError('existing scene receipt metadata differs')
        expected={job['key']+'.npz':job for job in jobs}
        actual=receipt.get('files',[])
        if len(actual)!=len(expected) or {row['path'] for row in actual}!=set(expected):
            raise ValueError('existing scene receipt does not cover every job exactly once')
        for row in receipt['files']:
            job=expected[row['path']]
            if (row.get('frames')!=len(job['row']['record']['actions']) or
                    row.get('aliases')!=job['aliases'] or row.get('split')!=job['split']):
                raise ValueError('existing pose metadata differs')
            path=root/row['path']
            if not path.is_file() or path.stat().st_size!=row.get('bytes'):
                raise ValueError('existing pose is missing or has wrong size')
            if file_sha(path)!=row['sha256']: raise ValueError('existing pose checksum differs')
        return receipt
    root.mkdir(exist_ok=True)
    backend=NativeReplayBackend(Path(assets)/scene/(scene+'.glb'),**configuration)
    files=[]; start=time.monotonic()
    try:
        for job in jobs:
            row=job['row']; n=len(row['record']['actions'])
            # Temporary numeric names validate action count only, not JPEG existence.
            result=replay_episode(backend,row['record'],row['episode'],
                [str(i)+'.jpg' for i in range(n)],dataset_source=row['source'])
            target=root/(job['key']+'.npz')
            temporary=root/(job['key']+'.tmp.npz')
            np.savez(temporary,positions=result['positions'],quaternions_xyzw=result['quaternions_xyzw'],
                     state_index=np.arange(n,dtype=np.int64))
            os.replace(temporary,target)
            files.append(dict(path=target.name,sha256=file_sha(target),frames=n,bytes=target.stat().st_size,
                              aliases=job['aliases'],split=job['split']))
    finally:backend.close()
    receipt=dict(scene=scene,identity=identity,files=files,seconds=time.monotonic()-start,
                 status='POSE_ONLY_RGB_JOIN_PENDING')
    temporary=root/'DONE.tmp'
    temporary.write_text(json.dumps(receipt,indent=2)+'\n');os.replace(temporary,done)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations-root',type=Path,required=True)
    p.add_argument('--metadata-root',type=Path,required=True)
    p.add_argument('--assets-manifest',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--configuration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,required=True)
    a=p.parse_args()
    if a.workers<1: p.error('workers must be positive')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':p.error('set CUDA_VISIBLE_DEVICES empty for CPU replay')
    cfg=json.loads(a.configuration.read_text())
    if cfg.get('render_rgb') is not False: p.error('configuration must disable rendering')
    import habitat_sim
    import lwm_stream.replay as replay_module
    runtime=dict(habitat_version=habitat_sim.__version__,configuration=cfg,
                 replay_sha256=file_sha(replay_module.__file__))
    scene_parts={}
    for asset in json.loads(a.assets_manifest.read_text())['files']:
        path=a.assets/asset['scene']/Path(asset['path']).name
        if file_sha(path)!=asset['sha256']: raise ValueError('asset differs from manifest')
        scene_parts.setdefault(asset['scene'],[]).append((path.name,asset['sha256']))
    scene_ids={scene:digest(sorted(parts)) for scene,parts in scene_parts.items()}
    census=json.loads((a.metadata_root/'BUILDING_SPLIT_REPLAY_CENSUS.json').read_text())
    splits={name:census[name] for name in ['train','dev','final_mp3d']}
    rows=[]; input_hashes={}
    for source,filename in [('r2r','annotations_v1-3.json'),('rxr','annotations.json')]:
        annotation=a.annotations_root/({'r2r':'R2R','rxr':'RxR'}[source])/filename
        pose=a.metadata_root/(source.upper()+'_MATCHED_TRAIN_POSES.json')
        input_hashes[str(annotation)]=file_sha(annotation);input_hashes[str(pose)]=file_sha(pose)
        episodes=json.loads(pose.read_text())['episodes']
        mapping={str(ep['episode_id']):ep for ep in episodes}
        if len(mapping)!=len(episodes):raise ValueError('duplicate original episode id')
        for record in json.loads(annotation.read_text()):
            rows.append(dict(source=source,record=record,episode=mapping[str(record['id'])]))
    jobs=group_replay_jobs(rows,runtime_identity=digest(runtime),scene_identities=scene_ids,splits=splits)
    admission=dict(runtime=runtime,input_hashes=input_hashes,scene_identities=scene_ids,splits=splits,
                   job_keys=[job['key'] for job in jobs],aliases=sum(len(j['aliases']) for j in jobs))
    a.output.mkdir(exist_ok=True)
    admission_path=a.output/'ADMISSION.json'
    if admission_path.exists():
        if json.loads(admission_path.read_text())!=admission:raise ValueError('output admission identity differs')
    else:
        if any(a.output.iterdir()):raise ValueError('output contains unadmitted files')
        with admission_path.open('x') as f:json.dump(admission,f,indent=2)
    scenes={}
    for job in jobs:scenes.setdefault(job['scene'],[]).append(job)
    started=time.monotonic();receipts=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(replay_scene,s,j,str(a.assets),cfg,str(a.output)):s for s,j in scenes.items()}
        for future in as_completed(futures):
            receipt=future.result();receipts.append(receipt)
            print(json.dumps(dict(scene=receipt['scene'],done=len(receipts),total=len(scenes),
                                  groups=len(receipt['files']),seconds=receipt['seconds'])),flush=True)
    result=dict(status='POSE_ONLY_RGB_JOIN_PENDING',scenes=len(receipts),groups=len(jobs),
                aliases=len(rows),frames=sum(r['frames'] for s in receipts for r in s['files']),
                bytes=sum(r['bytes'] for s in receipts for r in s['files']),
                workers=a.workers,seconds=time.monotonic()-started,admission_sha256=file_sha(admission_path))
    (a.output/'STATUS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
