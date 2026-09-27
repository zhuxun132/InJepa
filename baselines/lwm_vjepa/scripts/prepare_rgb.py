"""Reuse published JPEGs and extract only missing original trajectories."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import time
import numpy as np
from lwm_stream.data import _identity, select_keyframes
from lwm_stream.dataset import index_trajectory


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


class HashingReader:
    def __init__(self, stream, sizes):
        self.stream=stream;self.hash=hashlib.sha256();self.bytes=0
        self.sizes=sizes;self.part_hashes=[hashlib.sha256() for _ in sizes]
        self.part=0;self.offset=0
    def read(self,n=-1):
        data=self.stream.read(n);self.hash.update(data);self.bytes+=len(data)
        cursor=0
        while cursor<len(data):
            if self.part>=len(self.sizes):raise ValueError('archive exceeds admitted part sizes')
            count=min(len(data)-cursor,self.sizes[self.part]-self.offset)
            self.part_hashes[self.part].update(data[cursor:cursor+count])
            cursor+=count;self.offset+=count
            if self.offset==self.sizes[self.part]:self.part+=1;self.offset=0
        return data


def prepare_source(source, configuration, jobs, output):
    root=Path(output)/source;root.mkdir(exist_ok=True)
    annotations=Path(configuration['annotations'])
    if sha(annotations)!=configuration['expected_annotation_sha256']:
        raise ValueError('original annotation checksum differs')
    records={r['video']:r for r in json.loads(annotations.read_text())}
    existing=Path(configuration['existing_images'])
    missing={};result=[];excluded=[]
    started=time.monotonic()
    for job in jobs:
        record=records[job['video']]
        if job['keyframe_count']<2:
            excluded.append(dict(video=job['video'],reason='no_future_keyframe'));continue
        rgb_dir=existing/Path(job['video']).name/'rgb'
        if rgb_dir.is_dir():
            index=index_trajectory(record,job['pose_path'],rgb_dir)
            result.append(dict(index,source=source,split=job['split'],pose_key=job['pose_key']))
        else:
            missing[Path(job['video']).name]=(record,job)
    (root/'MISSING.json').write_text(json.dumps(sorted(missing),indent=2)+'\n')
    print(json.dumps(dict(source=source,reused=len(result),missing=len(missing),excluded=len(excluded))),flush=True)
    archive_receipt=None;copied=0;copied_bytes=0
    if missing:
        parts=[str(Path(path)) for path in configuration['archive_parts']]
        proc=subprocess.Popen(['cat',*parts],stdout=subprocess.PIPE)
        sizes=[Path(path).stat().st_size for path in parts]
        reader=HashingReader(proc.stdout,sizes)
        try:
            with gzip.GzipFile(fileobj=reader,mode='rb') as decompressed:
                with tarfile.open(fileobj=decompressed,mode='r|') as archive:
                    for member in archive:
                        parts_in=PurePosixPath(member.name).parts
                        if len(parts_in)!=4 or parts_in[0]!='images' or parts_in[2]!='rgb':continue
                        video,name=parts_in[1],parts_in[3]
                        if video not in missing:continue
                        if not member.isfile() or re.fullmatch(r'[0-9]+\.jpg',name) is None:
                            raise ValueError('unexpected member in required original RGB directory')
                        directory=root/'images'/video/'rgb';directory.mkdir(parents=True,exist_ok=True)
                        target=directory/name
                        with archive.extractfile(member) as stream:data=stream.read()
                        if len(data)!=member.size:raise ValueError('truncated original JPEG')
                        if target.exists():
                            if target.read_bytes()!=data:raise ValueError('existing extracted JPEG differs')
                        else:
                            temporary=target.with_suffix('.tmp')
                            temporary.write_bytes(data);os.replace(temporary,target)
                        copied+=1;copied_bytes+=len(data)
                        if copied%20000==0:print(json.dumps(dict(source=source,extracted=copied,bytes=copied_bytes)),flush=True)
                # Read through gzip EOF to verify its CRC/trailer, not merely tar end blocks.
                while decompressed.read(4*1024*1024):pass
            if proc.wait()!=0:raise RuntimeError('original archive concatenation failed')
            expected=configuration['expected_archive_sha256']
            part_hashes=[h.hexdigest() for h in reader.part_hashes]
            actual=part_hashes if isinstance(expected,list) else reader.hash.hexdigest()
            if reader.bytes!=sum(sizes) or actual!=expected:
                raise ValueError('original compressed archive checksum differs')
            archive_receipt=dict(parts=configuration['archive_parts'],compressed_bytes=reader.bytes,
                                 compressed_sha256=reader.hash.hexdigest(),part_sha256=part_hashes,gzip_crc='PASS')
        finally:
            proc.stdout.close()
            if proc.poll() is None:proc.terminate();proc.wait()
        for video,(record,job) in missing.items():
            index=index_trajectory(record,job['pose_path'],root/'images'/video/'rgb')
            result.append(dict(index,source=source,split=job['split'],pose_key=job['pose_key']))
    result.sort(key=lambda row:row['video'])
    manifest=root/'INDEX.jsonl';temporary=root/'INDEX.tmp'
    with temporary.open('w') as f:
        for row in result:f.write(json.dumps(row,separators=(',',':'))+'\n')
    os.replace(temporary,manifest)
    receipt=dict(source=source,status='RGB_POSE_INDEX_READY',trajectories=len(result),excluded=excluded,
                 reused=len(result)-len(missing),extracted_trajectories=len(missing),extracted_jpegs=copied,
                 extracted_bytes=copied_bytes,archive=archive_receipt,annotation_sha256=sha(annotations),
                 index_sha256=sha(manifest),seconds=time.monotonic()-started)
    (root/'STATUS.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',type=Path,required=True)
    p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,required=True)
    a=p.parse_args()
    if a.workers<1:p.error('workers must be positive')
    configuration=json.loads(a.sources.read_text())
    if set(configuration)!={'r2r','rxr'}:p.error('exactly original r2r and rxr sources are required')
    original=json.loads((a.replay/'ADMISSION.json').read_text())
    for source,cfg in configuration.items():
        cfg['expected_annotation_sha256']=original['input_hashes'][cfg['annotations']]
    jobs=validate_replay_index(a.replay,configuration)
    admission=dict(replay_admission_sha256=sha(a.replay/'ADMISSION.json'),sources=configuration)
    a.output.mkdir(exist_ok=True)
    target=a.output/'ADMISSION.json'
    if target.exists():
        if json.loads(target.read_text())!=admission:raise ValueError('RGB output identity differs')
    else:
        if any(a.output.iterdir()):raise ValueError('RGB output contains unadmitted files')
        with target.open('x') as f:json.dump(admission,f,indent=2)
    receipts=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,len(configuration))) as pool:
        futures=[pool.submit(prepare_source,source,cfg,jobs[source],str(a.output)) for source,cfg in configuration.items()]
        for future in as_completed(futures):receipts.append(future.result())
    result=dict(status='RGB_POSE_INDEX_READY',sources=receipts,admission_sha256=sha(target))
    (a.output/'STATUS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

def validate_replay_index(replay_root, configuration):
    replay_root=Path(replay_root)
    admission=json.loads((replay_root/'ADMISSION.json').read_text())
    expected_records={}
    for source,cfg in configuration.items():
        path=Path(cfg['annotations'])
        if sha(path)!=admission['input_hashes'].get(str(path)):
            raise ValueError('annotation differs from original replay admission')
        if not cfg.get('expected_archive_sha256'):raise ValueError('original archive checksum is required')
        for record in json.loads(path.read_text()):
            key=(source,record['video'])
            if key in expected_records:raise ValueError('duplicate original annotation')
            expected_records[key]=record
    scene_splits={}
    for split in ('train','dev','final_mp3d'):
        for scene in admission['splits'][split]:
            if scene in scene_splits:raise ValueError('overlapping scene splits')
            scene_splits[scene]=split
    expected_keys=set(admission['job_keys'])
    if len(expected_keys)!=len(admission['job_keys']):raise ValueError('duplicate admitted pose keys')
    jobs={source:[] for source in configuration};seen_keys=set();seen_aliases=set()
    for done in sorted(replay_root.glob('*/DONE.json')):
        receipt=json.loads(done.read_text());scene=done.parent.name
        if receipt.get('scene')!=scene or receipt.get('status')!='POSE_ONLY_RGB_JOIN_PENDING':
            raise ValueError('invalid scene receipt')
        split=scene_splits.get(scene)
        if split not in ('train','dev'):raise ValueError('unadmitted replay scene')
        for row in receipt['files']:
            path=done.parent/row['path'];key=path.stem
            if row['path']!=key+'.npz' or key not in expected_keys or key in seen_keys:
                raise ValueError('pose key is unexpected or repeated')
            seen_keys.add(key)
            if row['split']!=split:raise ValueError('pose split differs from scene admission')
            if not path.is_file() or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
                raise ValueError('pose size/checksum differs')
            with np.load(path,allow_pickle=False) as poses:
                if len(poses['positions'])!=row['frames']:raise ValueError('pose frame count differs')
                n=len(select_keyframes(poses['positions']))
            for alias in row['aliases']:
                alias_key=(alias['source'],alias['video'])
                if alias_key in seen_aliases or alias_key not in expected_records:
                    raise ValueError('duplicate or unexpected replay alias')
                seen_aliases.add(alias_key);record=expected_records[alias_key]
                record_scene,record_source,record_id=_identity(record)
                if record_scene!=scene or record_source!=alias['source'] or record_id!=alias['id']:
                    raise ValueError('alias identity differs from replay scene/source')
                if alias['id']!=record['id'] or row['frames']!=len(record['actions']):
                    raise ValueError('alias id/frame count differs from original annotation')
                jobs[alias['source']].append(dict(video=alias['video'],split=split,
                    pose_key=key,pose_path=str(path),keyframe_count=n))
    if seen_keys!=expected_keys or seen_aliases!=set(expected_records) or len(seen_aliases)!=admission['aliases']:
        raise ValueError('replay receipt coverage differs from admission')
    return jobs


if __name__=='__main__':main()
