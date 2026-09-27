"""Fit the two original-size LWM codebooks on the admitted WM/IL half, CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from lwm_stream.dataset import partition_rows,codebook_samples,physical_rows


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rgb-index',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--threads',type=int,required=True)
    p.add_argument('--physical-dedup',action='store_true',help='One representative per physical trajectory; aliases remain provenance only')
    a=p.parse_args()
    if a.seed<0 or a.threads<1:p.error('seed must be nonnegative and threads positive')
    status=json.loads((a.rgb_index/'STATUS.json').read_text())
    if status['status']!='RGB_POSE_INDEX_READY':raise ValueError('RGB/pose index is incomplete')
    rows=[];inputs={}
    sources={r['source']:r for r in status['sources']}
    if set(sources)!={'r2r','rxr'}:raise ValueError('both original sources are required')
    for source in ('r2r','rxr'):
        path=a.rgb_index/source/'INDEX.jsonl'
        digest=sha(path)
        if digest!=sources[source]['index_sha256']:raise ValueError('source index checksum differs')
        current=[json.loads(line) for line in path.read_text().splitlines()]
        if len(current)!=sources[source]['trajectories'] or any(row['source']!=source for row in current):
            raise ValueError('source index coverage differs')
        rows.extend(current);inputs[str(path)]=digest
    partitions=partition_rows(rows,seed=a.seed)
    input_alias_counts={name:len(values) for name,values in partitions.items()}
    if a.physical_dedup:
        partitions={name:physical_rows(values) for name,values in partitions.items()}
    a.output.mkdir(exist_ok=False)
    partition_receipts={}
    for name,values in partitions.items():
        path=a.output/(name+'.jsonl')
        with path.open('x') as f:
            for row in values:f.write(json.dumps(row,separators=(',',':'))+'\n')
        partition_receipts[name]=dict(rows=len(values),physical_trajectories=len({r['pose_key'] for r in values}),
                                      windows=sum(len(r['keyframe_indices'])-1 for r in values),sha256=sha(path))
    start=time.monotonic()
    print(json.dumps(dict(stage='collect_geometry',partitions=partition_receipts)),flush=True)
    samples=codebook_samples(partitions['wm_il'])
    if a.physical_dedup and (not np.all(samples['delta_weights']==1) or not np.all(samples['trajectory_weights']==1)):
        raise ValueError('physical codebook samples must not inherit alias multiplicity')
    kwargs=dict(n_clusters=64,init='k-means++',n_init=1,max_iter=300,tol=1e-4,
                random_state=a.seed,algorithm='lloyd')
    receipts={}
    with threadpool_limits(limits=a.threads):
        for name,data,weights,shape in [
            ('action_centers',samples['deltas'],samples['delta_weights'],(64,2)),
            ('trajectory_centers',samples['trajectories'].reshape(-1,189),samples['trajectory_weights'],(64,63,3))]:
            if len(data)<64:raise ValueError('not enough training samples for 64 original centers')
            print(json.dumps(dict(stage='fit',name=name,samples=len(data),weight_sum=float(weights.sum()))),flush=True)
            fit_start=time.monotonic()
            model=KMeans(**kwargs).fit(data,sample_weight=weights)
            centers=model.cluster_centers_.reshape(shape)
            if not np.isfinite(centers).all():raise ValueError('nonfinite codebook centers')
            path=a.output/(name+'.json');path.write_text(json.dumps(centers.tolist())+'\n')
            receipts[name]=dict(samples=len(data),weight_sum=float(weights.sum()),shape=list(shape),
                                inertia=float(model.inertia_),iterations=int(model.n_iter_),
                                seconds=time.monotonic()-fit_start,sha256=sha(path))
            print(json.dumps(dict(stage='finished',name=name,receipt=receipts[name])),flush=True)
    result=dict(status='CPU_CODEBOOKS_READY',input_sha256=inputs,partitions=partition_receipts,
                representation='physical_trajectories_v2' if a.physical_dedup else 'language_aliases_v1',
                input_alias_counts=input_alias_counts,
                parameters=kwargs,threads=a.threads,sklearn_version=sklearn.__version__,
                books=receipts,seconds=time.monotonic()-start,
                parameter_provenance='64 centers/63 future from LWM; k-means implementation controls explicit provisional choices')
    (a.output/'STATUS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
