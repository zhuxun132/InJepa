"""IL cached labels bind the original epoch-zero image pair."""
import hashlib
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(HERE/'source'))
from lwm_stream.dataset import PseudoDataset

class PairDataset:
    def __init__(self): self.calls=[]
    def __len__(self): return 2
    def __getitem__(self,key):
        self.calls.append(key)
        return {'now':torch.tensor(key),'goal':torch.tensor(key)+100}


def tokens():
    values=np.full((2,65),66,dtype=np.int16)
    values[0,:4]=[64,3,9,65]; values[1,:3]=[64,1,65]
    return values


def write(tmp_path,values):
    path=tmp_path/'labels.npy'; np.save(path,values)
    return path,hashlib.sha256(path.read_bytes()).hexdigest()


def test_pseudo_labels_fixed_epoch0_pair_long_tokens_and_readonly_original(tmp_path):
    path,sha=write(tmp_path,tokens()); before=path.read_bytes(); pair=PairDataset()
    dataset=PseudoDataset(pair,path,expected_sha256=sha)
    assert len(dataset)==2
    first=dataset[(7,1)]; second=dataset[1]
    assert pair.calls==[(0,1),(0,1)]
    assert set(first)=={'now','goal','tokens'}
    torch.testing.assert_close(first['now'],torch.tensor([0,1]),rtol=0,atol=0)
    torch.testing.assert_close(first['goal'],torch.tensor([100,101]),rtol=0,atol=0)
    assert first['tokens'].dtype==torch.long
    torch.testing.assert_close(first['tokens'],torch.tensor(tokens()[1],dtype=torch.long),rtol=0,atol=0)
    first['tokens'][1]=42
    assert second['tokens'][1].item()==1 and dataset[(3,1)]['tokens'][1].item()==1
    assert path.read_bytes()==before

@pytest.mark.parametrize('invalid',['sha','shape','count','no_eos','motion_after_eos'])
def test_invalid_cache_rejected_at_construction(tmp_path,invalid):
    values=tokens()
    if invalid=='shape': values=values[:,:64]
    elif invalid=='count': values=values[:1]
    elif invalid=='no_eos': values[0,3]=2
    elif invalid=='motion_after_eos': values[0,4]=2
    path,sha=write(tmp_path,values)
    if invalid=='sha': sha='0'*64
    pair=PairDataset()
    with pytest.raises(ValueError): PseudoDataset(pair,path,expected_sha256=sha)
    assert pair.calls==[]


class LargePairDataset(PairDataset):
    def __init__(self,count): super().__init__(); self.count=count
    def __len__(self): return self.count


def pseudo_spawn_probe(dataset,queue):
    mapped=dataset.tokens
    sample=dataset[(9,0)]; sample['tokens'][1]=42
    queue.put({'is_memmap':isinstance(mapped,np.memmap),'writeable':bool(mapped.flags.writeable),
               'filename':str(mapped.filename) if mapped.filename is not None else None,
               'original_token':int(mapped[0,1]),'copied_token':int(sample['tokens'][1]),
               'next_token':int(dataset[0]['tokens'][1])})


def test_pickle_and_real_spawn_reopen_readonly_cache_without_serializing_tokens(tmp_path):
    import multiprocessing
    import pickle
    count=20000
    values=np.tile(tokens()[:1],(count,1))
    path,sha=write(tmp_path,values)
    dataset=PseudoDataset(LargePairDataset(count),path,expected_sha256=sha)
    blob=pickle.dumps(dataset,protocol=pickle.HIGHEST_PROTOCOL)
    restored=pickle.loads(blob)
    context=multiprocessing.get_context('spawn'); queue=context.Queue()
    process=context.Process(target=pseudo_spawn_probe,args=(dataset,queue))
    process.start()
    try:
        child=queue.get(timeout=30); process.join(timeout=5)
        assert process.exitcode==0
    finally:
        if process.is_alive(): process.terminate(); process.join(timeout=5)
        queue.close(); queue.join_thread()
    observed={'pickle_bytes':len(blob),'label_bytes':values.nbytes,
              'restored_memmap':isinstance(restored.tokens,np.memmap),
              'restored_writeable':bool(restored.tokens.flags.writeable),
              'restored_filename':str(restored.tokens.filename) if restored.tokens.filename is not None else None,
              'child':child}
    print(observed)
    assert len(blob)<65536,observed # metadata-only transfer, far below2.6MB labels
    assert isinstance(restored.tokens,np.memmap) and not restored.tokens.flags.writeable,observed
    assert Path(restored.tokens.filename)==path,observed
    assert child=={'is_memmap':True,'writeable':False,'filename':str(path),'original_token':3,'copied_token':42,'next_token':3},observed
