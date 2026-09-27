"""Global-batch preserving epoch input for the unchanged LWM trainer."""
import math
import numbers
import torch
from torch.utils.data import DataLoader


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_move(item, device) for item in value)
    return value

class EpochBatches:
    def __init__(self, dataset, *, global_batch, micro_batch, rank, world_size, seed, workers, device, shuffle=True):
        for name, value, minimum in [('global_batch',global_batch,1),('micro_batch',micro_batch,1),
                                     ('rank',rank,0),('world_size',world_size,1),('seed',seed,0),('workers',workers,0)]:
            if isinstance(value,bool) or not isinstance(value,numbers.Integral) or value<minimum:
                raise ValueError(f'{name} is out of range')
        if rank>=world_size or world_size>global_batch:
            raise ValueError('each rank must receive at least one real global-batch item')
        if not isinstance(shuffle,bool):
            raise ValueError('shuffle must be boolean')
        self.dataset,self.global_batch,self.micro_batch=dataset,global_batch,micro_batch
        self.rank,self.world_size,self.seed,self.workers=rank,world_size,seed,workers
        self.device,self.shuffle=torch.device(device),shuffle
        self.total_updates=len(dataset)//global_batch
        self.micro_count=math.ceil(len(range(rank,global_batch,world_size))/micro_batch)

    def __call__(self, epoch, offset):
        if any(isinstance(v,bool) or not isinstance(v,numbers.Integral) or v<0 for v in (epoch,offset)):
            raise ValueError('epoch and update offset must be nonnegative integers')
        if offset>self.total_updates:
            raise ValueError('update offset exceeds the epoch')
        generator=torch.Generator().manual_seed(self.seed+epoch)
        order=(torch.randperm(len(self.dataset),generator=generator) if self.shuffle
               else torch.arange(len(self.dataset)))
        def sampler():
            for update in range(offset,self.total_updates):
                start=update*self.global_batch
                local=order[start+self.rank:start+self.global_batch:self.world_size].tolist()
                for cursor in range(0,len(local),self.micro_batch):
                    yield [(epoch,index) for index in local[cursor:cursor+self.micro_batch]]
        def updates():
            loader=DataLoader(self.dataset,batch_sampler=sampler(),num_workers=self.workers,
                              generator=generator,pin_memory=self.device.type=='cuda',
                              multiprocessing_context='spawn' if self.workers else None)
            iterator=iter(loader)
            for _ in range(offset,self.total_updates):
                yield [_move(next(iterator),self.device) for _ in range(self.micro_count)]
        return self.total_updates,updates()
