"""Memory batching preserves deterministic eval encoder semantics and sample order."""
import pytest
import torch
from rae_stream.inference_batching import install_encoder_batching

class Encoder(torch.nn.Module):
    def __init__(self):super().__init__();self.child=torch.nn.Dropout(.1);self.calls=[]
    def encode(self,x):self.calls.append(len(x));return x*2+1
    def decode(self,x):return x-1

def test_encoder_batching_preserves_order_outputs_rng_and_decode():
    model=Encoder().eval();decode=model.decode
    x=torch.arange(30.).reshape(10,3);rng=torch.get_rng_state().clone()
    install_encoder_batching(model,batch_size=4)
    with torch.no_grad():out=model.encode(x)
    assert torch.equal(out,x*2+1)
    assert model.calls==[4,4,2]
    assert torch.equal(torch.get_rng_state(),rng)
    assert model.decode==decode

@pytest.mark.parametrize('size',[True,0,-1,1.5,float('nan')])
def test_encoder_batch_size_must_be_positive_integer(size):
    with pytest.raises((ValueError,TypeError)):install_encoder_batching(Encoder().eval(),batch_size=size)

def test_encoder_wrapper_rejects_grad_and_training_submodule():
    model=Encoder().eval();install_encoder_batching(model,batch_size=2)
    with pytest.raises((RuntimeError,ValueError)):model.encode(torch.ones(3,2))
    model.child.train()
    with torch.no_grad(),pytest.raises((RuntimeError,ValueError)):model.encode(torch.ones(3,2))
    assert model.calls==[]

def test_encoder_wrapper_cannot_stack():
    model=Encoder().eval();install_encoder_batching(model,batch_size=2)
    with pytest.raises((RuntimeError,ValueError)):install_encoder_batching(model,batch_size=1)

class WorldModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.norm=torch.nn.LayerNorm(3);self.calls=[]
    def forward(self,x,t,**kwargs):
        self.calls.append(len(x))
        return self.norm(x)+t.reshape(-1,1)+kwargs['y']+kwargs['x_cond']+kwargs['rel_t'].reshape(-1,1)

def test_model_batching_aligns_all_condition_batches_and_preserves_rng():
    from rae_stream.inference_batching import make_batched_model
    model=WorldModel().eval();x=torch.arange(21.).reshape(7,3)
    cond={'y':x+3,'x_cond':x+20,'rel_t':torch.arange(7.)};t=torch.arange(7.)*2
    with torch.no_grad():expected=model(x,t,**cond)
    model.calls.clear();wrapper=make_batched_model(model,batch_size=3);rng=torch.get_rng_state().clone()
    with torch.no_grad():actual=wrapper(x,t,**cond)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert model.calls==[3,3,1]
    assert torch.equal(torch.get_rng_state(),rng)
    assert not actual.requires_grad


def test_model_batching_broadcasts_scalar_time_and_rejects_condition_mismatch():
    from rae_stream.inference_batching import make_batched_model
    model=WorldModel().eval();wrapper=make_batched_model(model,batch_size=2)
    x=torch.ones(5,3);cond={'y':x,'x_cond':x,'rel_t':torch.arange(5.)}
    with torch.no_grad():
        actual=wrapper(x,torch.tensor(.5),**cond)
        torch.testing.assert_close(actual,model(x,torch.tensor(.5),**cond))
        with pytest.raises((ValueError,TypeError)):wrapper(x,torch.tensor(.5),**{**cond,'y':x[:4]})


def test_model_batching_rejects_grad_training_and_invalid_batch_size():
    from rae_stream.inference_batching import make_batched_model
    model=WorldModel().eval();wrapper=make_batched_model(model,batch_size=2)
    with pytest.raises((RuntimeError,ValueError)):wrapper(torch.ones(1,3),torch.tensor(0.))
    model.norm.train()
    with torch.no_grad(),pytest.raises((RuntimeError,ValueError)):wrapper(torch.ones(1,3),torch.tensor(0.))
    for size in [True,0,-1,1.5]:
        with pytest.raises((ValueError,TypeError)):make_batched_model(model,batch_size=size)


def test_model_single_sample_preserves_direct_call_objects():
    from rae_stream.inference_batching import make_batched_model
    class Identity(torch.nn.Module):
        def forward(self,x,t,**kwargs):
            assert x is original_x and t is original_t and kwargs['y'] is original_y
            return x
    original_x=torch.ones(1,3);original_t=torch.zeros(1);original_y=torch.ones(1,2)
    wrapper=make_batched_model(Identity().eval(),batch_size=16)
    with torch.no_grad():assert wrapper(original_x,original_t,y=original_y) is original_x


def test_endpoint_only_sampler_keeps_exact_final_state_without_trajectory_storage():
    from rae_stream.inference_batching import install_endpoint_only_sampler
    class Sampler:
        def sample_ode(self,**options):
            self.options=options
            def sample(z,model,**kwargs):
                self.seen=(z,model,kwargs)
                self.full=torch.stack([z+i for i in range(50)])
                return self.full
            return sample
    sampler=Sampler();install_endpoint_only_sampler(sampler)
    fn=sampler.sample_ode(num_steps=50,sampling_method='euler')
    z=torch.arange(12.).reshape(4,3);model=object();condition=torch.ones(4,3)
    rng=torch.get_rng_state().clone()
    with torch.no_grad():result=fn(z,model,y=condition)
    assert isinstance(result,tuple) and len(result)==1
    assert torch.equal(result[-1],z+49)
    assert result[-1].untyped_storage().nbytes()==z.numel()*z.element_size()
    assert result[-1].untyped_storage().data_ptr()!=sampler.full.untyped_storage().data_ptr()
    assert sampler.options=={'num_steps':50,'sampling_method':'euler'}
    assert sampler.seen[0] is z and sampler.seen[1] is model and sampler.seen[2]['y'] is condition
    assert torch.equal(torch.get_rng_state(),rng)
    with pytest.raises((RuntimeError,ValueError)):install_endpoint_only_sampler(sampler)


def test_endpoint_only_sampler_rejects_grad_before_original_execution():
    from rae_stream.inference_batching import install_endpoint_only_sampler
    class Sampler:
        def sample_ode(self):
            def fn(*args,**kwargs):raise AssertionError('grad-enabled invocation reached original')
            return fn
    sampler=Sampler();install_endpoint_only_sampler(sampler)
    with pytest.raises((RuntimeError,ValueError)):sampler.sample_ode()(torch.ones(1),object())
