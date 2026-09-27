"""Official model calls; no new ranking, tokenizer or predicted-state memory."""
import math
import torch


class EmptyPlanError(ValueError):
    def __init__(self,result):
        self.result=result
        super().__init__('official winner is empty; no executable waypoint, no reranking')


class OfficialLWMPlanner:
    def __init__(self, *, policy, world_model, tokenizer, mode='policy_wm',
                 num_samples=32, temperature=1.0, codebook_metric=None):
        if mode not in ('policy_wm','wm_only'):
            raise ValueError('unknown inference mode')
        if type(num_samples) is not int or num_samples<=0:
            raise ValueError('num_samples must be a positive integer')
        if isinstance(temperature,bool) or not math.isfinite(temperature) or temperature<=0:
            raise ValueError('temperature must be finite and positive')
        self.policy,self.world_model,self.tokenizer=policy,world_model,tokenizer
        self.mode,self.num_samples,self.temperature=mode,num_samples,temperature
        self.codebook_metric=codebook_metric

    def reset(self):
        # No factual/imagined features or plans are retained across calls.
        pass

    @torch.no_grad()
    def plan(self, now, goal):
        if now.ndim!=4 or now.shape[0]!=1 or goal.shape!=now.shape:
            raise ValueError('one equally shaped current/goal tensor pair required')
        if self.mode=='policy_wm':
            tokens=self.policy.roll_out(now,goal,num_sample=self.num_samples,temperature=self.temperature)
            actions,lengths=self.tokenizer.batch_decode(tokens.view(self.num_samples,self.policy.max_len))
            if actions.shape!=(self.num_samples,63,3) or lengths.shape!=(self.num_samples,):
                raise ValueError('official decoded shapes mismatch')
            if not torch.isfinite(actions).all() or ((lengths<0)|(lengths>63)).any():
                raise ValueError('invalid decoded trajectory')
            normalized=.1*actions.view(1,self.num_samples,63,3)
            rewards=self.world_model.get_reward(now,goal,normalized,lengths)
            if rewards.shape!=(1,self.num_samples) or not torch.isfinite(rewards).all():
                raise ValueError('invalid official rewards')
            winner=int(rewards[0].argmax().item())
            length=int(lengths[winner].item())
            if length==0:
                raise EmptyPlanError({'mode':self.mode,'winner_index':winner,'valid_length':0,
                    'rewards':rewards[0].cpu().tolist(),'winner_tokens':tokens.view(self.num_samples,self.policy.max_len)[winner].cpu().tolist()})
            trajectory=actions[winner,:length]
            extra={'rewards':rewards[0].cpu().tolist(),'winner_tokens':tokens.view(self.num_samples,self.policy.max_len)[winner].cpu().tolist()}
        else:
            actions=self.codebook_metric
            if actions is None or actions.ndim!=3 or actions.shape[1:]!=(63,3) or not torch.isfinite(actions).all():
                raise ValueError('released metric trajectory codebook required')
            idx,end=self.world_model.eval_wm(now,goal,.1*actions)
            winner=int(idx[0].item());length=int(end[0].item())+1
            if not 0<=winner<len(actions) or not 1<=length<=63:
                raise ValueError('invalid official codebook selection')
            trajectory=actions[winner,:length]
            extra={'scoring_endpoint_index':length-1}
        return {'mode':self.mode,'winner_index':winner,'valid_length':length,
                'metric_waypoint':trajectory[0].cpu().tolist(),
                'metric_trajectory':trajectory.cpu().tolist(),**extra}


def _rgb(value):
    import numpy as np
    if not isinstance(value,np.ndarray) or value.dtype!=np.uint8 or value.ndim!=3 or value.shape[-1]!=3:
        raise ValueError('uint8 HWC RGB only')
    return value.copy()


class Policy:
    """Current RGB only; runner history/poses/metrics cannot reach LWM."""
    def __init__(self,backend):
        self.backend=backend
        self.goal=None
        self.last_plan=None

    def reset(self,goal_rgb):
        self.goal=_rgb(goal_rgb)
        self.last_plan=None
        reset=getattr(self.backend,'reset',None)
        if callable(reset):reset()

    def act(self,current_rgb,goal_rgb,factual_history):
        import numpy as np
        current,goal=_rgb(current_rgb),_rgb(goal_rgb)
        if self.goal is None:raise ValueError('reset required before act')
        if not np.array_equal(self.goal,goal):raise ValueError('goal changed without reset')
        self.last_plan=self.backend.plan(current,goal)
        return {'continuous_action':self.last_plan['metric_waypoint']}

    def close(self):
        close=getattr(self.backend,'close',None)
        if callable(close):close()
        self.goal=None
        self.last_plan=None
