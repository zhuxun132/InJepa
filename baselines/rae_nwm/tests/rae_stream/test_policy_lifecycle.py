from contextlib import nullcontext
from types import SimpleNamespace
import numpy as np
import pytest

from rae_stream.habitat_policy import OfficialRAEPlannerBackend, RAEStreamPolicy


class FakeBackend:
    def __init__(self):
        self.calls = []

    def plan(self, context, goal):
        self.calls.append((np.array(context, copy=True), np.array(goal, copy=True)))
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)


def test_policy_warms_four_frames_and_only_uses_factual_history():
    backend = FakeBackend()
    policy = RAEStreamPolicy(backend)
    goal = np.full((8, 8, 3), 9, dtype=np.uint8)
    current = np.full((8, 8, 3), 1, dtype=np.uint8)
    assert policy.reset(goal) is None
    assert policy.act(current, goal, []) == 1
    context, seen_goal = backend.calls[-1]
    assert context.shape[0] == 4
    assert np.all(context == 1)
    assert np.all(seen_goal == 9)

    previous = np.full((8, 8, 3), 2, dtype=np.uint8)
    # A predicted frame deliberately appears in an unrelated key and must be ignored.
    history = [{"rgb": previous, "action": "FWD", "order": 0, "mask": True,
                "predicted_rgb": np.full_like(previous, 99)}]
    current2 = np.full((8, 8, 3), 3, dtype=np.uint8)
    assert policy.act(current2, goal, history) == 1
    context, _ = backend.calls[-1]
    assert context.shape[0] == 4
    assert np.all(context[-2] == 2)
    assert np.all(context[-1] == 3)
    assert not np.any(context == 99)


def test_policy_drops_planner_output_between_decisions():
    backend = FakeBackend()
    policy = RAEStreamPolicy(backend)
    goal = np.zeros((4, 4, 3), dtype=np.uint8)
    current = np.ones((4, 4, 3), dtype=np.uint8)
    policy.reset(goal)
    policy.act(current, goal, [])
    assert not hasattr(policy, "predicted_frames")


def test_policy_context_size_one_does_not_leak_all_history():
    backend = FakeBackend()
    policy = RAEStreamPolicy(backend, context_size=1)
    goal = np.zeros((4, 4, 3), dtype=np.uint8)
    current = np.ones((4, 4, 3), dtype=np.uint8)
    policy.reset(goal)
    assert policy.act(current, goal, [{"rgb": np.full_like(current, 7)}]) == 1
    context, _ = backend.calls[-1]
    assert context.shape == (1, 4, 4, 3)
    assert np.all(context[0] == 1)


def test_official_callback_capture_restores_hook_and_returns_first_delta(monkeypatch):
    import sys

    module = sys.modules[__name__]
    original = getattr(module, "save_planning_pred", None)

    class FakeTensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def __getitem__(self, index):
            return FakeTensor(self.value[index])

        def unsqueeze(self, dim):
            return FakeTensor(np.expand_dims(self.value, axis=dim))

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def reshape(self, *shape):
            return FakeTensor(self.value.reshape(*shape))

        def tolist(self):
            return self.value.tolist()

    class FakeTorch:
        float32 = np.float32
        bfloat16 = "bfloat16"
        no_grad = staticmethod(nullcontext)
        amp = SimpleNamespace(autocast=lambda *args, **kwargs: nullcontext())

        @staticmethod
        def as_tensor(value, *args, **kwargs):
            return FakeTensor(value)

        @staticmethod
        def tensor(value):
            return FakeTensor(value)

        @staticmethod
        def stack(values, dim=0):
            return FakeTensor(np.stack([item.value for item in values], axis=dim))

        @staticmethod
        def zeros(shape, dtype=None):
            return FakeTensor(np.zeros(shape, dtype=dtype))

    fake_torch = FakeTorch()
    calls = []

    class Evaluator:
        __module__ = __name__

        def __init__(self):
            self.args = type("Args", (), {"save_preds": False})()

        def generate_actions(self, *_args):
            calls.append(_args)
            module.save_planning_pred(None, 1, fake_torch.zeros((1, 1)), None, None, None,
                                      fake_torch.tensor([[[1.0, 2.0, 3.0]]]), None, None)

    module.save_planning_pred = lambda *args, **kwargs: None
    try:
        backend = OfficialRAEPlannerBackend(
            Evaluator(),
            transform=lambda image: fake_torch.zeros((3, 4, 4)),
            torch_module=fake_torch,
        )
        command = backend.plan(np.zeros((4, 4, 4, 3), dtype=np.uint8), np.zeros((4, 4, 3), dtype=np.uint8))
        assert command == (1.0, 2.0, 3.0)
        assert calls and calls[-1][-1] == 8
        assert module.save_planning_pred is not None
        assert backend.evaluator.args.save_preds is False
    finally:
        if original is None:
            try:
                delattr(module, "save_planning_pred")
            except AttributeError:
                pass
        else:
            module.save_planning_pred = original


def test_official_callback_capture_restores_hook_on_exception():
    import sys

    module = sys.modules[__name__]
    original = getattr(module, "save_planning_pred", None)

    class FakeTensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def unsqueeze(self, dim):
            return FakeTensor(np.expand_dims(self.value, axis=dim))

    class FakeTorch:
        float32 = np.float32
        bfloat16 = "bfloat16"
        no_grad = staticmethod(nullcontext)
        amp = SimpleNamespace(autocast=lambda *args, **kwargs: nullcontext())

        @staticmethod
        def as_tensor(value, *args, **kwargs):
            return FakeTensor(value)

        @staticmethod
        def stack(values, dim=0):
            return FakeTensor(np.stack([item.value for item in values], axis=dim))

        @staticmethod
        def zeros(shape, dtype=None):
            return FakeTensor(np.zeros(shape, dtype=dtype))

    fake_torch = FakeTorch()

    class Evaluator:
        __module__ = __name__

        def __init__(self):
            self.args = type("Args", (), {"save_preds": False})()

        def generate_actions(self, *_args):
            raise RuntimeError("planner boom")

    marker = lambda *args, **kwargs: None
    module.save_planning_pred = marker
    try:
        backend = OfficialRAEPlannerBackend(
            Evaluator(),
            transform=lambda image: FakeTensor(np.zeros((3, 4, 4), dtype=np.float32)),
            torch_module=fake_torch,
        )
        with pytest.raises(RuntimeError, match="ABI failed"):
            backend.plan(np.zeros((4, 4, 4, 3), dtype=np.uint8), np.zeros((4, 4, 3), dtype=np.uint8))
        assert module.save_planning_pred is marker
        assert backend.evaluator.args.save_preds is False
    finally:
        if original is None:
            try:
                delattr(module, "save_planning_pred")
            except AttributeError:
                pass
        else:
            module.save_planning_pred = original
