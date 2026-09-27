"""Strict classified JSONL metrics for context-four joint training."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor


METRIC_SUBDIRS = (
    "raw",
    "loss",
    "grad",
    "lr",
    "validation",
    "timing",
    "resources",
    "checkpoints",
)


def _validate_finite_json(value: object, *, path: str = "record") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite numbers")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings")
            _validate_finite_json(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_finite_json(child, path=f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains a non-JSON value")


def _append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    _validate_finite_json(record)
    encoded = json.dumps(
        dict(record),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


class ModeDiagnosticAccumulator:
    """Stream detached K=4 statistics without retaining a proposal tape."""

    def __init__(self, *, qg_objective: str = "marginal") -> None:
        if qg_objective not in {"marginal", "posterior_weighted"}:
            raise ValueError("qg_objective must be marginal or posterior_weighted")
        self.qg_objective = qg_objective
        self.occurrences = 0
        self.responsibility_sum = torch.zeros(4, dtype=torch.float64)
        self.posterior_sum = torch.zeros((4, 4), dtype=torch.float64)
        self.gradient_square_sum = torch.zeros(4, dtype=torch.float64)
        self.effective_sum = 0.0
        self.tape_pair_sum = torch.zeros(6, dtype=torch.float64)
        self.action_pair_sum = torch.zeros(6, dtype=torch.float64)

    def update(
        self,
        *,
        q_responsibility: Tensor,
        q_tape: Tensor,
        q_active_h: Tensor,
        qg_logits: Tensor,
        qg_labels: Tensor,
    ) -> None:
        tensors = (q_responsibility, q_tape, q_active_h, qg_logits, qg_labels)
        if not all(isinstance(value, Tensor) for value in tensors):
            raise TypeError("mode diagnostic inputs must be tensors")
        if q_responsibility.ndim != 2 or q_responsibility.shape[1] != 4:
            raise ValueError("mode diagnostics require exactly K=4 responsibilities")
        batch = int(q_responsibility.shape[0])
        if batch < 1:
            return
        if q_tape.ndim != 5 or q_tape.shape[:2] != (batch, 4):
            raise ValueError("mode diagnostic tape must have shape [N,4,H,S,D]")
        if q_active_h.shape != (batch, q_tape.shape[2]) or q_active_h.dtype != torch.bool:
            raise ValueError("mode diagnostic horizon mask is inconsistent")
        if qg_logits.shape != (batch, 4, 4):
            raise ValueError("mode diagnostic logits must have shape [N,4,4]")
        if qg_labels.shape != (batch,) or qg_labels.dtype != torch.int64:
            raise ValueError("mode diagnostic labels must be int64 [N]")
        if bool((qg_labels < 0).any()) or bool((qg_labels >= 4).any()):
            raise ValueError("mode diagnostic labels are outside four actions")
        if bool((q_active_h.sum(dim=1) < 1).any()):
            raise ValueError("mode diagnostic horizon must be nonempty")
        floating = (q_responsibility, q_tape, qg_logits)
        if not all(value.is_floating_point() and bool(torch.isfinite(value).all()) for value in floating):
            raise ValueError("mode diagnostic tensors must be finite floating point")

        responsibility = q_responsibility.detach().float()
        if bool((responsibility < 0).any()) or not bool(
            torch.allclose(
                responsibility.sum(dim=1),
                torch.ones(batch, device=responsibility.device),
                rtol=1e-5,
                atol=1e-6,
            )
        ):
            raise ValueError("mode responsibilities must be normalized")
        posterior = torch.softmax(qg_logits.detach().float(), dim=-1)
        chosen = posterior.gather(
            -1,
            qg_labels[:, None, None].expand(-1, 4, 1),
        ).squeeze(-1)
        if self.qg_objective == "posterior_weighted":
            mode_weight = responsibility
        else:
            chosen_sum = chosen.sum(dim=1, keepdim=True)
            if bool((chosen_sum <= 0).any()) or not bool(torch.isfinite(chosen_sum).all()):
                raise ValueError("mode action likelihood is not finite and positive")
            mode_weight = chosen / chosen_sum
        one_hot = torch.nn.functional.one_hot(qg_labels, num_classes=4).float()[:, None]
        analytic_gradient = mode_weight[:, :, None] * (posterior - one_hot)

        safe_log = torch.where(
            responsibility > 0,
            responsibility.log(),
            torch.zeros_like(responsibility),
        )
        effective = (-(responsibility * safe_log).sum(dim=1)).exp()
        mask = q_active_h[:, :, None, None]
        denominator = (
            q_active_h.sum(dim=1).to(torch.float32)
            * int(q_tape.shape[3])
            * int(q_tape.shape[4])
        )
        pairs = tuple((left, right) for left in range(4) for right in range(left + 1, 4))
        tape_sums: list[Tensor] = []
        action_sums: list[Tensor] = []
        tape = q_tape.detach()
        for left, right in pairs:
            # Convert only the current pair, avoiding a full raw tape FP32 copy.
            distance = (tape[:, left].float() - tape[:, right].float()).abs()
            per_row = distance.masked_fill(~mask, 0.0).sum(dim=(1, 2, 3)) / denominator
            tape_sums.append(per_row.sum())
            action_sums.append(
                (posterior[:, left] - posterior[:, right]).abs().mean(dim=1).sum()
            )

        self.occurrences += batch
        self.responsibility_sum += responsibility.sum(dim=0).double().cpu()
        self.posterior_sum += posterior.sum(dim=0).double().cpu()
        self.gradient_square_sum += analytic_gradient.square().sum(dim=(0, 2)).double().cpu()
        self.effective_sum += float(effective.sum().double().cpu())
        self.tape_pair_sum += torch.stack(tape_sums).double().cpu()
        self.action_pair_sum += torch.stack(action_sums).double().cpu()

    def summary(self) -> Mapping[str, object]:
        if self.occurrences < 1:
            return _empty_mode_diagnostics()
        count = float(self.occurrences)
        gradient_l2 = self.gradient_square_sum.sqrt()
        gradient_total = float(gradient_l2.sum())
        if not math.isfinite(gradient_total) or gradient_total <= 0:
            raise ValueError("mode analytic gradient share is not finite and positive")
        posterior = self.posterior_sum / count
        pairs = tuple((left, right) for left in range(4) for right in range(left + 1, 4))
        return {
            "q_occurrences": self.occurrences,
            "q_responsibility": (self.responsibility_sum / count).tolist(),
            "g_action_posterior": posterior.tolist(),
            "qg_logit_gradient_share": (gradient_l2 / gradient_total).tolist(),
            "k_effective": self.effective_sum / count,
            "pairwise_mode_distance": [
                {
                    "left": left,
                    "right": right,
                    "tape": float(self.tape_pair_sum[index] / count),
                    "action": float(self.action_pair_sum[index] / count),
                }
                for index, (left, right) in enumerate(pairs)
            ],
            "action_argmax_distinct_count": len(
                set(int(value) for value in posterior.argmax(dim=1).tolist())
            ),
        }


def _empty_mode_diagnostics() -> Mapping[str, object]:
    return {
        "q_occurrences": 0,
        "q_responsibility": [0.0] * 4,
        "g_action_posterior": [[0.0] * 4 for _ in range(4)],
        "qg_logit_gradient_share": [0.0] * 4,
        "k_effective": 0.0,
        "pairwise_mode_distance": [
            {"left": left, "right": right, "tape": 0.0, "action": 0.0}
            for left in range(4)
            for right in range(left + 1, 4)
        ],
        "action_argmax_distinct_count": 0,
    }


def summarize_mode_diagnostics(
    *,
    q_responsibility: Tensor,
    q_tape: Tensor,
    q_active_h: Tensor,
    qg_logits: Tensor,
    qg_labels: Tensor,
) -> Mapping[str, object]:
    """Summarize Q/G modes with analytic gradients and no candidate gate."""

    accumulator = ModeDiagnosticAccumulator()
    accumulator.update(
        q_responsibility=q_responsibility,
        q_tape=q_tape,
        q_active_h=q_active_h,
        qg_logits=qg_logits,
        qg_labels=qg_labels,
    )
    return accumulator.summary()


class TrainingMetricsStore:
    """Own one run's append-only ``data/training_metrics`` namespace."""

    def __init__(self, run_root: str | os.PathLike[str]) -> None:
        root = Path(run_root)
        if root.exists() and not root.is_dir():
            raise ValueError("metrics run root must be a directory")
        self.root = root / "data" / "training_metrics"
        for name in METRIC_SUBDIRS:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def write_update(self, record: Mapping[str, object]) -> None:
        if not isinstance(record, Mapping):
            raise TypeError("update record must be a mapping")
        update = record.get("successful_update")
        if isinstance(update, bool) or not isinstance(update, int) or update < 1:
            raise ValueError("successful_update must be a positive integer")
        required = ("loss", "numerator", "denominator", "grad", "lr", "timing", "resources")
        missing = [name for name in required if name not in record]
        if missing:
            raise ValueError(f"update record is missing required fields: {missing}")
        _validate_finite_json(record)
        _append_jsonl(self.root / "raw" / "updates.jsonl", record)
        for category in ("loss", "grad", "lr", "timing", "resources"):
            category_record: dict[str, object] = {
                "successful_update": update,
                category: record[category],
            }
            if category == "loss":
                category_record["numerator"] = record["numerator"]
                category_record["denominator"] = record["denominator"]
                if "stochastic_q" in record:
                    category_record["stochastic_q"] = record["stochastic_q"]
            _append_jsonl(self.root / category / "updates.jsonl", category_record)

    def write_validation(self, record: Mapping[str, object]) -> None:
        if not isinstance(record, Mapping):
            raise TypeError("validation record must be a mapping")
        if "epoch" not in record or "dev" not in record:
            raise ValueError("validation record must include epoch and dev")
        _append_jsonl(self.root / "validation" / "epochs.jsonl", record)

    def write_rank_resource(self, record: Mapping[str, object]) -> None:
        if not isinstance(record, Mapping):
            raise TypeError("rank resource record must be a mapping")
        rank = record.get("rank")
        update = record.get("successful_update")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError("rank resource record has an invalid rank")
        if isinstance(update, bool) or not isinstance(update, int) or update < 1:
            raise ValueError("rank resource record has an invalid successful update")
        _append_jsonl(
            self.root / "resources" / f"rank_{rank:04d}.jsonl",
            record,
        )

    def write_model_probe(self, record: Mapping[str, object]) -> None:
        if not isinstance(record, Mapping) or record.get("phase") != "dev":
            raise ValueError("model probe record must identify dev phase")
        for name, minimum in (("rank", 0), ("epoch", 1), ("successful_update", 1)):
            value = record.get(name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"model probe has invalid {name}")
        probe = record.get("model_probe")
        if not isinstance(probe, Mapping) or probe.get("schema") != "j2j.bounded_model_probe.v1":
            raise ValueError("model probe payload has an invalid schema")
        _validate_finite_json(record)
        _append_jsonl(self.root / "raw" / f"model_probe_dev_rank_{record['rank']:04d}.jsonl", record)

    def write_rank_diagnostic(self, record: Mapping[str, object]) -> None:
        if not isinstance(record, Mapping):
            raise TypeError("rank diagnostic record must be a mapping")
        rank = record.get("rank")
        update = record.get("successful_update")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError("rank diagnostic record has an invalid rank")
        if isinstance(update, bool) or not isinstance(update, int) or update < 1:
            raise ValueError("rank diagnostic record has an invalid successful update")
        required = ("pre_clip_grad", "activation", "modes")
        if any(name not in record for name in required):
            raise ValueError("rank diagnostic record is incomplete")
        _validate_finite_json(record)
        _append_jsonl(
            self.root / "raw" / f"diagnostics_rank_{rank:04d}.jsonl",
            record,
        )
        _append_jsonl(
            self.root / "grad" / f"fqn_rank_{rank:04d}.jsonl",
            {
                "rank": rank,
                "successful_update": update,
                "pre_clip_grad": record["pre_clip_grad"],
            },
        )

    def write_checkpoint(self, record: Mapping[str, object]) -> None:
        if not isinstance(record, Mapping):
            raise TypeError("checkpoint record must be a mapping")
        _append_jsonl(self.root / "checkpoints" / "epochs.jsonl", record)
