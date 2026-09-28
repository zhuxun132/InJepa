# Train and evaluate baselines

Run common launch commands from the repository root. Use the [evaluation environment](ENVIRONMENT.md) and the same `SCENE_ROOT` as InJepa. `baselines/checkpoints.json` records the final experiment checkpoint identities, and `data/baseline_settings.json` records the training and controller settings.

## LWM-CroCo and LWM-VJEPA

Install the selected variant in its own environment:

```bash
python -m pip install -e baselines/lwm_croco
```

The corresponding V-JEPA install command uses `baselines/lwm_vjepa`. Both use the same released-trajectory preparation pipeline. Run shared data preparation from `baselines/lwm_croco`: prepare poses with `scripts/prepare_replay.py`, join original JPEGs with `scripts/prepare_rgb.py`, then fit codebooks:

```bash
export REPLAY_METADATA="$REPO/data/lwm_replay"
export REPLAY_ASSETS_MANIFEST="$REPLAY_METADATA/scene_assets.json"
CUDA_VISIBLE_DEVICES='' python scripts/prepare_replay.py \
  --annotations-root "$ANNOTATIONS" --metadata-root "$REPLAY_METADATA" \
  --assets-manifest "$REPLAY_ASSETS_MANIFEST" --assets "$SCENE_ROOT/mp3d" \
  --configuration configs/streamvln_replay.json --output "$POSE_OUTPUT" --workers 8
python scripts/prepare_rgb.py --replay "$POSE_OUTPUT" \
  --sources configs/server_rgb_sources.json --output "$RGB_INDEX" --workers 8
python scripts/prepare_codebooks.py --rgb-index "$RGB_INDEX" \
  --output "$CODEBOOKS" --seed 0 --threads 8 --physical-dedup
```

Resolve the RGB source paths in `configs/server_rgb_sources.json` first, and set `POSE_OUTPUT`, `RGB_INDEX` and `CODEBOOKS` to new local directories. `data/lwm_replay` supplies the matched original R2R/RxR episode poses (`R2R_MATCHED_TRAIN_POSES.json`, `RXR_MATCHED_TRAIN_POSES.json`) and `BUILDING_SPLIT_REPLAY_CENSUS.json`. Its asset manifest lists the 61 training/dev scenes and their file hashes. Install these scenes under the same MP3D layout before replay. Replay recovers poses from recorded actions. RGB preparation reads the released JPEGs. Keep the generated receipts with the data. The fitted experiment centres are in `data/lwm_action_centers.json` and `data/lwm_trajectory_centers.json` and in the respective tokenizer directories.

For training, change to the selected variant directory and copy its `configs/train.json` to `train.local.json`. Set `codebooks`, `croco` and `output`. For LWM-VJEPA also set `vision.source_root` and `vision.checkpoint`. Train in stage order:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_lwm.py --config train.local.json --stage wm
python scripts/train_lwm.py --config train.local.json --stage pseudo
torchrun --standalone --nproc_per_node=2 scripts/train_lwm.py --config train.local.json --stage il
python scripts/train_lwm.py --config train.local.json --stage rl --micro-batch 8
```

WM uses 50 epochs and global batch 8. IL and RL each use 40 epochs and global batch 24. The trainer derives accumulation from the selected local microbatch. Use `--preflight` to validate a stage's predecessor artifacts before launching it. Keep each stage's `COMPLETE.json` with its checkpoint.

### LWM-CroCo evaluation

The CroCo evaluator consumes plain model state dictionaries. Export the trained WM and RL payloads using their completion receipts. Set `LWM_RUN` to the training output and run this from the repository root:

```bash
export LWM_RUN=/absolute/path/to/completed/lwm_croco_run
python - <<'PY'
import hashlib, json, os
from pathlib import Path
import torch
root = Path(os.environ['LWM_RUN'])
out = root / 'evaluation_weights'
out.mkdir(exist_ok=False)
for stage in ('wm', 'rl'):
    receipt = json.loads((root / stage / 'COMPLETE.json').read_text())
    assert receipt['status'] == 'COMPLETE'
    source = root / stage / receipt['checkpoint']
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == receipt['sha256']
    payload = torch.load(source, map_location='cpu', weights_only=False)
    state = payload['model']
    destination = out / (stage + '.pt')
    torch.save(state, destination)
    loaded = torch.load(destination, map_location='cpu', weights_only=True)
    assert state.keys() == loaded.keys()
    assert all(torch.equal(value, loaded[key]) for key, value in state.items())
PY
python scripts/prepare_evaluation.py --method lwm_croco --scene-root "$SCENE_ROOT" \
  --wm-checkpoint "$LWM_RUN/evaluation_weights/wm.pt" \
  --policy-checkpoint "$LWM_RUN/evaluation_weights/rl.pt" --output runs/prepared/lwm_croco
python scripts/evaluate.py --method lwm_croco --config runs/prepared/lwm_croco/evaluate.json \
  --episodes 1 --seed 0 --output runs/lwm_croco_one_task
```

### LWM-VJEPA evaluation

Pass the resolved training configuration so the evaluator can locate the completed WM/RL stages and their predecessor receipts:

```bash
python scripts/prepare_evaluation.py --method lwm_vjepa --scene-root "$SCENE_ROOT" \
  --training-config /absolute/path/to/lwm_vjepa/train.local.json --output runs/prepared/lwm_vjepa
python scripts/evaluate.py --method lwm_vjepa --config runs/prepared/lwm_vjepa/evaluate.json \
  --episodes 1 --seed 0 --output runs/lwm_vjepa_one_task
```

## NoMaD

Obtain the official NoMaD checkpoint from the link in `baselines/checkpoints.json` and set `NOMAD_CHECKPOINT` to that local file. The evaluator passes the goal image directly to the goal-conditioned policy and uses its original waypoint controller.

```bash
python scripts/prepare_evaluation.py --method nomad --scene-root "$SCENE_ROOT" \
  --checkpoint "$NOMAD_CHECKPOINT" --output runs/prepared/nomad
python scripts/evaluate.py --method nomad --config runs/prepared/nomad/evaluate.json \
  --episodes 1 --seed 0 --output runs/nomad_one_task
```

For training, prepare trajectories in the official ViNT dataset layout, set the data/split paths in `baselines/nomad/official/train/config/nomad.yaml`, install the local train package, and run from that `train/` directory:

```bash
python -m pip install -e .
python train.py -c config/nomad.yaml
```

Model construction and diffusion sampling use the official source revision recorded in `baselines/nomad/upstream.json`.

## RAE-NWM

Create two runtime checkouts with the fixed public Git base, one for training and one for evaluation:

```bash
export RAE_EXPORT="$PWD/baselines/rae_nwm"
for runtime in third_party/rae_train_runtime third_party/rae_eval_runtime; do
  git clone https://github.com/20robo/raenwm.git "$runtime"
  git -C "$runtime" checkout 0219ce41c44d515f86719dd763c1efe7c7f72519
  cp "$RAE_EXPORT"/*.py "$runtime/"
  cp -R "$RAE_EXPORT"/config "$RAE_EXPORT"/RAE "$RAE_EXPORT"/diffusion \
    "$RAE_EXPORT"/rae_stream "$RAE_EXPORT"/scripts "$RAE_EXPORT"/tests "$runtime/"
done
cp "$RAE_EXPORT"/evaluation_overlay/planning_eval.py third_party/rae_eval_runtime/
cp "$RAE_EXPORT"/evaluation_overlay/rae_stream/*.py third_party/rae_eval_runtime/rae_stream/
cp "$RAE_EXPORT"/evaluation_overlay/scripts/*.py third_party/rae_eval_runtime/scripts/
```

Keep the public base commit as the checkout's HEAD. The evaluation overlay implements the experiment's Euler10 planner. Install the official decoder, latent statistics and DINOv2 assets at the paths checked by `rae_stream/assets.py`. Use its pinned Hugging Face revision `a1d738ccfa7ae170945f210395d99dde8adb1805` for `facebook/dinov2-with-registers-base`.

From each RAE runtime checkout, download the decoder/statistics into `models/`. Use the same dedicated encoder cache for training and evaluation:

```bash
export HF_HOME="$RAE_HF_HOME"
python - <<'PYASSETS'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
snapshot_download('nyu-visionx/RAE-collections', local_dir='models', allow_patterns=[
    'decoders/dinov2/wReg_base/ViTXL_n08/model.pt',
    'stats/dinov2/wReg_base/imagenet1k/stat.pt'])
revision = 'a1d738ccfa7ae170945f210395d99dde8adb1805'
snapshot = Path(snapshot_download('facebook/dinov2-with-registers-base',
    revision=revision, cache_dir=Path(os.environ['HF_HOME']) / 'hub'))
refs = snapshot.parent.parent / 'refs'
refs.mkdir(exist_ok=True)
(refs / 'main').write_text(revision)
PYASSETS
```

From `third_party/rae_train_runtime`, create the training data:

```bash
python scripts/prepare_rae_stream.py --output-root "$RAE_DATA" \
  --r2r-annotations "$ANNOTATIONS/R2R/annotations_v1-3.json" \
  --rxr-annotations "$ANNOTATIONS/RxR/annotations.json" \
  --r2r-archive "$RGB_ARCHIVES/R2R_images_v1-3.tar.gz" \
  --rxr-archive-part "$RGB_ARCHIVES/RxR_images.tar.gz.part0" \
  --rxr-archive-part "$RGB_ARCHIVES/RxR_images.tar.gz.part1" --workers 8
python scripts/generate_code_authority.py --repo-root "$PWD" --output "$RAE_DATA/code_authority.json"
python scripts/launch_rae_stream.py plan --config config/rae_stream.yaml \
  --manifest "$RAE_DATA/rae_stream_manifest.jsonl" --dry-run
```

Connect the generated data to the paths in `config/rae_stream.yaml`, validate it, and qualify the selected batch layout. Set `RAE_HF_HOME` to the dedicated cache with the pinned visual assets and `RAE_GPUS` to three available physical GPU indices. Set `RAE_GPU_UUIDS` to their comma-separated GPU UUIDs in the same order. Obtain them with `nvidia-smi --query-gpu=index,uuid --format=csv,noheader`:

```bash
ln -s "$RAE_DATA/data" data
ln -s "$RAE_DATA/data_splits" data_splits
file_sha() { sha256sum "$1" | cut -d ' ' -f 1; }
python scripts/validate_rae_stream.py \
  --data-root "$RAE_DATA/data/rae_stream" --split-root "$RAE_DATA/data_splits/rae_stream" \
  --manifest "$RAE_DATA/rae_stream_manifest.jsonl" --official-root "$PWD" \
  --prepare-receipt "$RAE_DATA/rae_stream_prepare_receipt.json" \
  --prepare-receipt-sha256 "$(file_sha "$RAE_DATA/rae_stream_prepare_receipt.json")" \
  --authority "$RAE_DATA/code_authority.json" \
  --authority-sha256 "$(file_sha "$RAE_DATA/code_authority.json")" \
  --receipt "$RAE_DATA/validation.json"
python scripts/memory_preflight_rae_stream.py --repo-root "$PWD" \
  --config config/rae_stream.yaml --world-size 3 --gpu-indices "$RAE_GPUS" \
  --gpu-uuids "$RAE_GPU_UUIDS" \
  --gradient-accumulation-steps 4 --hf-home "$RAE_HF_HOME" \
  --output "$RAE_DATA/memory.json" --report-dir "$RAE_DATA/memory_probe"
python scripts/launch_rae_stream.py train --config config/rae_stream.yaml \
  --manifest "$RAE_DATA/rae_stream_manifest.jsonl" \
  --prepare-receipt "$RAE_DATA/rae_stream_prepare_receipt.json" \
  --prepare-receipt-sha256 "$(file_sha "$RAE_DATA/rae_stream_prepare_receipt.json")" \
  --validation-receipt "$RAE_DATA/validation.json" \
  --validation-receipt-sha256 "$(file_sha "$RAE_DATA/validation.json")" \
  --memory-preflight "$RAE_DATA/memory.json" \
  --memory-preflight-sha256 "$(file_sha "$RAE_DATA/memory.json")" \
  --authority "$RAE_DATA/code_authority.json" \
  --authority-sha256 "$(file_sha "$RAE_DATA/code_authority.json")" \
  --hf-home "$RAE_HF_HOME" --world-size 3 --gpu-indices "$RAE_GPUS" \
  --gradient-accumulation-steps 4 \
  --nvidia-smi-sha256 "$(file_sha "$(command -v nvidia-smi)")" \
  --receipt "$RAE_DATA/launch.json" --log "$RAE_DATA/train.log"
```

The experiment uses global batch 96, 3 ranks, accumulation 4, 50 scheduled epochs, seed 42 and BF16. The launcher preserves global batch size while producing the per-process configuration.

For evaluation, set `RAE_CHECKPOINT` to an EMA checkpoint in the original payload format (`ema` state dictionary). The recorded experiment used update 155,000, selected by dev DreamSim. Run from the release root:

```bash
python scripts/prepare_evaluation.py --method rae_nwm --scene-root "$SCENE_ROOT" \
  --checkpoint "$RAE_CHECKPOINT" --rae-root "$PWD/third_party/rae_eval_runtime" \
  --output runs/prepared/rae_nwm
python scripts/evaluate.py --method rae_nwm --config runs/prepared/rae_nwm/evaluate.json \
  --episodes 1 --seed 0 --output runs/rae_nwm_one_task
```

For every baseline, change `--episodes 1` to `--episodes 150` and choose a new result directory to evaluate the complete fixed task set. Use seeds 0, 1 and 2 in separate runs.
