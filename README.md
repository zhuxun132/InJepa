# InJepa

Code for **Intention First: Latent State Planning with Coupled JEPAs for Visual Navigation**.

Start with the released E12 model on one fixed Clean150 task, then run all 150 tasks. Commands below run from the repository root on Linux.

## 1. Set up the environment and assets

Follow [environment setup](docs/ENVIRONMENT.md) to install the evaluation environment, Habitat-Lab, Habitat-Sim and the V-JEPA encoder. Activate the evaluation environment, then set paths to your local assets:

```bash
export REPO="$PWD"
export SCENE_ROOT=/absolute/path/to/scene_root
export MP3D_ARCHIVE=/absolute/path/to/mp3d_habitat.zip
export HABITAT_LAB_SOURCE=/absolute/path/to/habitat-lab
export HABITAT_SIM_SOURCE=/absolute/path/to/habitat-sim
export HABITAT_PREFIX=/absolute/path/to/evaluation_environment
export VJEPA_SOURCE="$REPO/third_party/vjepa2"
export VJEPA_CHECKPOINT="$REPO/assets/vjepa2_1_vitb_dist_vitG_384.pt"
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=0
```

Keep `PYTHONHASHSEED=0` for both preparation and evaluation, including runs with `--seed 1` or `--seed 2`. The latter controls evaluation randomness separately.

`SCENE_ROOT` contains `mp3d/<scene_id>/<scene_id>.glb` and the matching `.navmesh`. Obtain the MP3D assets through the [Matterport3D dataset](https://niessner.github.io/Matterport/) and retain the downloaded `mp3d_habitat.zip` at `MP3D_ARCHIVE`. The exact 11 scenes, all 150 start/goal poses and goal-image orientations are in [data/clean150_tasks.json](data/clean150_tasks.json). Follow [Clean150 data](docs/CLEAN150.md) to check the scene files, then load the supplied task file directly to construct the same 150 tasks. It already fixes the task selection and order.

Use the E12 checkpoint for non-commercial academic research under the applicable StreamVLN and MP3D terms. Keep the [weight usage notice](weights/injepa/README.md) and linked data agreements with redistributed copies; the code's MIT license does not replace these terms.

Retrieve the E12 checkpoint using Git LFS:

```bash
git lfs install
git lfs pull --include='weights/injepa/epoch_0012.pt'
sha256sum -c weights/SHA256SUMS
```

## 2. Prepare local evaluation configurations

```bash
python scripts/prepare_evaluation.py --method injepa \
  --scene-root "$SCENE_ROOT" --mp3d-archive "$MP3D_ARCHIVE" --output runs/prepared/injepa \
  --vjepa-source "$VJEPA_SOURCE" --vjepa-checkpoint "$VJEPA_CHECKPOINT" \
  --habitat-lab-source "$HABITAT_LAB_SOURCE" \
  --habitat-sim-source "$HABITAT_SIM_SOURCE" --habitat-prefix "$HABITAT_PREFIX"
```

With the environment, scene assets, official V-JEPA encoder and E12 checkpoint in place, this verifies the checkpoint, binds local paths, validates the calibration data and runs a Habitat reset/replay check. It writes four ready-to-run configurations. Goal images are rendered from the supplied goal poses at runtime. Use a fresh output directory when preparing a different installation.

## 3. Run one task, then Clean150

```bash
python scripts/evaluate.py --method injepa \
  --config runs/prepared/injepa/full.json --episodes 1 --seed 0 \
  --output runs/injepa_full_one_task

python scripts/evaluate.py --method injepa \
  --config runs/prepared/injepa/full.json --episodes 150 --seed 0 \
  --output runs/injepa_full_clean150_seed0
```

The one-task run uses the first task in the same fixed ledger. Both commands keep the 200-step budget and 1 m arrival radius. Use `--seed 1` and `--seed 2` with separate output directories for the other evaluation seeds. Add `--dry-run` to inspect the resolved configuration and command before execution.

All four deployments use the same E12 checkpoint and K=8, H=4:

| Deployment | Prepared configuration | Candidate selection |
|---|---|---|
| Full, F→Q+G | `full.json` | Predicted endpoint and realization error, weight 1 |
| Proposal only | `p_only.json` | Proposed endpoint distance |
| Post-hoc F | `posthoc_f.json` | Predicted endpoint and realization error, weight 1 |
| F→G | `goal_gf.json` | Predicted endpoint distance |

Select a deployment by changing `--config`, for example:

```bash
python scripts/evaluate.py --method injepa \
  --config runs/prepared/injepa/posthoc_f.json --episodes 1 --seed 0 \
  --output runs/injepa_posthoc_f_one_task
```

Each run stores its resolved configuration beside the result directory. For InJepa, `result.json` contains the per-task results in its `episodes` array; the other methods write their results inside the same chosen directory. Traces and videos follow the options in the selected configuration. Choose a new result directory for each run.

## Training and baselines

- [Train InJepa](docs/TRAINING.md): prepare StreamVLN trajectories and spatial features, then train Q/A/G/F.
- [Run and train baselines](docs/BASELINES.md): LWM-CroCo, LWM-VJEPA, RAE-NWM and NoMaD.
- [Environment setup](docs/ENVIRONMENT.md): dependencies and upstream source revisions.
- [Clean150](docs/CLEAN150.md): scenes, poses, sensor configuration and task order.
- [Data and weight terms](docs/DATA_LICENSES.md): MP3D and StreamVLN access agreements, E12 usage and the V-JEPA encoder release.

| Directory | Contents |
|---|---|
| `injepa/` | Model, losses, feature caching, training and original evaluation implementations |
| `baselines/` | Baseline models, training adapters, controllers and evaluation implementations |
| `scripts/` | Local evaluation preparation and common launch commands |
| `data/` | Fixed tasks, source identities, scene partitions and experiment settings |
| `weights/injepa/` | E12 checkpoint and its training configuration |
| `configs/` | Place your resolved training configurations here |

Third-party code retains its original license and attribution; see [THIRD_PARTY.md](THIRD_PARTY.md).
