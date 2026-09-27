# Environment setup

Use Linux with an NVIDIA GPU. Select available devices with `CUDA_VISIBLE_DEVICES`.

## Habitat evaluation

Create a Python 3.9 conda environment and install the headless Bullet build of Habitat-Sim 0.2.4:

```bash
conda create -n injepa-eval python=3.9 -y
conda activate injepa-eval
conda install habitat-sim=0.2.4 withbullet headless -c conda-forge -c aihabitat
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r environments/evaluation-core.txt
mkdir -p third_party assets
git clone https://github.com/facebookresearch/habitat-lab.git third_party/habitat-lab
git -C third_party/habitat-lab checkout 1639e1ae732ba1e84199a1a04b79c7243c3f8586
python -m pip install -e third_party/habitat-lab/habitat-lab
git clone https://github.com/facebookresearch/habitat-sim.git third_party/habitat-sim
git -C third_party/habitat-sim checkout f179b584bcd713c5a2a998132211e2cae881d6d1
```

Keep these Git checkouts: the reset/replay check verifies their revisions against the installed Habitat packages. Also retain Habitat-Sim's original `.tar.bz2` package in the conda package cache; the check compares the installed Python and native-extension files with that archive. Keep it at the `package_tarball_full_path` recorded in `$CONDA_PREFIX/conda-meta/habitat-sim-0.2.4-*.json` when cleaning conda caches. Set `HABITAT_LAB_SOURCE` and `HABITAT_SIM_SOURCE` to their absolute paths and `HABITAT_PREFIX` to `$CONDA_PREFIX`.

Before running any preparation or evaluation command, set the interpreter hash seed in the shell:

```bash
export PYTHONHASHSEED=0
```

Keep this value at 0 across evaluation seeds; change method randomness through `scripts/evaluate.py --seed`.

Install the V-JEPA source and encoder weights using Meta's official release below. Follow the [V-JEPA asset and license notes](DATA_LICENSES.md#v-jepa-21-encoder-weights), retaining the upstream copyright and license notices:

```bash
git clone https://github.com/facebookresearch/vjepa2.git third_party/vjepa2
git -C third_party/vjepa2 checkout 204698b45b3712590f06245fbfba32d3be539812
curl -L --fail https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt \
  -o assets/vjepa2_1_vitb_dist_vitG_384.pt
echo '848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d  assets/vjepa2_1_vitb_dist_vitG_384.pt' | sha256sum -c -
```

The preparation command verifies the encoder and navigation checkpoint before constructing the local evaluator configurations.

## InJepa training

Use a separate Python 3.10 environment:

```bash
python3.10 -m venv .env-train
. .env-train/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r environments/training-core.txt
```

The captured training environment is also recorded in `injepa/requirements-cu124.lock`. Use `environments/training-core.txt` for the core training imports.

## Baselines

Install the baseline dependencies in the environment used by the selected method:

```bash
python -m pip install -r environments/baselines.txt
```

For LWM training, install the selected variant, for example `python -m pip install -e baselines/lwm_croco`. Use separate environments for the two LWM variants, which expose the same package names. The common evaluation launcher selects each variant's source in a separate process.

NoMaD uses `diffusers==0.27.2`, `huggingface_hub==0.24.7`, `efficientnet-pytorch`, and `warmup-scheduler`; install its local `baselines/nomad/official/train` package. RAE uses `torchdiffeq==0.2.5` and `decord==0.6.0`. Its source checkout procedure is in [BASELINES.md](BASELINES.md).
