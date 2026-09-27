# Use the fixed Clean150 tasks

Clean150 is reconstructed by loading the fixed episode file into Habitat with the matching scene assets. It already contains the selected 150 start/goal pairs and viewing directions, so evaluation uses these records directly rather than repeating candidate sampling or manual filtering.

Follow [Data access and use](DATA_LICENSES.md) to obtain the licensed MP3D Habitat assets, retain the downloaded `mp3d_habitat.zip` for `MP3D_ARCHIVE`, and place the scene files under `SCENE_ROOT` with this layout:

```text
scene_root/
  mp3d/
    <scene_id>/
      <scene_id>.glb
      <scene_id>.navmesh
```

The 11 scene IDs are `2azQ1b91cZZ`, `8194nk5LbLH`, `EU6Fwq7SyZv`, `QUCTc6BB5sX`, `TbHJrupSAjP`, `X7HyMhZNoso`, `Z6MFQCViBuw`, `oLBMNvg9in8`, `pLe4wQe7qrG`, `x8F5xyUWy9e` and `zsNo4HB9uLZ`.

`data/clean150_tasks.json` and `.csv` provide all 150 tasks, their scene IDs, start and goal positions, start rotations, goal-image rotations and initial geodesic distances. Coordinates use the Habitat world frame in metres; quaternions use `[x, y, z, w]`. `data/clean150_construction.json` records the generation/filtering rules and scene asset identities. From the repository root, compare the local GLB and navigation-mesh SHA-256 values with its `generation.scene_assets` entries:

```bash
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["SCENE_ROOT"]).expanduser().resolve()
assets = json.loads(Path("data/clean150_construction.json").read_text())["generation"]["scene_assets"]
for scene, expected in sorted(assets.items()):
    for extension in ("glb", "navmesh"):
        path = root / "mp3d" / scene / f"{scene}.{extension}"
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected[f"{extension}_sha256"]:
            raise SystemExit(f"Scene asset SHA-256 mismatch: {path}")
        print(f"OK {scene}/{path.name}")
print(f"Verified {len(assets)} scenes and {2 * len(assets)} files.")
PY
```

The evaluator reads `data/clean150/episodes150.json.gz`. This preserves the evaluated episode order and all goal-view metadata. `info.goal_image_rotation`, `goal_view_mode`, `goal_approach_anchor` and `approach_anchor_arc_m` are consumed by `ViewAlignedImageGoalSensor`. Keep these fields together and use the supplied sensor when rendering the goal image. Native PointNav goals use `position` and `radius`.

With the evaluation environment and encoder installed, follow the [README preparation and evaluation commands](../README.md#2-prepare-local-evaluation-configurations). Keep `PYTHONHASHSEED=0` in that shell. `scripts/prepare_evaluation.py` binds `scene_root` to the task file and creates a local sensor configuration; the evaluator renders the initial and goal views from their stored poses and loads E12. The sensor settings are 640×480 RGB, 79° horizontal field of view, sensor height 1.25 m, agent height 1.5 m and radius 0.1 m. Discrete actions advance 0.25 m or turn 15°. Evaluation allows 200 environment steps and uses a 1 m geodesic arrival radius.

`--episodes 1` selects the first ledger entry, and `--episodes 150` selects the entire fixed list. Seeds change method randomness while retaining these tasks and poses. Start every run with a fresh output directory.

The file checksum used by the evaluators is:

```text
3988c711cade807cc08bf6b93fa6378657d5c4209d0ad8215fb9874138938258  data/clean150/episodes150.json.gz
```
