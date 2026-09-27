# Data and experiment records

| Files | Purpose |
|---|---|
| `data_sources.json`, `trajectory_index.jsonl`, `scene_splits.json` | Released source identities and fixed partitions |
| `offline_diagnostic_pairs.json` | The fixed 4,096 dev pairs |
| `clean150_tasks.json`, `clean150_tasks.csv` | All 150 start/goal poses and orientations |
| `clean150_construction.json`, `evaluation_settings.json` | Task construction and evaluation protocol |
| `clean150/` | Sensor template and complete Habitat task ledger |
| `training_settings.json`, `encoder_assets.json` | Encoder assets, model/training configurations and checkpoint identities |
| `baseline_settings.json`, `lwm_*_centers.json` | Baseline configurations and fitted LWM codebooks |

Configuration summaries are distinct from executable templates in each method's
`configs/` directory. Use [the evaluation preparation command](../README.md) to bind local files and regenerate hashes.
