# Data and pretrained asset access

Complete each provider's access procedure before downloading its data. Keep the original attribution and applicable terms with derived data and model artifacts.

## Matterport3D for Clean150

Request access through the [Matterport3D project](https://niessner.github.io/Matterport/), which requires signing its [academic-use agreement](https://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf). The agreement restricts use to non-commercial academic purposes. Sections 1 and 2.4 also cover derived information, including trained models, and require including the agreement or its link with published derived material. Substantial dataset redistribution has additional acceptance requirements specified there.

Use the authorized Habitat scene assets with the fixed task definitions in [CLEAN150.md](CLEAN150.md). Keep this agreement link with Clean150 metadata, rendered views and applicable trained-model artifacts when redistributing them.

## StreamVLN for training and development

Obtain the R2R/RxR trajectories from the [official StreamVLN dataset](https://huggingface.co/datasets/cywan/StreamVLN-Trajectory-Data) after signing in and accepting its access agreement. The access agreement identifies [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/): attribution, non-commercial use and ShareAlike apply under its terms. The dataset page also displays a different `cc-by-sa-4.0` metadata tag; follow the presented access agreement and ask the dataset maintainers to resolve any licensing ambiguity before broader reuse.

The official card identifies MP3D and R2R/RxR as upstream sources. Their applicable terms continue to apply. Use the pinned revision and file identities in `data/training_assets.json`, then follow [TRAINING.md](TRAINING.md) to build the separate training and development splits.

## V-JEPA 2.1 encoder weights

Download `vjepa2_1_vitb_dist_vitG_384.pt` from Meta's [official V-JEPA 2.1 checkpoint table](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/README.md#v-jepa-21-pretrained-checkpoints). This is the ViT-B/16 encoder at resolution 384. [ENVIRONMENT.md](ENVIRONMENT.md) provides the official download URL, pinned source revision and SHA-256 check.

The pinned upstream release declares MIT as its main project license in [LICENSE](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/LICENSE), with copyright held by Meta Platforms, Inc. and affiliates. Its [license section](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/README.md#license) identifies three source files under [Apache 2.0](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/APACHE-LICENSE). Preserve the applicable upstream copyright and license notices when redistributing covered material. The checkpoint is linked from that release; the pinned repository does not provide a separate checkpoint-specific license statement, so consult the original release terms for weight reuse.

Use this frozen encoder together with the InJepa navigation checkpoint `weights/injepa/epoch_0012.pt` under the terms below.

## InJepa E12 checkpoint

The E12 checkpoint was trained on frozen V-JEPA features of StreamVLN R2R/RxR trajectories, whose visual observations originate from MP3D. Use it for non-commercial academic research in accordance with the applicable StreamVLN and MP3D terms above. When redistributing the checkpoint, retain its [usage notice](../weights/injepa/README.md), dataset attribution and agreement links. The code's MIT license and Meta's project license do not replace these training-data terms or grant additional commercial-use rights for E12.

## Code and asset licenses

The repository's code license does not grant additional rights to third-party datasets or pretrained weights. Source components retain the notices listed in [THIRD_PARTY.md](../THIRD_PARTY.md). Dataset-derived task metadata, features and trained weights remain subject to applicable upstream terms.
