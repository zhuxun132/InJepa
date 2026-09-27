# Third-party sources

The first copyright notice in the root [LICENSE](LICENSE) identifies the InJepa authors for their contributions. The second preserves the INTACT authors' copyright for inherited portions. The original INTACT license is also retained in `injepa/LICENSE`.

Keep each component's license and copyright notice when redistributing its source.

| Component | Upstream | Revision / attribution |
|---|---|---|
| InJepa model base | [INTACT](https://github.com/zju3dv/INTACT-JEPA) | `235b6a3a92db4d0f1b3a40597ab1f407db4fd15b`; MIT notice in `injepa/LICENSE` |
| LeWM-derived modules | [LeWM](https://github.com/lucas-maes/le-wm) | MIT notice in `injepa/third_party/LeWM-MIT-LICENSE.txt` |
| V-JEPA 2.1 source and encoder release | [Meta V-JEPA](https://github.com/facebookresearch/vjepa2) | `204698b45b3712590f06245fbfba32d3be539812`; [MIT notice](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/LICENSE) and source exceptions; [encoder asset details](docs/DATA_LICENSES.md#v-jepa-21-encoder-weights) |
| LWM networks | [LWM](https://github.com/wzm206/LWM) | `5d2e0fd6c8b46850e6c0ac528d23a2db863e86d2`; `baselines/lwm_*/source/official` |
| RAE-NWM | [RAE-NWM](https://github.com/20robo/raenwm) | `0219ce41c44d515f86719dd763c1efe7c7f72519`; `baselines/rae_nwm/LICENSE` |
| NoMaD | [visualnav-transformer](https://github.com/robodhruv/visualnav-transformer) | `dca79815b704e5aa9c6bdc3082351f9e3b2848c2`; `baselines/nomad/official/LICENSE` |
| Diffusion Policy components | [diffusion_policy](https://github.com/real-stanford/diffusion_policy) | `baselines/nomad/diffusion_dependency/LICENSE` |

Use V-JEPA and Habitat from the official checkouts specified in [environment setup](docs/ENVIRONMENT.md). Follow [data and pretrained asset access](docs/DATA_LICENSES.md) for MP3D, StreamVLN and V-JEPA, including terms for derived task metadata and model artifacts. The MIT license retained from the model base applies to its covered source; third-party code, data and pretrained assets retain their own terms.
