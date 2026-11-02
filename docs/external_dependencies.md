# External dependencies and attribution

Existing requirements describe historical environments, not a universal lockfile. Versions below are recorded evidence, not new compatibility recommendations. No environment installation or dependency declaration is changed by this phase.

## Environment inventory

| Workflow | Existing evidence | Outstanding work |
|---|---|---|
| Legacy ALBEF | Original README: PyTorch 1.8.0, Transformers 4.8.1, timm 0.4.9; legacy requirements pin Python packages | Recover successful training Python/CUDA/package export |
| Newer single-view/fusion | Current scripts use newer Pillow APIs alongside legacy model APIs | Verify compatible isolated stack; do not silently upgrade core model libraries |
| BioViL-T | `health_multimodal` inference APIs and model enums | Recover exact distribution/version, source revision and weight hashes |
| Historical segmentation/refinement | `torchxrayvision.baseline_models.chestx_det.PSPNet` | Recover package and segmentation-weight revisions |
| Analysis | Pointing notebook declares Python 3.10+; uses scientific Python | Separate analysis environment and actual successful versions |
| Cog / inherited tasks | `cog.yaml` specifies Python 3.8, CUDA 11.1, torch 1.10.0, torchvision 0.11.1, timm 0.4.12 | Keep distinct from thesis training; inventory YAML/Java/optional dependencies |

[Legacy requirements](../requirements_albef_legacy.txt) include NumPy 1.21.6, pandas 1.5.3, Pillow 8.4.0, Transformers 4.8.1, tokenizers 0.10.3 and timm 0.4.9. [The other freeze](../requirements.txt) contains a different CUDA/notebook stack, including torch 1.8.0+cu111. Do not concatenate them or represent either as an exact final thesis environment.

Import spellings differ from distribution names: PIL/Pillow, cv2/OpenCV, sklearn/scikit-learn and skimage/scikit-image. `iterstrat` is used for optional multi-label splitting. Apex optimizers, TensorFlow conversion and Cog are conditional paths, not requirements for every experiment. Resolve package names and successful versions explicitly before producing locks.

## Upstream code and model assets

| Component | Verified reference | Missing provenance |
|---|---|---|
| ALBEF | [Paper](https://arxiv.org/abs/2107.07651), Salesforce notices and [license](../LICENSE.txt) | Exact upstream base revision; complete local patch inventory |
| DeiT initialization | [Explicit checkpoint URL](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth); source enables hash checking | Actual downloaded asset hash/cache inventory |
| BERT | `bert-base-uncased`; [vocabulary URL in local tokenizer](https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt) | Exact model/tokenizer snapshot and asset hashes |
| Released ALBEF initialization | [14M checkpoint URL](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF.pth) in inherited README | Which release initialized each thesis run |
| BioViL-T | `health_multimodal` API and BIOVIL_T selectors in extractors | Verified source repository/revision, package version, downloaded model IDs/hashes |
| CheXzero | External-output comparison notebook | Complete upstream model/extraction/evaluation implementation and checkpoints |
| CheXmask | Streaming CSV, RLE, SQLite and cached-mask implementation | Exact source release, license and source checksums |
| TorchXRayVision | PSPNet model class and saved prior metadata | Package revision and weights checksum |
| Optimizers/schedulers | Embedded attribution/URLs in source | Per-file source revision, local modifications and notices |
| RefCOCO/VQA tools | Vendored Python and Java artifacts | Complete license/version inventory and inherited-task requirements |

URLs above were present in repository evidence; download availability was not checked in this documentation phase. A mutable `main` asset reference is not a pinned model revision.

## Pinning policy for the next phase

Record verified upstream URL, commit/release, license, local modifications, environment and asset checksum in an external-dependency manifest. Prefer pinned installation or a submodule for unchanged integrations. Vendor modified code only with retained notices and a patch inventory. Never substitute an arbitrary current upstream commit and call it the thesis version.

Do not download weights with TLS or checksum verification disabled. Keep credentials out of configuration, documentation and logs. Missing external implementation/version information remains an explicit gap, not a guessed URL or package pin.
