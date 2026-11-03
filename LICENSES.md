# License and data notes

## Project license

This repository is licensed under the GNU Affero General Public License, version three; see [LICENSE](LICENSE). This is a deliberate copyleft choice aligned with the YOLOv5 source revision used by the notebook.

## YOLOv5

The notebook fetches, but does not vendor, YOLOv5 commit `f9578796b8a64889aa9e66dc06914d19f40d338e`. The exact revision's upstream `LICENSE` identifies GNU AGPL version three. The notebook and training script verify the full commit, required train/detect options, AutoAnchor implementation, and training log before accepting the detector run.

The pinned runtime helpers `ultralytics==8.4.164` and `ultralytics-thop==2.2.1` are also distributed under AGPL-3.0. They are installed as dependencies and are not copied into this repository.

- [Pinned YOLOv5 license](https://github.com/ultralytics/yolov5/blob/f9578796b8a64889aa9e66dc06914d19f40d338e/LICENSE)
- [Pinned training interface](https://github.com/ultralytics/yolov5/blob/f9578796b8a64889aa9e66dc06914d19f40d338e/train.py)
- [Pinned detection interface](https://github.com/ultralytics/yolov5/blob/f9578796b8a64889aa9e66dc06914d19f40d338e/detect.py)

## Datasets

The Zenodo record [10.5281/zenodo.7799387](https://zenodo.org/records/7799387) was checked against its record metadata on 2026-11-03. Its Rights field identifies **CC BY-SA 4.0**. The four downloadable `.7z` files total 57,472,189,381 bytes (about 53.55 GiB). No archive or derived training data is included in this repository. The local machine had about 20.3 GB free, so the complete record cannot be downloaded there; an archive must not be treated as integrated until its contents have actually been ingested and evaluated.

The license terms require attribution and share-alike for covered adaptations. Review the exact record and license before redistributing data or derived artifacts. Zenodo licenses are record-specific; see [Zenodo license guidance](https://help.zenodo.org/docs/deposit/describe-records/licenses/). The DeepSig SPAWC 2021 dataset has not been downloaded, and its applicable access/licensing terms have not been verified for this project.

## Python dependencies

Direct runtime and development dependencies are version-pinned in the requirements files. They are installed by the environment rather than copied into this repository. Before distributing a bundled environment, weights, or derived data, review the license notices shipped with each exact package version and any dataset license terms.
