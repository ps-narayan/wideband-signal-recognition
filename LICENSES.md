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

The current milestone uses only generated synthetic signals and does not download or redistribute external recordings. The project owner will verify the Zenodo companion record's current file size and record-level license before use. The stated CC BY-SA version is not independently confirmed here because the record metadata endpoint was inaccessible; do not rely on it until the owner checks the record's Rights field. DeepSig access and licensing are also unverified and remain optional.

Zenodo's own documentation says licenses are record-specific and points users to each record's Rights section; a repository-wide default must not be assumed. See [Zenodo license guidance](https://help.zenodo.org/docs/deposit/describe-records/licenses/) and the [companion record DOI](https://doi.org/10.5281/zenodo.7799387).

## Python dependencies

Direct runtime and development dependencies are version-pinned in the requirements files. They are installed by the environment rather than copied into this repository. Before distributing a bundled environment, weights, or derived data, review the license notices shipped with each exact package version and any dataset license terms.
