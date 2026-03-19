###################################################################

## YOLOv8 training pipeline

This repository now bundles scripts that convert CrackForest annotations into YOLO-ready splits, train/evaluate YOLOv8n, and organize experiment logs:

- `scripts/prepare_data.py` – parse `.seg` files into YOLO bounding boxes and deterministic train/val/test splits under `data/processed/crackforest`.
- `configs/data/crackforest.yaml` & `configs/train/yolov8n_crack.yaml` – dataset declaration plus default hyper-parameters for YOLOv8n.
- `scripts/train.py`, `scripts/eval.py`, `scripts/infer.py` – thin wrappers over Ultralytics to standardize training, validation, inference, and artifact storage.
- `logs/` – canonical home for TensorBoard summaries, checkpoints, and optional Weights & Biases caches.

See [docs/USAGE.md](docs/USAGE.md) for step-by-step instructions covering environment setup, preprocessing, training, evaluation, inference, and log management.
#                                                                 #
#    CrackForest Dataset                                          #
#    Limeng Cui (lmcui932-at-163.com)                             #
#                                                                 #
###################################################################

1.Introduction.

CrackForest Dataset is an annotated road crack image database which can reflect urban road surface condition in general.

If you use this crack image dataset, we appreciate it if you cite an appropriate subset of the following papers:

@article{shi2016automatic,<br />
&nbsp;&nbsp;title={Automatic road crack detection using random structured forests},<br />
&nbsp;&nbsp;author={Shi, Yong and Cui, Limeng and Qi, Zhiquan and Meng, Fan and Chen, Zhensong},<br />
&nbsp;&nbsp;journal={IEEE Transactions on Intelligent Transportation Systems},<br />
&nbsp;&nbsp;volume={17},<br />
&nbsp;&nbsp;number={12},<br />
&nbsp;&nbsp;pages={3434--3445},<br />
&nbsp;&nbsp;year={2016},<br />
&nbsp;&nbsp;publisher={IEEE}<br />
}

@inproceedings{cui2015pavement,<br />
&nbsp;&nbsp;title={Pavement Distress Detection Using Random Decision Forests},<br />
&nbsp;&nbsp;author={Cui, Limeng and Qi, Zhiquan and Chen, Zhensong and Meng, Fan and Shi, Yong},<br />
&nbsp;&nbsp;booktitle={International Conference on Data Science},<br />
&nbsp;&nbsp;pages={95--102},<br />
&nbsp;&nbsp;year={2015},<br />
&nbsp;&nbsp;organization={Springer}<br />
}

###################################################################

2.License.

The dataset is made available for non-commercial research purposes only.

###################################################################

3.History.

Version 1.0 (2015/09/29)
 - initial version

###################################################################
