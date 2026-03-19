# CrackForest → YOLOv8n workflow

This repository now contains a repeatable pipeline that takes the original CrackForest dataset (`image/`, `seg/`, `groundTruth/`) and converts it into YOLO-ready splits, trains the lightweight YOLOv8n detector, and records artifacts plus logs for future experiments.

## Repository layout

```
configs/
  data/crackforest.yaml      # Dataset declaration consumed by Ultralytics
  train/yolov8n_crack.yaml   # Default hyper-parameters for YOLOv8n
data/
  raw/.gitkeep               # Place optional mirrored raw datasets here
  processed/.gitkeep         # Output of scripts/prepare_data.py
docs/USAGE.md                # You are here
logs/                        # TensorBoard summaries, checkpoints, eval logs
scripts/                     # CLI helpers (prepare_data, train, eval, infer)
```

## 1. Environment setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Tip:** Ultralytics automatically logs to TensorBoard when `tensorboard` is installed. Weights & Biases logging is available once you `pip install wandb` (already included here) and set `WANDB_API_KEY`.

## 2. Preparing the dataset

1. Make sure the original directories (`image/` with `.jpg` files and `seg/` with `.seg` run-length encodings) are available. They can live in the repository root (already the case) or under `data/raw/CrackForest`.
2. Run the conversion script (the example below keeps the existing layout by pointing `--raw-root` to the repo root):

```powershell
python scripts/prepare_data.py --raw-root . --clear-output --save-masks
```

Key flags:

- `--train-ratio` / `--val-ratio`: tweak split sizes (defaults: 70/20/10).
- `--min-area`: filter tiny crack fragments before generating YOLO boxes.
- `--save-masks`: optional PNG masks for visual QA (`data/processed/.../masks`).

The resulting structure follows the Ultralytics convention:

```
data/processed/crackforest/
  images/{train,val,test}/*.jpg
  labels/{train,val,test}/*.txt   # YOLO-format bounding boxes
  masks/{split}/*.png             # optional debug masks
```

## 3. Training YOLOv8n

Use the provided hyper-parameter file as a starting point:

```powershell
python scripts/train.py --config configs/train/yolov8n_crack.yaml
```

Useful overrides:

- `--batch`, `--epochs`, `--imgsz`, `--device` to adapt hardware.
- `--run-name my-exp` to distinguish experiments.
- `--project-dir logs/runs` (default) keeps artifacts inside versioned logs.
- `--disable-wandb` switches to offline mode even when `WANDB_API_KEY` is set.

Training outputs:

- `logs/runs/<name>/weights/best.pt`: best-performing weights.
- `logs/runs/<name>/weights/last.pt`: final epoch.
- `logs/runs/<name>/results.csv`: per-epoch metrics.
- `logs/tensorboard`: mirrored summaries for `tensorboard --logdir logs/runs`.
- `logs/wandb`: local cache when W&B is enabled.

## 4. Monitoring and logging

- **TensorBoard:** `tensorboard --logdir logs/runs --port 6006`
- **Weights & Biases:** set `setx WANDB_API_KEY "<token>"` (PowerShell) and re-run training. Ultralytics automatically streams metrics; runs appear under the project name taken from the YAML (`yolov8n-crackforest`).
- **Checkpoints:** copy relevant `.pt` files from `logs/runs/.../weights/` into `logs/checkpoints/` if you want to curate long-term snapshots.

## 5. Evaluation

```powershell
python scripts/eval.py --weights logs/runs/yolov8n-crackforest/weights/best.pt --split test
```

Options:

- `--data`: point to a different dataset YAML if you keep multiple variants.
- `--imgsz` / `--batch` / `--device`: evaluation-time overrides.

The script prints the `metrics.results_dict` contents so you can archive them inside experiment logs.

## 6. Inference

```powershell
python scripts/infer.py ^
  --weights logs/runs/yolov8n-crackforest/weights/best.pt ^
  --source data/processed/crackforest/images/test ^
  --save-dir logs/inference
```

Flags:

- `--save-txt` writes YOLO-format predictions.
- `--save-crop` exports cropped detections for visual QA.
- `--show` pops up an OpenCV window (requires local GUI).

Outputs are stored under `logs/inference/predictions/`.

## 7. Recommended workflow checklist

1. `pip install -r requirements.txt`
2. `python scripts/prepare_data.py --raw-root . --clear-output`
3. Verify `data/processed/crackforest` and update `configs/data/crackforest.yaml` only if you relocate it.
4. `python scripts/train.py --config configs/train/yolov8n_crack.yaml`
5. Monitor via TensorBoard or W&B.
6. `python scripts/eval.py --weights <run>/weights/best.pt --split test`
7. `python scripts/infer.py --weights <run>/weights/best.pt --source <custom images>`

## Notes

- The CrackForest license only permits **non-commercial research**. Keep this notice when redistributing derived annotations.
- Both data preparation and training honor a deterministic seed (`--seed` in preparation, `seed` in the training YAML) to ensure reproducibility, but GPU kernels might still introduce minor nondeterminism.
- Feel free to create additional configs under `configs/train/` (e.g., `yolov8s_crack.yaml`) and pass them to `scripts/train.py`.
