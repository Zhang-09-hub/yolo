#!/usr/bin/env python3
"""Wrapper around Ultralytics YOLOv8 training with repository defaults."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

try:
    from ultralytics import YOLO
except ImportError as err:  # pragma: no cover - dependency message
    raise SystemExit(
        "Ultralytics is not installed. Please run `pip install -r requirements.txt`."
    ) from err


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on CrackForest.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train/yolov8n_crack.yaml"),
        help="YAML file containing default training hyper-parameters.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Override dataset YAML path (defaults to the value in the config).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size.")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="CUDA device spec passed to Ultralytics (e.g., '0' or 'cpu').",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Experiment name stored inside project/run directories.",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help="Directory that will store Ultralytics runs (defaults to logs/runs).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from a checkpoint path or run name.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow existing runs to be overwritten.",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Set WANDB_MODE=disabled to keep training fully offline.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a mapping. Received: {config}")
    return config


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    overrides = {k: v for k, v in config.items() if k != "model"}
    if args.data:
        overrides["data"] = args.data
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch is not None:
        overrides["batch"] = args.batch
    if args.imgsz is not None:
        overrides["imgsz"] = args.imgsz
    if args.device:
        overrides["device"] = args.device
    if args.project_dir:
        overrides["project"] = str(args.project_dir)
    overrides.setdefault("project", "logs/runs")
    overrides["exist_ok"] = args.exist_ok or overrides.get("exist_ok", False)
    if args.run_name:
        overrides["name"] = args.run_name
    if args.resume:
        overrides["resume"] = args.resume

    if args.disable_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    model_path = config.get("model", "yolov8n.pt")
    model = YOLO(model_path)
    results = model.train(**overrides)
    print(f"Training complete. Run directory: {results.save_dir}")


if __name__ == "__main__":
    main()
