#!/usr/bin/env python3
"""Evaluate trained YOLOv8 weights on CrackForest splits."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

try:
    from ultralytics import YOLO
except ImportError as err:  # pragma: no cover
    raise SystemExit(
        "Ultralytics is not installed. Run `pip install -r requirements.txt` first."
    ) from err


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a YOLOv8 checkpoint.")
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to the trained checkpoint (e.g., logs/runs/.../weights/best.pt).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/data/crackforest.yaml"),
        help="Dataset YAML definition.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices={"train", "val", "test"},
        help="Which split to evaluate.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for validation.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for validation.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device.")
    return parser.parse_args()


def load_data_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Dataset YAML must be a mapping.")
    return cfg


def main() -> None:
    args = parse_args()
    data_cfg = load_data_yaml(args.data)
    data_cfg["split"] = args.split

    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    print("Validation metrics:")
    for key, value in metrics.results_dict.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
