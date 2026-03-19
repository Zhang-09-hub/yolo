#!/usr/bin/env python3
"""Run inference with a trained YOLOv8 checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError as err:  # pragma: no cover
    raise SystemExit(
        "Ultralytics is not installed. Please run `pip install -r requirements.txt`."
    ) from err


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference helper for CrackForest models.")
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to model weights (e.g., logs/runs/.../weights/best.pt).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/processed/crackforest/images/test",
        help="Source path, file, directory, or glob for inference.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for inference.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device.")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("logs/inference"),
        help="Directory where predictions will be written.",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save predictions as YOLO-format txt files alongside images.",
    )
    parser.add_argument(
        "--save-crop",
        action="store_true",
        help="Save cropped detection patches for inspection.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display predictions in an OpenCV window (requires GUI support).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.weights))
    results = model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        save_crop=args.save_crop,
        show=args.show,
        project=str(args.save_dir),
        name="predictions",
    )
    print(f"Inference complete. Outputs saved to: {results[0].save_dir}")


if __name__ == "__main__":
    main()
