#!/usr/bin/env python3
"""Convert CrackForest annotations into YOLO-friendly train/val/test splits."""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

Sample = Tuple[Path, Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare CrackForest dataset for YOLOv8 training."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("."),
        help="Path containing the original 'image' and 'seg' folders.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="image",
        help="Relative directory (inside raw-root) with source images.",
    )
    parser.add_argument(
        "--seg-dir",
        type=str,
        default="seg",
        help="Relative directory (inside raw-root) with .seg annotation files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/crackforest"),
        help="Destination directory that will contain images/ and labels/ folders.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Portion of samples assigned to the training split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Portion of samples assigned to the validation split (remainder goes to test).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic splits.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=20,
        help="Minimum contour area (in pixels) kept when extracting bounding boxes.",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="When set, binary masks are exported alongside YOLO labels for debugging.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Remove existing output directory before writing new data.",
    )
    parser.add_argument(
        "--img-ext",
        type=str,
        default=".jpg",
        help="Image extension to use when copying files to the processed dataset.",
    )
    return parser.parse_args()


def collect_samples(raw_root: Path, image_dir: str, seg_dir: str) -> List[Sample]:
    image_path = raw_root / image_dir
    seg_path = raw_root / seg_dir
    if not image_path.exists():
        raise FileNotFoundError(f"Image directory not found: {image_path}")
    if not seg_path.exists():
        raise FileNotFoundError(f"Segmentation directory not found: {seg_path}")

    samples: List[Sample] = []
    for img_file in sorted(image_path.glob("*")):
        if not img_file.is_file():
            continue
        seg_file = seg_path / f"{img_file.stem}.seg"
        if seg_file.exists():
            samples.append((img_file, seg_file))
    if not samples:
        raise RuntimeError(
            f"No valid image/seg pairs discovered under {image_path} and {seg_path}."
        )
    return samples


def parse_segmentation(seg_file: Path) -> np.ndarray:
    width = height = None
    reading_data = False
    runs: List[Tuple[int, int, int, int]] = []

    with seg_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            lower = line.lower()
            if not reading_data:
                if lower.startswith("width"):
                    _, value = line.split(maxsplit=1)
                    width = int(value)
                    continue
                if lower.startswith("height"):
                    _, value = line.split(maxsplit=1)
                    height = int(value)
                    continue
                if lower == "data":
                    reading_data = True
                    continue
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(
                    f"Malformed data line '{line}' in segmentation file {seg_file}"
                )
            a, b, c, d = map(int, parts)
            # CrackForest commonly stores runs as: label row col_start col_end.
            # Keep backward compatibility with alternative exports that may use:
            # row col_start label col_end.
            if a in (0, 1) and 0 <= b < (height or 0):
                label, row, col_start, col_end = a, b, c, d
            else:
                row, col_start, label, col_end = a, b, c, d
            runs.append((row, col_start, label, col_end))

    if width is None or height is None:
        raise ValueError(f"Missing width/height metadata in {seg_file}")

    mask = np.zeros((height, width), dtype=np.uint8)
    for row, col_start, label, col_end in runs:
        col_start = max(0, col_start)
        col_end = min(width - 1, col_end)
        if row < 0 or row >= height:
            continue
        if col_end < col_start:
            continue
        mask[row, col_start : col_end + 1] = 1 if label > 0 else 0
    return mask


def mask_to_bboxes(mask: np.ndarray, min_area: int) -> List[Tuple[int, int, int, int]]:
    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        return []
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, x + w, y + h))
    return boxes


def format_yolo_line(
    bbox: Tuple[int, int, int, int], width: int, height: int, class_id: int = 0
) -> str:
    x_min, y_min, x_max, y_max = bbox
    box_width = max(1, x_max - x_min)
    box_height = max(1, y_max - y_min)
    x_center = x_min + box_width / 2.0
    y_center = y_min + box_height / 2.0
    return " ".join(
        [
            str(class_id),
            f"{x_center / width:.6f}",
            f"{y_center / height:.6f}",
            f"{box_width / width:.6f}",
            f"{box_height / height:.6f}",
        ]
    )


def split_dataset(
    samples: Sequence[Sample], train_ratio: float, val_ratio: float, seed: int
) -> Dict[str, List[Sample]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be within (0, 1).")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be within [0, 1).")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1.")

    samples_copy = list(samples)
    random.Random(seed).shuffle(samples_copy)
    total = len(samples_copy)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return {
        "train": samples_copy[:train_end],
        "val": samples_copy[train_end:val_end],
        "test": samples_copy[val_end:],
    }


def ensure_structure(root: Path, splits: Iterable[str], save_masks: bool) -> None:
    for split in splits:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        if save_masks:
            (root / "masks" / split).mkdir(parents=True, exist_ok=True)


def write_label_file(label_path: Path, yolo_lines: Sequence[str]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(yolo_lines), encoding="utf-8")


def copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def save_mask(mask: np.ndarray, mask_path: Path) -> None:
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), mask * 255)


def main() -> None:
    args = parse_args()
    samples = collect_samples(args.raw_root, args.image_dir, args.seg_dir)
    splits = split_dataset(samples, args.train_ratio, args.val_ratio, args.seed)

    if args.clear_output and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    ensure_structure(args.output_dir, splits.keys(), args.save_masks)

    stats = {split: 0 for split in splits}
    empty_labels = 0

    for split, subset in splits.items():
        progress = tqdm(subset, desc=f"Processing {split}", unit="img")
        for image_path, seg_path in progress:
            mask = parse_segmentation(seg_path)
            boxes = mask_to_bboxes(mask, args.min_area)
            yolo_lines = [format_yolo_line(box, mask.shape[1], mask.shape[0]) for box in boxes]
            if not yolo_lines:
                empty_labels += 1
            label_dest = args.output_dir / "labels" / split / f"{image_path.stem}.txt"
            write_label_file(label_dest, yolo_lines)

            img_ext = args.img_ext if args.img_ext.startswith(".") else f".{args.img_ext}"
            dest_image = args.output_dir / "images" / split / f"{image_path.stem}{img_ext}"
            copy_image(image_path, dest_image)

            if args.save_masks:
                mask_dest = args.output_dir / "masks" / split / f"{image_path.stem}.png"
                save_mask(mask, mask_dest)

            stats[split] += 1

    summary_lines = [
        "Preparation complete:",
        *(f"  {split}: {count} images" for split, count in stats.items()),
        f"  Images without positive cracks: {empty_labels}",
        f"Output directory: {args.output_dir}",
    ]
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
