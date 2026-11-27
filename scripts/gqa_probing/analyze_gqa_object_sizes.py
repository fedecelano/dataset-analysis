#!/usr/bin/env python3
"""Describe absolute GQA probing_5k bounding-box dimensions across train/val/test.

This is a DESCRIPTIVE analysis only: NO size-group labels, candidate cutoffs,
tertiles, or thresholds are computed or assigned. The aim is to inspect the
real distributions first and choose small/medium/large boundaries later.

Every retained object instance contributes one rectangular bounding box:
    area_px2 = width_px * height_px
The area is expressed in ORIGINAL annotation pixels squared, not normalized
by image dimensions. It measures the bounding rectangle, not foreground pixels.

Outputs, by default:
  outputs/gqa_probing/size_analysis/
    size_analysis_summary.json
    tables/bbox_instances.csv
    tables/invalid_bboxes.csv
    tables/split_statistics.csv
    tables/area_quantiles.csv
    tables/macro_class_statistics.csv
    tables/original_class_statistics.csv
    plots/bbox_area_distribution_by_split.png

Only ONE figure is produced per execution (potentially saved in several formats):
a comparative histogram of the train/val/test area distributions, using common
log-spaced bins and percentages of object instances PER SPLIT.

Example (from the repository root):
    python scripts/gqa_probing/analyze_gqa_object_sizes.py
    python scripts/gqa_probing/analyze_gqa_object_sizes.py --format png pdf

Dependencies: numpy, pandas, matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd

LOG = logging.getLogger("gqa_object_sizes")
SPLITS = ("train", "val", "test")
COLORS = {"train": "#2864A0", "val": "#D3922D", "test": "#2C8876"}
PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
INSTANCE_FIELDS = (
    "split", "image_id", "object_id", "macro_class", "original_class",
    "x_px", "y_px", "width_px", "height_px", "area_px2",
    "equivalent_side_px", "aspect_ratio",
)
INVALID_FIELDS = ("split", "image_id", "object_id", "reason", "raw_w", "raw_h")


def positive_finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        val = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return val if math.isfinite(val) and val > 0 else None


def finite_coordinate(value: Any) -> float | str:
    if value is None or isinstance(value, bool):
        return ""
    try:
        val = float(value)
    except (TypeError, ValueError, OverflowError):
        return ""
    return val if math.isfinite(val) else ""


def safe_label(value: Any) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else "[unknown]"


def parse_split(path: Path, split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Read GQA graphs; validate width/height but do not clip to image boundaries."""
    with path.open("r", encoding="utf-8") as handle:
        graphs = json.load(handle)
    if not isinstance(graphs, dict):
        raise ValueError(f"Expected a mapping image_id -> scene graph in {path}")

    records: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for image_id, graph in graphs.items():
        if not isinstance(graph, dict):
            LOG.warning("Ignoring malformed graph %s/%s", split, image_id)
            continue
        objects = graph.get("objects", {})
        if isinstance(objects, dict):
            iterator = objects.items()
        elif isinstance(objects, list):
            iterator = ((str(i), obj) for i, obj in enumerate(objects))
        else:
            LOG.warning("Ignoring malformed objects collection %s/%s", split, image_id)
            continue

        for object_id, obj in iterator:
            base = {"split": split, "image_id": str(image_id), "object_id": str(object_id)}
            if not isinstance(obj, dict):
                invalid.append({**base, "reason": "invalid_object_record", "raw_w": "", "raw_h": ""})
                continue
            raw_w, raw_h = obj.get("w"), obj.get("h")
            w, h = positive_finite(raw_w), positive_finite(raw_h)
            if w is None or h is None:
                invalid.append({**base, "reason": "missing_non_numeric_or_nonpositive_dimensions",
                                "raw_w": str(raw_w), "raw_h": str(raw_h)})
                continue
            area = w * h
            if not math.isfinite(area) or area <= 0:
                invalid.append({**base, "reason": "invalid_area",
                                "raw_w": str(raw_w), "raw_h": str(raw_h)})
                continue
            records.append({
                **base,
                "macro_class": safe_label(obj.get("name")),
                "original_class": safe_label(obj.get("original_name", obj.get("name"))),
                "x_px": finite_coordinate(obj.get("x")),
                "y_px": finite_coordinate(obj.get("y")),
                "width_px": w,
                "height_px": h,
                "area_px2": area,
                "equivalent_side_px": math.sqrt(area),
                "aspect_ratio": w / h,
            })
    return records, invalid, len(graphs)


def statistics(values: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"count": 0}
    result: dict[str, float | int] = {
        "count": int(arr.size),
        "min_px2": float(arr.min()),
        "mean_px2": float(arr.mean()),
        "median_px2": float(np.median(arr)),
        "std_px2": float(arr.std(ddof=0)),
        "max_px2": float(arr.max()),
    }
    for percentile in PERCENTILES:
        result[f"p{percentile}_px2"] = float(np.percentile(arr, percentile))
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    LOG.info("SAVED %s", path)


def per_class_statistics(df: pd.DataFrame, column: str) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    for (split, label), sub in df.groupby(["split", column], sort=True):
        item = {"split": split, column: label, "images_with_class": int(sub.image_id.nunique())}
        item.update(statistics(sub.area_px2.to_numpy(dtype=float)))
        out.append(item)
    return pd.DataFrame(out).sort_values(
        ["split", "count", column], ascending=[True, False, True]
    ) if out else pd.DataFrame(columns=["split", column, "images_with_class", "count"])


def format_tick(x: float, _pos: int) -> str:
    if x < 1:
        return f"{x:g}"
    return f"{x:,.0f}"


def plot_comparison(df: pd.DataFrame, plot_dir: Path, bins: int,
                    formats: list[str], dpi: int) -> None:
    """One figure, 3 curves, same logarithmic area bins for every split."""
    arrays = {split: df.loc[df.split == split, "area_px2"].to_numpy(dtype=float)
              for split in SPLITS}
    all_areas = df.area_px2.to_numpy(dtype=float)
    min_area, max_area = float(all_areas.min()), float(all_areas.max())
    # Account for the degenerate case where every bounding box has identical area.
    if min_area == max_area:
        edges = np.geomspace(min_area / 1.5, max_area * 1.5, bins + 1)
    else:
        edges = np.geomspace(min_area, max_area, bins + 1)

    fig, ax = plt.subplots(figsize=(13.33, 7.5), facecolor="white")
    fig.subplots_adjust(left=0.105, right=0.97, bottom=0.19, top=0.77)
    ax.set_facecolor("white")

    for split in SPLITS:
        values = arrays[split]
        if values.size == 0:
            continue
        counts, _ = np.histogram(values, bins=edges)
        pct = counts * (100.0 / values.size)
        median = float(np.median(values))
        ax.stairs(pct, edges, color=COLORS[split], linewidth=2.7,
                  label=f"{split.capitalize()} · {values.size:,} boxes · median {median:,.0f} px²")

    ax.set_xscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, color="#E9EEF4", linewidth=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#BAC5D2")
    ax.tick_params(axis="both", labelsize=11, length=0, pad=8)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=9))
    ax.xaxis.set_major_formatter(FuncFormatter(format_tick))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Absolute bounding-box area (px²) · logarithmic scale",
                  fontsize=13, labelpad=15)
    ax.set_ylabel("Boxes per area interval (% of each split)", fontsize=12, labelpad=13)
    ax.legend(loc="upper right", frameon=False, fontsize=11, handlelength=2.8)

    fig.text(0.105, 0.937, "Bounding-box area distributions across dataset splits",
             color="#17253A", fontsize=22, fontweight="bold", ha="left", va="top")
    fig.text(0.105, 0.884,
             "GQA probing_5k · Original pixel area (width × height) · Train / Val / Test",
             color="#516274", fontsize=12, ha="left", va="top")
    fig.text(0.105, 0.839,
             "Each curve shows the percentage of that split's boxes in the same area intervals.",
             color="#627386", fontsize=11, ha="left", va="top")
    fig.text(0.105, 0.052,
             "Descriptive analysis only: no small / medium / large boundaries have been selected. "
             "Areas are not normalized by image size.",
             color="#627386", fontsize=10, ha="left", va="bottom")

    for ext in formats:
        output = plot_dir / f"bbox_area_distribution_by_split.{ext}"
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
        LOG.info("SAVED %s", output)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations-dir", type=Path,
                        default=Path("data/gqa_probing/subsets/probing_5k/annotations"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/gqa_probing/size_analysis"))
    parser.add_argument("--bins", type=int, default=55,
                        help="Number of equal-width LOG-AREA bins shared by all splits (default: 55).")
    parser.add_argument("--format", nargs="+", choices=("png", "pdf", "svg"), default=["png"])
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.bins < 10 or args.bins > 200:
        parser.error("--bins must be between 10 and 200")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")

    tables_dir = args.output_dir / "tables"
    plots_dir = args.output_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    image_counts: dict[str, int] = {}
    for split in SPLITS:
        path = args.annotations_dir / f"{split}_sceneGraphs.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing annotations: {path}")
        records, bad, images = parse_split(path, split)
        rows.extend(records)
        invalid.extend(bad)
        image_counts[split] = images
        LOG.info("READ %s: %d images, %d valid bboxes, %d invalid bboxes",
                 split, images, len(records), len(bad))
        if not records:
            raise ValueError(f"No valid bbox areas for split {split}")

    df = pd.DataFrame.from_records(rows, columns=INSTANCE_FIELDS)
    write_csv(tables_dir / "bbox_instances.csv", rows, INSTANCE_FIELDS)
    write_csv(tables_dir / "invalid_bboxes.csv", invalid, INVALID_FIELDS)

    summary_rows = []
    quantiles = []
    for split in SPLITS:
        sub = df.loc[df.split == split]
        areas = sub.area_px2.to_numpy(dtype=float)
        stat = {
            "split": split,
            "images": image_counts[split],
            "images_with_valid_bbox": int(sub.image_id.nunique()),
            "invalid_bboxes": sum(1 for item in invalid if item["split"] == split),
            "median_width_px": float(sub.width_px.median()),
            "median_height_px": float(sub.height_px.median()),
        }
        stat.update(statistics(areas))
        summary_rows.append(stat)
        for q in PERCENTILES:
            quantiles.append({"split": split, "percentile": q,
                              "area_px2": float(np.percentile(areas, q))})

    pd.DataFrame(summary_rows).to_csv(tables_dir / "split_statistics.csv", index=False)
    pd.DataFrame(quantiles).to_csv(tables_dir / "area_quantiles.csv", index=False)
    per_class_statistics(df, "macro_class").to_csv(
        tables_dir / "macro_class_statistics.csv", index=False)
    per_class_statistics(df, "original_class").to_csv(
        tables_dir / "original_class_statistics.csv", index=False)
    for name in ("split_statistics.csv", "area_quantiles.csv", "macro_class_statistics.csv",
                 "original_class_statistics.csv"):
        LOG.info("SAVED %s", tables_dir / name)

    plot_comparison(df, plots_dir, args.bins, args.format, args.dpi)

    summary = {
        "methodology": {
            "dataset": "GQA probing_5k prepared scene graphs",
            "measure": "absolute bounding-box rectangle area in original annotation pixels squared: w*h",
            "not_measured": "foreground object pixel count, image-normalized area, resized-crop pixel count",
            "bbox_clipping": "not clipped to image bounds; annotation widths/heights used as stored",
            "statistical_unit": "object instances; each box counted once within its split",
            "plot_normalization": "percent of bounding boxes within each split per shared log-area bin",
            "classification": "none; no size thresholds or small/medium/large assignments",
            "invalid_dimensions": "missing, nonnumeric, nonpositive or nonfinite width/height excluded and reported",
        },
        "annotation_dir": str(args.annotations_dir),
        "splits": summary_rows,
        "invalid_bboxes_total": len(invalid),
        "plot": "plots/bbox_area_distribution_by_split",
        "note": "Previously generated charts or candidate-threshold files from older script versions are not deleted.",
    }
    summary_path = args.output_dir / "size_analysis_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    LOG.info("SAVED %s", summary_path)
    LOG.info("DONE: %d valid bounding boxes across %d annotated images",
             len(df), sum(image_counts.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
