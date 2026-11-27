#!/usr/bin/env python3
"""Count GQA probing_5k bounding boxes within candidate absolute-area ranges.

Read the bbox_instances.csv created by analyze_gqa_object_sizes.py. Count
object instances separately for train/val/test; no annotation files or plots
are modified. The candidate cutoffs do NOT assign permanent size labels.

Usage, from the repository root:
    python scripts/gqa_probing/check_gqa_size_thresholds.py
    python scripts/gqa_probing/check_gqa_size_thresholds.py \
        --small-max-px2 1000 --medium-max-px2 10000

Default ranges:
    Small:  0 < area <= 1000 px^2
    Medium: 1000 < area <= 10000 px^2
    Large:  area > 10000 px^2

Output:
    outputs/gqa_probing/size_analysis/tables/size_group_counts.csv

This script only evaluates candidate boundaries. It does not modify scene
graphs or bbox_instances.csv, and it does not draw additional figures.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

SPLITS = ("train", "val", "test")
GROUPS = ("Small", "Medium", "Large")


def finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0


def assign_group(area: float, small_max: float, medium_max: float) -> str:
    if area <= small_max:
        return "Small"
    if area <= medium_max:
        return "Medium"
    return "Large"


def read_counts(path: Path, small_max: float, medium_max: float) -> dict[str, Counter]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run analyze_gqa_object_sizes.py first."
        )
    counts = {split: Counter() for split in SPLITS}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "area_px2"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        for line_num, row in enumerate(reader, start=2):
            split = (row["split"] or "").strip().lower()
            if split not in counts:
                raise ValueError(f"Unexpected split {split!r} in line {line_num}")
            try:
                area = float(row["area_px2"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid area in line {line_num}: {row['area_px2']!r}") from exc
            if not finite_positive(area):
                raise ValueError(f"Area must be finite and positive in line {line_num}: {area}")
            counts[split][assign_group(area, small_max, medium_max)] += 1
    for split in SPLITS:
        if sum(counts[split].values()) == 0:
            raise ValueError(f"No bounding boxes found for split {split!r}")
    return counts


def create_rows(counts: dict[str, Counter], small_max: float, medium_max: float) -> list[dict]:
    rows = []
    totals = Counter()
    for split in SPLITS:
        totals.update(counts[split])
    boundaries = {
        "Small": (0, small_max),
        "Medium": (small_max, medium_max),
        "Large": (medium_max, ""),
    }
    for split, group_counts in [*((s, counts[s]) for s in SPLITS), ("total", totals)]:
        total = sum(group_counts.values())
        for group in GROUPS:
            lower, upper = boundaries[group]
            count = group_counts[group]
            rows.append({
                "split": split,
                "size_group": group.lower(),
                "lower_bound_exclusive_px2": lower,
                "upper_bound_inclusive_px2": upper,
                "bbox_count": count,
                "total_bboxes_split": total,
                "percentage": round(100.0 * count / total, 4),
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-csv", type=Path, default=Path(
        "outputs/gqa_probing/size_analysis/tables/bbox_instances.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path(
        "outputs/gqa_probing/size_analysis/tables/size_group_counts.csv"))
    parser.add_argument("--small-max-px2", type=float, default=1000.0)
    parser.add_argument("--medium-max-px2", type=float, default=10000.0)
    args = parser.parse_args()

    if not (finite_positive(args.small_max_px2) and
            finite_positive(args.medium_max_px2) and
            args.small_max_px2 < args.medium_max_px2):
        parser.error("Require 0 < --small-max-px2 < --medium-max-px2 (finite values)")

    counts = read_counts(args.input_csv, args.small_max_px2, args.medium_max_px2)
    rows = create_rows(counts, args.small_max_px2, args.medium_max_px2)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Candidate thresholds: Small <= {args.small_max_px2:g} px^2; "
          f"Medium <= {args.medium_max_px2:g} px^2; Large > {args.medium_max_px2:g} px^2")
    print(f"{'Split':<12}{'Small':>19}{'Medium':>19}{'Large':>19}{'Total':>12}")
    for split in (*SPLITS, "total"):
        subset = [r for r in rows if r["split"] == split]
        cells = [f'{r["bbox_count"]:,} ({r["percentage"]:.2f}%)' for r in subset]
        print(f'{split:<12}{cells[0]:>19}{cells[1]:>19}{cells[2]:>19}{subset[0]["total_bboxes_split"]:>12,}')
    print(f"SAVED {args.output_csv}")
    print("NOTE: Counts are for object instances, not distinct images. "
          "No size labels were written into annotations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
