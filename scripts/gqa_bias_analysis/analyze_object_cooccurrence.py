#!/usr/bin/env python3
"""Analyze image-level object–object co-occurrence in frozen GQA probing_5k.

Levels:
  macro_object: 14 canonical object classes
  original_object: original GQA object names preserved on retained instances

For two DIFFERENT labels A and B, an image contributes once to their
unordered co-occurrence pair if it contains >=1 object of each class.
All probabilities and lift use DISTINCT IMAGES as the sample unit:
  P(B|A) = images_with_A_and_B / images_with_A
  P(B) = images_with_B / total_images
  lift(A,B) = P(A,B) / (P(A) * P(B))

Same-class multiplicity (>=2 instances of a label in one image) is reported
separately, not conflated with co-occurrence of two distinct classes. There
is NO requirement for a scene-graph relation between co-occurring objects.
The script does not alter dataset files or selected image IDs.

Run from the repository root:
  python scripts/gqa_bias_analysis/analyze_object_cooccurrence.py
  python scripts/gqa_bias_analysis/analyze_object_cooccurrence.py --min-pair-images 20
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test")
LEVELS = ("macro_object", "original_object")

PAIR_COLUMNS = [
    "split", "level", "object_a", "object_b", "total_images",
    "object_a_images", "object_b_images", "pair_images",
    "object_a_prevalence_pct", "object_b_prevalence_pct", "pair_prevalence_pct",
    "p_b_given_a_pct", "p_a_given_b_pct", "lift", "status",
]
DIRECTIONAL_COLUMNS = [
    "split", "level", "anchor_object", "context_object", "total_images",
    "anchor_images", "context_images", "pair_images", "anchor_prevalence_pct",
    "context_prevalence_pct", "pair_prevalence_pct", "p_context_given_anchor_pct",
    "p_context_pct", "lift", "status",
]
REPEATED_COLUMNS = [
    "split", "level", "object_label", "total_images", "object_images",
    "images_with_two_or_more", "instances", "p_repeated_given_present_pct",
    "repeated_image_prevalence_pct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-dir", type=Path,
        default=REPO_ROOT / "data/gqa_probing/subsets/probing_5k/annotations",
        help="Directory containing train/val/test_sceneGraphs.json.",
    )
    parser.add_argument(
        "--config-dir", type=Path,
        default=REPO_ROOT / "scripts/gqa_probing/config",
        help="Directory containing object_mapping_macro.json.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "outputs/gqa_bias_analysis/probing_5k/tables/object_cooccurrence",
        help="Output directory for CSV tables and JSON summary.",
    )
    parser.add_argument(
        "--min-pair-images", type=int, default=20,
        help="Minimum images containing both labels for ranked associations (default: 20).",
    )
    parser.add_argument(
        "--min-anchor-images", type=int, default=50,
        help="Minimum anchor-label image support for ranked associations (default: 50).",
    )
    args = parser.parse_args()
    if args.min_pair_images < 1 or args.min_anchor_images < 1:
        parser.error("--min-pair-images and --min-anchor-images must be positive.")
    return args


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def load_object_mapping(path: Path) -> dict:
    mapping = read_json(path)
    if set(mapping) != {"keep", "merge", "remove"}:
        raise ValueError(f"{path}: expected keep, merge and remove")
    keep, merge, remove = mapping["keep"], mapping["merge"], mapping["remove"]
    if not isinstance(keep, list) or not isinstance(merge, dict) or not isinstance(remove, list):
        raise ValueError(f"{path}: invalid mapping types")
    if (len(keep) != 14 or len(set(keep)) != 14
            or any(not isinstance(s, str) or not s for s in [*keep, *merge.keys(), *merge.values(), *remove])):
        raise ValueError(f"{path}: invalid canonical labels")
    if (set(keep) & set(merge) or set(keep) & set(remove)
            or set(merge) & set(remove) or not set(merge.values()) <= set(keep)):
        raise ValueError(f"{path}: conflicting mapping rules")
    return {"keep": set(keep), "merge": merge, "original_labels": set(keep) | set(merge)}


def analyze_split(path: Path, mapping: dict) -> dict:
    graphs = read_json(path)
    metrics = {
        level: {"images": Counter(), "pairs": Counter(), "instances": Counter(), "repeated": Counter()}
        for level in LEVELS
    }
    total_objects = 0
    for image_id, graph in graphs.items():
        objects = graph.get("objects") if isinstance(graph, dict) else None
        if not isinstance(objects, dict) or not objects:
            raise ValueError(f"{path}: missing or empty objects in image {image_id!r}")
        image_labels = {level: Counter() for level in LEVELS}
        for object_id, record in objects.items():
            where = f"{path}: {image_id}/{object_id}"
            if not isinstance(record, dict):
                raise ValueError(f"{where}: invalid object")
            macro = record.get("name")
            original = record.get("original_name")
            if not isinstance(original, str) or not isinstance(macro, str) or macro not in mapping["keep"]:
                raise ValueError(f"{where}: missing or invalid object labels")
            expected = original if original in mapping["keep"] else mapping["merge"].get(original)
            if expected != macro:
                raise ValueError(f"{where}: original_name does not map to canonical name")
            total_objects += 1
            image_labels[LEVELS[0]][macro] += 1
            image_labels[LEVELS[1]][original] += 1
        for level, counts in image_labels.items():
            metric = metrics[level]
            metric["instances"].update(counts)
            labels = sorted(counts)
            metric["images"].update(labels)
            metric["repeated"].update(label for label, count in counts.items() if count >= 2)
            metric["pairs"].update(combinations(labels, 2))
    return {"total_images": len(graphs), "total_objects": total_objects, "levels": metrics}


def pct(numerator: int, denominator: int) -> float | None:
    return round(100.0 * numerator / denominator, 6) if denominator else None


def get_lift(pair_images: int, a_images: int, b_images: int, total_images: int) -> float | None:
    if not a_images or not b_images or not total_images:
        return None
    value = pair_images * total_images / (a_images * b_images)
    return round(value, 6) if math.isfinite(value) else None


def make_tables(split: str, level: str, result: dict, labels: set[str]) -> tuple[list[dict], list[dict], list[dict]]:
    n = result["total_images"]
    metric = result["levels"][level]
    support = metric["images"]
    pairs = metric["pairs"]
    pair_rows: list[dict] = []
    directional_rows: list[dict] = []
    repeated_rows: list[dict] = []
    for a, b in combinations(sorted(labels), 2):
        a_n, b_n, ab_n = support[a], support[b], pairs[(a, b)]
        lift = get_lift(ab_n, a_n, b_n, n)
        status = ("absent_object" if not a_n or not b_n else
                  "absent_pair" if not ab_n else "present")
        pair_rows.append({
            "split": split, "level": level, "object_a": a, "object_b": b,
            "total_images": n, "object_a_images": a_n, "object_b_images": b_n,
            "pair_images": ab_n, "object_a_prevalence_pct": pct(a_n, n),
            "object_b_prevalence_pct": pct(b_n, n),
            "pair_prevalence_pct": pct(ab_n, n),
            "p_b_given_a_pct": pct(ab_n, a_n), "p_a_given_b_pct": pct(ab_n, b_n),
            "lift": lift, "status": status,
        })
        for anchor, context, anchor_n, context_n in ((a, b, a_n, b_n), (b, a, b_n, a_n)):
            directional_rows.append({
                "split": split, "level": level,
                "anchor_object": anchor, "context_object": context,
                "total_images": n, "anchor_images": anchor_n,
                "context_images": context_n, "pair_images": ab_n,
                "anchor_prevalence_pct": pct(anchor_n, n),
                "context_prevalence_pct": pct(context_n, n),
                "pair_prevalence_pct": pct(ab_n, n),
                "p_context_given_anchor_pct": pct(ab_n, anchor_n),
                "p_context_pct": pct(context_n, n),
                "lift": lift, "status": status,
            })
    for label in sorted(labels):
        label_n = support[label]
        repeated_n = metric["repeated"][label]
        repeated_rows.append({
            "split": split, "level": level, "object_label": label,
            "total_images": n, "object_images": label_n,
            "images_with_two_or_more": repeated_n,
            "instances": metric["instances"][label],
            "p_repeated_given_present_pct": pct(repeated_n, label_n),
            "repeated_image_prevalence_pct": pct(repeated_n, n),
        })
    return pair_rows, directional_rows, repeated_rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def main() -> None:
    args = parse_args()
    mapping = load_object_mapping(args.config_dir / "object_mapping_macro.json")
    summary = {
        "methodology": {
            "dataset": "GQA probing_5k prepared scene graphs",
            "unit": "distinct images",
            "cooccurrence": "two distinct object labels appearing in the same image; no relation edge required",
            "pair_order": "unordered, lexicographically sorted object labels",
            "conditional": "P(context|anchor) = images_with_both / images_with_anchor",
            "marginal": "P(context) = images_with_context / total_images",
            "lift": "P(context|anchor)/P(context), equivalent to P(A,B)/(P(A)P(B))",
            "repeated_labels": "same-class multiplicity (two or more instances in one image) analyzed separately",
            "zero_support": "undefined probabilities and lift emitted as empty CSV cells",
            "scope": "dataset co-occurrence, not scene-graph relations or model bias",
            "sampling": "label-enriched subset; estimates are not necessarily representative of source GQA",
        },
        "ranking_thresholds": {
            "min_pair_images": args.min_pair_images,
            "min_anchor_images": args.min_anchor_images,
        },
        "splits": {},
    }
    all_pairs: list[dict] = []
    all_directions: list[dict] = []
    all_repeated: list[dict] = []
    for split in SPLITS:
        source = args.annotations_dir / f"{split}_sceneGraphs.json"
        print(f"Analyzing {split}: {source}")
        result = analyze_split(source, mapping)
        split_summary = {"images": result["total_images"], "objects": result["total_objects"], "levels": {}}
        for level in LEVELS:
            metric = result["levels"][level]
            labels = mapping["keep"] if level == LEVELS[0] else mapping["original_labels"]
            pair_rows, direction_rows, repeated_rows = make_tables(split, level, result, labels)
            all_pairs.extend(pair_rows)
            all_directions.extend(direction_rows)
            all_repeated.extend(repeated_rows)
            split_summary["levels"][level] = {
                "observed_labels": sum(metric["images"][label] > 0 for label in labels),
                "possible_distinct_label_pairs": len(pair_rows),
                "observed_distinct_label_pairs": sum(row["pair_images"] > 0 for row in pair_rows),
                "images_with_repeated_label_total_by_label": sum(metric["repeated"].values()),
            }
        summary["splits"][split] = split_summary
        print(f"  images={result['total_images']:,}; objects={result['total_objects']:,}")
    out = args.output_dir
    write_csv(out / "object_cooccurrence_pairs.csv", all_pairs, PAIR_COLUMNS)
    write_csv(out / "object_cooccurrence_conditional.csv", all_directions, DIRECTIONAL_COLUMNS)
    write_csv(out / "object_repeated_class.csv", all_repeated, REPEATED_COLUMNS)
    ranked = [
        row for row in all_directions
        if row["status"] == "present" and row["pair_images"] >= args.min_pair_images
        and row["anchor_images"] >= args.min_anchor_images
    ]
    ranked.sort(key=lambda row: (
        row["split"], row["level"], -row["lift"], -row["pair_images"],
        row["anchor_object"], row["context_object"],
    ))
    write_csv(out / "ranked_object_cooccurrence.csv", ranked, DIRECTIONAL_COLUMNS)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "object_cooccurrence_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
