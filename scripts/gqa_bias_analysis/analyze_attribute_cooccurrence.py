#!/usr/bin/env python3
"""Analyze same-object attribute–attribute associations in frozen GQA probing_5k.

Two levels are analyzed:
  macro_attribute: two DIFFERENT canonical attribute types on one object
  original_attribute: two DIFFERENT KEEP/MERGE-mapped original labels on one object

The unit for probability calculations is a retained OBJECT INSTANCE, not an
image. A pair is counted once per object regardless of duplicated provenance.
An image contributes once to pair_images if it has >=1 object with the pair.
Pair ordering is ignored; conditional probabilities are reported both ways.
Original labels mapping to the same macro type are allowed to co-occur.
Only attributes mapped to one of the five probing types are included.
No co-occurrence across separate objects is counted. No graph is modified.

Run from the repository root:
    python scripts/gqa_bias_analysis/analyze_attribute_cooccurrence.py
    python scripts/gqa_bias_analysis/analyze_attribute_cooccurrence.py --min-pair-objects 30
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
LEVELS = ("macro_attribute", "original_attribute")
PAIR_COLUMNS = [
    "split", "level", "attribute_a", "attribute_b", "macro_type_a", "macro_type_b",
    "total_images", "total_objects", "attribute_a_objects", "attribute_b_objects",
    "pair_objects", "pair_images", "pair_image_prevalence_pct",
    "p_b_given_a_pct", "p_a_given_b_pct", "p_a_pct", "p_b_pct", "lift", "status",
]
RANK_COLUMNS = PAIR_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-dir", type=Path,
        default=REPO_ROOT / "data/gqa_probing/subsets/probing_5k/annotations",
        help="Folder containing train/val/test_sceneGraphs.json.",
    )
    parser.add_argument(
        "--config-dir", type=Path,
        default=REPO_ROOT / "scripts/gqa_probing/config",
        help="Folder containing attribute_mapping_macro.json.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "outputs/gqa_bias_analysis/probing_5k/tables/attribute_cooccurrence",
        help="Folder for CSV tables and JSON summary.",
    )
    parser.add_argument(
        "--min-pair-objects", type=int, default=20,
        help="Minimum same-object pair occurrences for rankings (default: 20).",
    )
    parser.add_argument(
        "--min-attribute-objects", type=int, default=50,
        help="Minimum object occurrences of each attribute for rankings (default: 50).",
    )
    args = parser.parse_args()
    if args.min_pair_objects < 1 or args.min_attribute_objects < 1:
        parser.error("Support thresholds must be positive integers.")
    return args


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON dictionary")
    return data


def load_attribute_mapping(path: Path) -> dict:
    m = read_json(path)
    if set(m) != {"keep", "merge", "remove"}:
        raise ValueError(f"{path}: expected keep/merge/remove")
    keep, merge, remove = m["keep"], m["merge"], m["remove"]
    if (not isinstance(keep, list) or len(keep) != 5
            or len(set(keep)) != 5 or not isinstance(merge, dict)
            or not isinstance(remove, list)):
        raise ValueError(f"{path}: invalid mapping structure")
    if any(not isinstance(v, str) or not v for v in [*keep, *merge, *merge.values(), *remove]):
        raise ValueError(f"{path}: invalid empty/non-string label")
    keep_set = set(keep)
    if (set(merge) & keep_set or set(remove) & keep_set or set(remove) & set(merge)
            or not set(merge.values()) <= keep_set or len(remove) != len(set(remove))):
        raise ValueError(f"{path}: inconsistent mapping actions")
    canonical = {label: label for label in keep}
    canonical.update(merge)
    return {"canonical": canonical, "macro_labels": keep, "original_labels": sorted(canonical)}


def analyze_split(path: Path, mapping: dict) -> dict:
    graphs = read_json(path)
    metrics = {level: {"attribute_objects": Counter(), "pair_objects": Counter(),
                       "pair_images": Counter()} for level in LEVELS}
    n_objects = 0
    discarded_original_occurrences = 0
    for image_id, graph in graphs.items():
        if not isinstance(graph, dict) or not isinstance(graph.get("objects"), dict):
            raise ValueError(f"{path}: invalid objects in image {image_id}")
        if not graph["objects"]:
            raise ValueError(f"{path}: empty graph in image {image_id}")
        image_pairs = {level: set() for level in LEVELS}
        for object_id, obj in graph["objects"].items():
            where = f"{path}: image {image_id}, object {object_id}"
            if not isinstance(obj, dict):
                raise ValueError(f"{where}: invalid object")
            macros, originals = obj.get("attributes"), obj.get("original_attributes")
            if not isinstance(macros, list) or not isinstance(originals, list):
                raise ValueError(f"{where}: missing attribute provenance")
            if (any(not isinstance(v, str) for v in macros + originals)
                    or len(macros) != len(set(macros))
                    or any(v not in mapping["macro_labels"] for v in macros)):
                raise ValueError(f"{where}: invalid or duplicate canonical attributes")
            mapped_original = {a for a in originals if a in mapping["canonical"]}
            expected_macros = {mapping["canonical"][a] for a in mapped_original}
            if set(macros) != expected_macros:
                raise ValueError(f"{where}: attributes inconsistent with original provenance")
            discarded_original_occurrences += sum(a not in mapping["canonical"] for a in originals)
            n_objects += 1
            for level, labels in (("macro_attribute", set(macros)),
                                  ("original_attribute", mapped_original)):
                m = metrics[level]
                for attribute in labels:
                    m["attribute_objects"][attribute] += 1
                for pair in combinations(sorted(labels), 2):
                    m["pair_objects"][pair] += 1
                    image_pairs[level].add(pair)
        for level in LEVELS:
            metrics[level]["pair_images"].update(image_pairs[level])
    return {"images": len(graphs), "objects": n_objects,
            "discarded_original_attribute_occurrences": discarded_original_occurrences,
            "metrics": metrics}


def rows_for_split(split: str, result: dict, mapping: dict, level: str) -> list[dict]:
    m = result["metrics"][level]
    labels = mapping["macro_labels"] if level == "macro_attribute" else mapping["original_labels"]
    n_objects, n_images = result["objects"], result["images"]
    rows = []
    for a, b in combinations(sorted(labels), 2):
        na, nb, nab = m["attribute_objects"][a], m["attribute_objects"][b], m["pair_objects"][(a, b)]
        p_a = na / n_objects if n_objects else None
        p_b = nb / n_objects if n_objects else None
        p_b_given_a = nab / na if na else None
        p_a_given_b = nab / nb if nb else None
        lift = nab * n_objects / (na * nb) if na and nb else None
        rows.append({
            "split": split, "level": level, "attribute_a": a, "attribute_b": b,
            "macro_type_a": mapping["canonical"][a], "macro_type_b": mapping["canonical"][b],
            "total_images": n_images, "total_objects": n_objects,
            "attribute_a_objects": na, "attribute_b_objects": nb,
            "pair_objects": nab, "pair_images": m["pair_images"][(a, b)],
            "pair_image_prevalence_pct": round(100 * m["pair_images"][(a, b)] / n_images, 6) if n_images else None,
            "p_b_given_a_pct": round(100 * p_b_given_a, 6) if p_b_given_a is not None else None,
            "p_a_given_b_pct": round(100 * p_a_given_b, 6) if p_a_given_b is not None else None,
            "p_a_pct": round(100 * p_a, 6) if p_a is not None else None,
            "p_b_pct": round(100 * p_b, 6) if p_b is not None else None,
            "lift": round(lift, 6) if lift is not None and math.isfinite(lift) else None,
            "status": "present" if nab else (
                "absent_attribute" if not na or not nb else "absent_pair"),
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PAIR_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def main() -> None:
    args = parse_args()
    mapping = load_attribute_mapping(args.config_dir / "attribute_mapping_macro.json")
    summary = {"methodology": {
        "dataset": "GQA probing_5k frozen prepared graphs",
        "sample_unit": "retained object instance",
        "pair_definition": "two distinct attributes annotated on the same object",
        "pair_order": "unordered",
        "pair_images": "images with at least one object containing both labels",
        "conditional": "P(B|A) and P(A|B) among retained object instances",
        "lift": "P(A,B)/(P(A)P(B)) among retained object instances",
        "original_labels": "only KEEP/MERGE original attributes, deduplicated per object",
        "same_macro_original_pairs": "original labels may map to the same canonical type",
        "zero_support": "undefined probabilities and lift exported as empty cells",
        "scope": "annotated associations; not model behavior or causal bias",
        "sampling": "label-enriched subset; not a random sample of original GQA",
    }, "ranking_thresholds": {"min_pair_objects": args.min_pair_objects,
                              "min_attribute_objects": args.min_attribute_objects},
        "splits": {}}
    rows = []
    for split in SPLITS:
        path = args.annotations_dir / f"{split}_sceneGraphs.json"
        print(f"Analyzing {split}: {path}")
        result = analyze_split(path, mapping)
        info = {"images": result["images"], "objects": result["objects"],
                "discarded_original_attribute_occurrences": result["discarded_original_attribute_occurrences"],
                "levels": {}}
        for level in LEVELS:
            m = result["metrics"][level]
            labels = mapping["macro_labels"] if level == "macro_attribute" else mapping["original_labels"]
            info["levels"][level] = {
                "configured_labels": len(labels),
                "observed_labels": sum(m["attribute_objects"][a] > 0 for a in labels),
                "possible_distinct_label_pairs": len(labels) * (len(labels)-1) // 2,
                "observed_distinct_label_pairs": len(m["pair_objects"]),
                "paired_object_occurrences": sum(m["pair_objects"].values()),
                "same_macro_type_original_pairs": (
                    sum(1 for a,b in m["pair_objects"] if mapping["canonical"][a] == mapping["canonical"][b])
                    if level == "original_attribute" else None),
            }
            rows.extend(rows_for_split(split, result, mapping, level))
        summary["splits"][split] = info
        print(f"  images={result['images']:,}; objects={result['objects']:,}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "attribute_cooccurrence_pairs.csv", rows)
    write_csv(args.output_dir / "macro_attribute_cooccurrence.csv",
              [r for r in rows if r["level"] == "macro_attribute"])
    ranked = [r for r in rows if r["status"] == "present"
              and r["pair_objects"] >= args.min_pair_objects
              and min(r["attribute_a_objects"], r["attribute_b_objects"]) >= args.min_attribute_objects]
    ranked.sort(key=lambda r: (r["split"], r["level"], -r["lift"],
                               -r["pair_objects"], r["attribute_a"], r["attribute_b"]))
    write_csv(args.output_dir / "ranked_attribute_cooccurrence.csv", ranked)
    summary_path = args.output_dir / "attribute_cooccurrence_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
