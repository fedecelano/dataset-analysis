#!/usr/bin/env python3
"""Analyze object–attribute associations in the frozen GQA probing subset.

Three association levels are exported:
  macro_object_macro_attribute: canonical object class x canonical attribute type
  macro_object_original_attribute: canonical object class x mapped original attribute
  original_object_original_attribute: original object label x mapped original attribute

The unit for P(attribute | object), P(attribute), and lift is a retained object
INSTANCE, not an image. Original attribute labels are deduplicated per object,
and only those in the KEEP/MERGE macro mapping are counted. Unmapped original
attributes are intentionally excluded from the analysis of probing targets.

Image support counts DISTINCT images with >=1 matching object–attribute pair.
No relation between different objects is inferred from image co-occurrence.
The source scene graphs, split IDs, and configuration are never modified.

Run from the repository root:
  python scripts/gqa_bias_analysis/analyze_object_attribute.py
  python scripts/gqa_bias_analysis/analyze_object_attribute.py --min-pair-objects 20
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test")
LEVELS = (
    "macro_object_macro_attribute",
    "macro_object_original_attribute",
    "original_object_original_attribute",
)
COLUMNS = [
    "split", "level", "object_label", "attribute_label",
    "total_images", "total_objects", "object_instances", "attribute_object_instances",
    "pair_object_instances", "pair_images", "pair_image_prevalence_pct",
    "p_attribute_given_object_pct", "p_attribute_pct", "lift", "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-dir", type=Path,
        default=REPO_ROOT / "data/gqa_probing/subsets/probing_5k/annotations",
        help="Directory with train/val/test_sceneGraphs.json.",
    )
    parser.add_argument(
        "--config-dir", type=Path,
        default=REPO_ROOT / "scripts/gqa_probing/config",
        help="Directory with the object and attribute macro mapping JSONs.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "outputs/gqa_bias_analysis/probing_5k/tables/object_attribute",
        help="Directory for CSV tables and JSON summary.",
    )
    parser.add_argument(
        "--min-pair-objects", type=int, default=20,
        help="Minimum paired object instances for the ranked association CSV (default: 20).",
    )
    parser.add_argument(
        "--min-object-instances", type=int, default=50,
        help="Minimum class instances for the ranked association CSV (default: 50).",
    )
    args = parser.parse_args()
    if args.min_pair_objects < 1 or args.min_object_instances < 1:
        parser.error("--min-pair-objects and --min-object-instances must be positive.")
    return args


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return result


def load_mapping(path: Path, expected_size: int) -> dict:
    mapping = read_json(path)
    if set(mapping) != {"keep", "merge", "remove"}:
        raise ValueError(f"{path}: expected keep/merge/remove keys")
    keep, merge, remove = mapping["keep"], mapping["merge"], mapping["remove"]
    if not isinstance(keep, list) or not isinstance(merge, dict) or not isinstance(remove, list):
        raise ValueError(f"{path}: incorrect mapping field types")
    if len(keep) != expected_size or len(set(keep)) != expected_size:
        raise ValueError(f"{path}: expected {expected_size} unique canonical labels")
    labels = [*keep, *merge.keys(), *merge.values(), *remove]
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError(f"{path}: invalid label")
    if (set(keep) & set(merge) or set(keep) & set(remove)
            or set(merge) & set(remove) or not set(merge.values()) <= set(keep)):
        raise ValueError(f"{path}: conflicting mapping rules")
    return {"keep": set(keep), "merge": merge, "remove": set(remove),
            "canonical_order": keep, "mapped_original": set(keep) | set(merge)}


def resolve_original(value: str, mapping: dict) -> str | None:
    if value in mapping["keep"]:
        return value
    return mapping["merge"].get(value)


def analyze_one_split(path: Path, object_map: dict, attr_map: dict) -> dict:
    graphs = read_json(path)
    total_objects = 0
    metrics = {
        level: {
            "object": Counter(), "attribute": Counter(), "pair": Counter(),
            "pair_images": Counter(), "object_labels": set(), "attribute_labels": set(),
        }
        for level in LEVELS
    }
    discarded_original_attribute_occurrences = 0
    for image_id, graph in graphs.items():
        if not isinstance(graph, dict) or not isinstance(graph.get("objects"), dict):
            raise ValueError(f"{path}: invalid objects for image {image_id!r}")
        if not graph["objects"]:
            raise ValueError(f"{path}: empty graph for image {image_id!r}")
        seen_in_image = {level: set() for level in LEVELS}
        for object_id, obj in graph["objects"].items():
            where = f"{path}: image={image_id}, object={object_id}"
            if not isinstance(obj, dict):
                raise ValueError(f"{where}: invalid object record")
            macro_object, original_object = obj.get("name"), obj.get("original_name")
            if (not isinstance(macro_object, str) or macro_object not in object_map["keep"]
                    or not isinstance(original_object, str)
                    or resolve_original(original_object, object_map) != macro_object):
                raise ValueError(f"{where}: inconsistent object label/provenance")
            macro_attributes = obj.get("attributes")
            original_attributes = obj.get("original_attributes")
            if not isinstance(macro_attributes, list) or not isinstance(original_attributes, list):
                raise ValueError(f"{where}: missing attribute lists")
            if (any(not isinstance(a, str) or a not in attr_map["keep"] for a in macro_attributes)
                    or len(macro_attributes) != len(set(macro_attributes))):
                raise ValueError(f"{where}: canonical attributes invalid or duplicated")
            if any(not isinstance(a, str) for a in original_attributes):
                raise ValueError(f"{where}: invalid original attribute")
            mapped_originals = {a for a in original_attributes if resolve_original(a, attr_map) is not None}
            discarded_original_attribute_occurrences += sum(
                resolve_original(a, attr_map) is None for a in original_attributes
            )
            expected_macro = {resolve_original(a, attr_map) for a in mapped_originals}
            if set(macro_attributes) != expected_macro:
                raise ValueError(f"{where}: canonical attributes disagree with provenance")
            total_objects += 1
            for level, obj_label, attributes in (
                (LEVELS[0], macro_object, set(macro_attributes)),
                (LEVELS[1], macro_object, mapped_originals),
                (LEVELS[2], original_object, mapped_originals),
            ):
                m = metrics[level]
                m["object"][obj_label] += 1
                m["object_labels"].add(obj_label)
                for attr in attributes:
                    m["attribute"][attr] += 1
                    m["attribute_labels"].add(attr)
                    m["pair"][(obj_label, attr)] += 1
                    seen_in_image[level].add((obj_label, attr))
        for level in LEVELS:
            metrics[level]["pair_images"].update(seen_in_image[level])
    # Include zero-occurrence canonical categories, and mapped original labels
    # absent from the split, to make missing associations explicit.
    metrics[LEVELS[0]]["object_labels"].update(object_map["keep"])
    metrics[LEVELS[0]]["attribute_labels"].update(attr_map["keep"])
    metrics[LEVELS[1]]["object_labels"].update(object_map["keep"])
    for level in LEVELS[1:]:
        metrics[level]["attribute_labels"].update(attr_map["mapped_original"])
    metrics[LEVELS[2]]["object_labels"].update(object_map["mapped_original"])
    return {"images": len(graphs), "objects": total_objects,
            "discarded_original_attribute_occurrences": discarded_original_attribute_occurrences,
            "metrics": metrics}


def make_rows(split: str, level: str, result: dict) -> list[dict]:
    m = result["metrics"][level]
    n_images = result["images"]
    n_objects = result["objects"]
    rows = []
    for obj in sorted(m["object_labels"]):
        n_obj = m["object"][obj]
        for attr in sorted(m["attribute_labels"]):
            n_attr = m["attribute"][attr]
            n_pair = m["pair"][(obj, attr)]
            p_attr_given_object = n_pair / n_obj if n_obj else None
            p_attr = n_attr / n_objects if n_objects else None
            lift = (p_attr_given_object / p_attr
                    if p_attr_given_object is not None and p_attr is not None and p_attr > 0
                    else None)
            rows.append({
                "split": split, "level": level, "object_label": obj,
                "attribute_label": attr, "total_images": n_images,
                "total_objects": n_objects, "object_instances": n_obj,
                "attribute_object_instances": n_attr, "pair_object_instances": n_pair,
                "pair_images": m["pair_images"][(obj, attr)],
                "pair_image_prevalence_pct": round(100 * m["pair_images"][(obj, attr)] / n_images, 6) if n_images else None,
                "p_attribute_given_object_pct": round(100 * p_attr_given_object, 6) if p_attr_given_object is not None else None,
                "p_attribute_pct": round(100 * p_attr, 6) if p_attr is not None else None,
                "lift": round(lift, 6) if lift is not None and math.isfinite(lift) else None,
                "status": ("absent_object" if not n_obj else "absent_attribute" if not n_attr
                           else "absent_pair" if not n_pair else "present"),
            })
    return rows


def write_csv(path: Path, rows: list[dict], columns: list[str] = COLUMNS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def main() -> None:
    args = parse_args()
    object_map = load_mapping(args.config_dir / "object_mapping_macro.json", 14)
    attr_map = load_mapping(args.config_dir / "attribute_mapping_macro.json", 5)
    results = {}
    all_rows = []
    summary = {"methodology": {
        "dataset": "GQA probing_5k prepared annotations (not original GQA)",
        "unit_for_probabilities": "retained object instances",
        "pair_images": "number of distinct images with >=1 object carrying the attribute",
        "original_attributes": "only KEEP/MERGE-mapped original labels, deduplicated per object",
        "removed_attributes": "excluded from association tables; tallied in summary",
        "lift": "P(attribute|object)/P(attribute), both based on retained object instances",
        "empty_denominators": "undefined probabilities/lift written as empty CSV cells",
        "scope": "dataset annotation associations, not model bias or verified visual ground truth",
        "sampling": "label-enriched subset; prevalence not guaranteed representative of source GQA",
    }, "splits": {}}
    for split in SPLITS:
        path = args.annotations_dir / f"{split}_sceneGraphs.json"
        print(f"Analyzing {split}: {path}")
        result = analyze_one_split(path, object_map, attr_map)
        results[split] = result
        summary["splits"][split] = {
            "images": result["images"], "objects": result["objects"],
            "discarded_original_attribute_occurrences": result["discarded_original_attribute_occurrences"],
            "levels": {},
        }
        for level in LEVELS:
            m = result["metrics"][level]
            summary["splits"][split]["levels"][level] = {
                "observed_object_labels": sum(v > 0 for v in m["object"].values()),
                "observed_attribute_labels": sum(v > 0 for v in m["attribute"].values()),
                "observed_pairs": len(m["pair"]),
                "paired_object_occurrences": sum(m["pair"].values()),
            }
            all_rows.extend(make_rows(split, level, result))
        print(f"  images={result['images']:,}; objects={result['objects']:,}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "object_attribute_associations.csv", all_rows)
    qualified = [row for row in all_rows if row["status"] == "present"
                 and row["pair_object_instances"] >= args.min_pair_objects
                 and row["object_instances"] >= args.min_object_instances]
    qualified.sort(key=lambda r: (r["split"], r["level"], -r["lift"],
                                   -r["pair_object_instances"], r["object_label"], r["attribute_label"]))
    write_csv(args.output_dir / "ranked_object_attribute_associations.csv", qualified)
    # One compact view focused on the five canonical attribute categories.
    macro_rows = [row for row in all_rows if row["level"] == LEVELS[0]]
    write_csv(args.output_dir / "macro_object_attribute_associations.csv", macro_rows)
    with (args.output_dir / "object_attribute_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Saved: {args.output_dir / 'object_attribute_summary.json'}")


if __name__ == "__main__":
    main()
