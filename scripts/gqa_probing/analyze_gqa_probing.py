#!/usr/bin/env python3
"""Analyze macro-category frequencies in prepared GQA probing scene graphs.

Reads train/val scene graphs produced by prepare_gqa_probing.py. Counts object
instances, attribute annotations and distinct objects carrying each attribute,
and directed relation annotations and (source, relation, target) class triples.
This script reads only; it never alters annotations or image files.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CLASSES = {"object": 14, "attribute": 5, "relation": 7}
SPLITS = ("train", "val")


def read_vocabulary(path: Path, expected_count: int) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    if not isinstance(mapping, dict) or not isinstance(mapping.get("keep"), list):
        raise ValueError(f"Invalid mapping file: {path}")
    labels = mapping["keep"]
    if (len(labels) != expected_count or len(set(labels)) != len(labels)
            or any(not isinstance(label, str) or not label for label in labels)):
        raise ValueError(f"Expected {expected_count} unique labels in {path}")
    return labels


def analyze_split(path: Path, vocabulary: dict[str, set[str]]) -> dict:
    with path.open(encoding="utf-8") as handle:
        graphs = json.load(handle)
    if not isinstance(graphs, dict):
        raise ValueError(f"{path}: expected a dictionary of image scene graphs")

    object_counts = Counter()
    attribute_counts = Counter()
    attribute_object_counts = Counter()
    relation_counts = Counter()
    triple_counts = Counter()

    for image_id, graph in graphs.items():
        if not isinstance(graph, dict) or not isinstance(graph.get("objects"), dict):
            raise ValueError(f"{path}: image {image_id} has invalid objects")
        objects = graph["objects"]
        for source_id, obj in objects.items():
            if not isinstance(obj, dict):
                raise ValueError(f"{path}: image {image_id}, object {source_id} is invalid")
            label = obj.get("name")
            if label not in vocabulary["object"]:
                raise ValueError(f"{path}: unrecognized object label {label!r} in image {image_id}")
            object_counts[label] += 1

            attrs = obj.get("attributes", [])
            if not isinstance(attrs, list):
                raise ValueError(f"{path}: invalid attributes at {image_id}/{source_id}")
            for attr in attrs:
                if attr not in vocabulary["attribute"]:
                    raise ValueError(f"{path}: unknown attribute {attr!r} at {image_id}/{source_id}")
                attribute_counts[attr] += 1
            for attr in set(attrs):
                attribute_object_counts[attr] += 1

            rels = obj.get("relations", [])
            if not isinstance(rels, list):
                raise ValueError(f"{path}: invalid relations at {image_id}/{source_id}")
            for rel in rels:
                if not isinstance(rel, dict):
                    raise ValueError(f"{path}: invalid relation at {image_id}/{source_id}")
                relation = rel.get("name")
                if relation not in vocabulary["relation"]:
                    raise ValueError(f"{path}: unknown relation {relation!r} at {image_id}/{source_id}")
                target_id = str(rel.get("object"))
                if target_id not in objects:
                    raise ValueError(f"{path}: missing relation target {target_id} at {image_id}/{source_id}")
                target = objects[target_id]
                if not isinstance(target, dict) or target.get("name") not in vocabulary["object"]:
                    raise ValueError(f"{path}: invalid relation target at {image_id}/{source_id}")
                relation_counts[relation] += 1
                triple_counts[(label, relation, target["name"])] += 1

    return {
        "images": len(graphs),
        "objects": object_counts,
        "attributes": attribute_counts,
        "attribute_objects": attribute_object_counts,
        "relations": relation_counts,
        "triples": triple_counts,
    }


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def make_frequency_rows(labels: list[str], results: dict, key: str) -> list[dict]:
    rows = [{"label": label,
             "train_count": results["train"][key][label],
             "val_count": results["val"][key][label]}
            for label in labels]
    return sorted(rows, key=lambda row: (-row["train_count"], -row["val_count"], row["label"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "data/gqa_probing/annotations",
                        help="Directory with train_sceneGraphs.json and val_sceneGraphs.json")
    parser.add_argument("--config-dir", type=Path, default=Path(__file__).resolve().parent / "config",
                        help="Directory with the three macro-category mapping files")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/gqa_probing/tables",
                        help="Directory for the generated CSV files")
    args = parser.parse_args()

    labels = {kind: read_vocabulary(args.config_dir / f"{kind}_mapping_macro.json", count)
              for kind, count in EXPECTED_CLASSES.items()}
    vocabulary = {kind: set(items) for kind, items in labels.items()}
    results = {split: analyze_split(args.input_dir / f"{split}_sceneGraphs.json", vocabulary)
               for split in SPLITS}

    for kind in ("object", "relation"):
        rows = make_frequency_rows(labels[kind], results, kind + "s")
        write_csv(args.output_dir / f"{kind}_macro_frequencies.csv",
                  ["label", "train_count", "val_count"], rows)

    attributes = make_frequency_rows(labels["attribute"], results, "attributes")
    for row in attributes:
        label = row["label"]
        row["train_objects_with_attribute"] = results["train"]["attribute_objects"][label]
        row["val_objects_with_attribute"] = results["val"]["attribute_objects"][label]
        # Percentage of ALL retained object instances; absence of an annotation
        # does not establish that the visual property is absent.
        for split in SPLITS:
            total = sum(results[split]["objects"].values())
            row[f"{split}_object_coverage_pct"] = round(
                100 * results[split]["attribute_objects"][label] / total, 4
            ) if total else 0.0
    write_csv(args.output_dir / "attribute_macro_frequencies.csv",
              ["label", "train_count", "val_count",
               "train_objects_with_attribute", "val_objects_with_attribute",
               "train_object_coverage_pct", "val_object_coverage_pct"], attributes)

    triples = set(results["train"]["triples"]) | set(results["val"]["triples"])
    triple_rows = [{"source_class": source, "relation": relation,
                    "target_class": target,
                    "train_count": results["train"]["triples"][(source, relation, target)],
                    "val_count": results["val"]["triples"][(source, relation, target)]}
                   for source, relation, target in triples]
    triple_rows.sort(key=lambda row: (-row["train_count"], -row["val_count"],
                                      row["source_class"], row["relation"], row["target_class"]))
    write_csv(args.output_dir / "relation_macro_class_triples.csv",
              ["source_class", "relation", "target_class", "train_count", "val_count"], triple_rows)

    for split in SPLITS:
        result = results[split]
        print(f"{split}: images={result['images']:,}; objects={sum(result['objects'].values()):,}; "
              f"attributes={sum(result['attributes'].values()):,}; "
              f"relations={sum(result['relations'].values()):,}")


if __name__ == "__main__":
    main()
