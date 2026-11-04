#!/usr/bin/env python3
"""Analyze 14/5/7 macro-label frequencies in prepared GQA probing graphs.

Read-only with respect to the dataset. Validate the canonical labels,
per-object attribute uniqueness, per-instance directed relation uniqueness,
and original-label provenance before exporting four frequency CSV tables.

Run from the repository root:
    python scripts/gqa_probing/analyze_gqa_probing.py
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


def load_rules(config_dir: Path) -> dict:
    """Validate the macro vocabularies and mapping rules."""
    all_rules = {}
    for kind, expected_count in EXPECTED_CLASSES.items():
        path = config_dir / f"{kind}_mapping_macro.json"
        with path.open("r", encoding="utf-8") as handle:
            mapping = json.load(handle)
        if not isinstance(mapping, dict) or set(mapping) != {"keep", "merge", "remove"}:
            raise ValueError(f"{path}: expected keep, merge and remove keys")
        keep, merge, remove = (mapping[key] for key in ("keep", "merge", "remove"))
        if not isinstance(keep, list) or not isinstance(merge, dict) or not isinstance(remove, list):
            raise ValueError(f"{path}: invalid mapping types")
        labels = [*keep, *merge.keys(), *merge.values(), *remove]
        if not all(isinstance(label, str) and bool(label) for label in labels):
            raise ValueError(f"{path}: labels must be nonempty strings")
        if len(keep) != expected_count or len(set(keep)) != expected_count:
            raise ValueError(f"{path}: expected {expected_count} distinct canonical labels")
        if len(remove) != len(set(remove)):
            raise ValueError(f"{path}: duplicate REMOVE labels")
        keep_set, merge_set, remove_set = set(keep), set(merge), set(remove)
        if keep_set & merge_set or keep_set & remove_set or merge_set & remove_set:
            raise ValueError(f"{path}: conflicting mapping actions")
        if not set(merge.values()) <= keep_set:
            raise ValueError(f"{path}: MERGE target outside canonical vocabulary")
        all_rules[kind] = {"labels": keep, "keep": keep_set, "merge": merge}
    return all_rules


def resolves_to(original: str, canonical: str, rules: dict) -> bool:
    """Whether an original label maps to the indicated canonical label."""
    return original == canonical and original in rules["keep"] or rules["merge"].get(original) == canonical


def analyze_split(path: Path, rules: dict) -> dict:
    """Analyze and validate one prepared split, retaining directed counts."""
    with path.open("r", encoding="utf-8") as handle:
        graphs = json.load(handle)
    if not isinstance(graphs, dict):
        raise ValueError(f"{path}: expected a dictionary of scene graphs")

    object_counts = Counter()
    attribute_counts = Counter()
    attribute_object_counts = Counter()
    relation_counts = Counter()
    triple_counts = Counter()

    for image_id, graph in graphs.items():
        if not isinstance(graph, dict) or not isinstance(graph.get("objects"), dict):
            raise ValueError(f"{path}: image {image_id!r} has invalid objects")
        objects = graph["objects"]
        if not objects:
            raise ValueError(f"{path}: empty retained graph {image_id!r}")
        # Validate all object labels first, including relation targets.
        for oid, obj in objects.items():
            if not isinstance(obj, dict) or obj.get("name") not in rules["object"]["keep"]:
                raise ValueError(f"{path}: invalid object {image_id}/{oid}")

        for source_id, obj in objects.items():
            where = f"{path}: {image_id}/{source_id}"
            label = obj["name"]
            original_name = obj.get("original_name")
            if not isinstance(original_name, str) or not resolves_to(original_name, label, rules["object"]):
                raise ValueError(f"{where}: missing or inconsistent original object label")
            object_counts[label] += 1

            attrs = obj.get("attributes")
            originals = obj.get("original_attributes")
            if not isinstance(attrs, list) or not isinstance(originals, list):
                raise ValueError(f"{where}: invalid attributes or original_attributes")
            if any(not isinstance(attr, str) or attr not in rules["attribute"]["keep"] for attr in attrs):
                raise ValueError(f"{where}: unknown canonical attribute")
            if len(attrs) != len(set(attrs)):
                raise ValueError(f"{where}: duplicate canonical attributes")
            if any(not isinstance(attr, str) for attr in originals):
                raise ValueError(f"{where}: invalid original attribute")
            expected_attrs = {
                original if original in rules["attribute"]["keep"]
                else rules["attribute"]["merge"].get(original)
                for original in originals
            } - {None}
            if set(attrs) != expected_attrs:
                raise ValueError(f"{where}: canonical attributes do not match original_attributes")
            for attr in attrs:
                attribute_counts[attr] += 1
                attribute_object_counts[attr] += 1

            rels = obj.get("relations")
            if not isinstance(rels, list):
                raise ValueError(f"{where}: invalid relations")
            seen_edges = set()
            for rel in rels:
                if not isinstance(rel, dict):
                    raise ValueError(f"{where}: invalid relation record")
                relation = rel.get("name")
                if not isinstance(relation, str) or relation not in rules["relation"]["keep"]:
                    raise ValueError(f"{where}: unknown canonical relation {relation!r}")
                target_id = str(rel.get("object"))
                if target_id not in objects:
                    raise ValueError(f"{where}: missing relation target {target_id}")
                edge = (relation, target_id)
                if edge in seen_edges:
                    raise ValueError(f"{where}: duplicate canonical relation to {target_id}")
                seen_edges.add(edge)
                original_names = rel.get("original_names")
                if (not isinstance(original_names, list) or not original_names
                        or any(not isinstance(name, str) or not resolves_to(name, relation, rules["relation"])
                               for name in original_names)
                        or rel.get("original_name") != original_names[0]):
                    raise ValueError(f"{where}: inconsistent relation provenance")
                relation_counts[relation] += 1
                triple_counts[(label, relation, objects[target_id]["name"])] += 1

    if sum(relation_counts.values()) != sum(triple_counts.values()):
        raise ValueError(f"{path}: relation/triple count mismatch")
    if attribute_counts != attribute_object_counts:
        raise ValueError(f"{path}: attribute occurrence/object count mismatch")

    return {
        "images": len(graphs),
        "objects": object_counts,
        "attributes": attribute_counts,
        "attribute_objects": attribute_object_counts,
        "relations": relation_counts,
        "triples": triple_counts,
    }


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    """Write deterministic CSV output, including headers for empty tables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def make_frequency_rows(labels: list[str], results: dict, key: str) -> list[dict]:
    rows = [
        {"label": label, "train_count": results["train"][key][label],
         "val_count": results["val"][key][label]}
        for label in labels
    ]
    return sorted(rows, key=lambda row: (-row["train_count"], -row["val_count"], row["label"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "data/gqa_probing/annotations")
    parser.add_argument("--config-dir", type=Path, default=Path(__file__).resolve().parent / "config")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/gqa_probing/tables")
    args = parser.parse_args()

    rules = load_rules(args.config_dir)
    results = {
        split: analyze_split(args.input_dir / f"{split}_sceneGraphs.json", rules)
        for split in SPLITS
    }

    for kind in ("object", "relation"):
        rows = make_frequency_rows(rules[kind]["labels"], results, kind + "s")
        write_csv(args.output_dir / f"{kind}_macro_frequencies.csv",
                  ["label", "train_count", "val_count"], rows)

    attributes = make_frequency_rows(rules["attribute"]["labels"], results, "attributes")
    totals = {split: sum(results[split]["objects"].values()) for split in SPLITS}
    for row in attributes:
        label = row["label"]
        for split in SPLITS:
            row[f"{split}_objects_with_attribute"] = results[split]["attribute_objects"][label]
            row[f"{split}_object_coverage_pct"] = (
                round(100 * results[split]["attribute_objects"][label] / totals[split], 4)
                if totals[split] else 0.0
            )
    write_csv(args.output_dir / "attribute_macro_frequencies.csv",
              ["label", "train_count", "val_count",
               "train_objects_with_attribute", "val_objects_with_attribute",
               "train_object_coverage_pct", "val_object_coverage_pct"], attributes)

    all_triples = set(results["train"]["triples"]) | set(results["val"]["triples"])
    triple_rows = [
        {"source_class": source, "relation": relation, "target_class": target,
         "train_count": results["train"]["triples"][(source, relation, target)],
         "val_count": results["val"]["triples"][(source, relation, target)]}
        for source, relation, target in all_triples
    ]
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
