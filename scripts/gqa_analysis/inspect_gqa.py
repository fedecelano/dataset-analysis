#!/usr/bin/env python3
"""Generate summary statistics and vocabulary tables for GQA scene graphs.

Analyze the original, cleaned, or probing_5k scene graphs without modifying
annotations or images. Counts represent annotation occurrences, not the number
of distinct images containing a label.

For probing_5k, additional original-label tables use provenance fields:
- original_name on retained objects;
- original_attributes on retained objects (including attributes filtered out of
  the macro vocabulary, as retained by prepare_gqa_probing.py);
- original_names on retained canonical relations (falling back to original_name).
Original-relation frequencies describe provenance entries of *retained* macro
relations; they do not reconstruct discarded source relations or multiplicity
lost during canonical deduplication.

Outputs: outputs/gqa_analysis/<dataset>/tables/

Usage:
    python scripts/gqa_analysis/inspect_gqa.py --dataset original
    python scripts/gqa_analysis/inspect_gqa.py --dataset cleaned
    python scripts/gqa_analysis/inspect_gqa.py --dataset probing_5k
    python scripts/gqa_analysis/inspect_gqa.py --dataset probing_5k \
        --annotations-dir data/gqa_probing/subsets/probing_5k_test/annotations \
        --output-dir outputs/gqa_analysis/probing_5k_test/tables
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIRS = {
    "original": PROJECT_ROOT / "data" / "gqa" / "annotations",
    "cleaned": PROJECT_ROOT / "data" / "gqa_cleaned" / "annotations",
    "probing_5k": PROJECT_ROOT / "data" / "gqa_probing" / "subsets" / "probing_5k" / "annotations",
}
DATASET_SPLITS = {
    "original": ("train", "val"),
    "cleaned": ("train", "val"),
    "probing_5k": ("train", "val", "test"),
}
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "gqa_analysis"
KINDS = ("object_classes", "attributes", "relations")
COUNTER_KEYS = {
    "object_classes": "class_counts",
    "attributes": "attribute_counts",
    "relations": "relation_counts",
}
ORIGINAL_COUNTER_KEYS = {
    "object_classes": "original_class_counts",
    "attributes": "original_attribute_counts",
    "relations": "original_relation_counts",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=tuple(DATASET_DIRS), default="original",
        help="Dataset version (default: original).",
    )
    parser.add_argument(
        "--annotations-dir", type=Path, default=None,
        help="Override the default scene-graph annotation directory.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override the default directory for CSV tables.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dictionary of scene graphs in {path}")
    return data


def _valid_label(label: object, context: str) -> str:
    if not isinstance(label, str) or not label:
        raise ValueError(f"{context}: expected a nonempty string label, got {label!r}")
    return label


def _provenance_labels(obj: dict, image_id: str, object_id: str) -> tuple[str, list[str]]:
    """Return original object name and raw source attributes; require provenance."""
    context = f"{image_id}/{object_id}"
    name = _valid_label(obj.get("original_name"), f"{context} original_name")
    attrs = obj.get("original_attributes")
    if not isinstance(attrs, list):
        raise ValueError(f"{context}: missing or invalid original_attributes list")
    return name, [_valid_label(a, f"{context} original_attributes") for a in attrs]


def _relation_original_names(rel: dict, context: str) -> list[str]:
    """Use all saved labels after relation deduplication, if available."""
    if "original_names" in rel:
        names = rel["original_names"]
        if not isinstance(names, list) or not names:
            raise ValueError(f"{context}: original_names must be a nonempty list")
        result = [_valid_label(label, f"{context} original_names") for label in names]
        if "original_name" in rel and rel["original_name"] != result[0]:
            raise ValueError(f"{context}: original_name must match first original_names entry")
        return result
    return [_valid_label(rel.get("original_name"), f"{context} original_name")]


def analyze_split(data: dict, include_provenance: bool = False) -> dict:
    """Count annotations in one split, plus source-label provenance if requested."""
    counters = {
        "class_counts": Counter(),
        "attribute_counts": Counter(),
        "relation_counts": Counter(),
    }
    if include_provenance:
        counters.update({key: Counter() for key in ORIGINAL_COUNTER_KEYS.values()})

    for image_id, image_data in data.items():
        if not isinstance(image_data, dict) or not isinstance(image_data.get("objects"), dict):
            raise ValueError(f"{image_id}: expected a scene graph with an objects dictionary")
        objects = image_data["objects"]
        for object_id, obj in objects.items():
            context = f"{image_id}/{object_id}"
            if not isinstance(obj, dict):
                raise ValueError(f"{context}: object must be a dictionary")
            counters["class_counts"][_valid_label(obj.get("name"), f"{context} name")] += 1
            attrs = obj.get("attributes")
            rels = obj.get("relations")
            if not isinstance(attrs, list) or not isinstance(rels, list):
                raise ValueError(f"{context}: attributes and relations must be lists")
            for attr in attrs:
                counters["attribute_counts"][_valid_label(attr, f"{context} attribute")] += 1
            if include_provenance:
                original_name, original_attrs = _provenance_labels(obj, str(image_id), str(object_id))
                counters["original_class_counts"][original_name] += 1
                counters["original_attribute_counts"].update(original_attrs)
            for rel in rels:
                if not isinstance(rel, dict):
                    raise ValueError(f"{context}: relation must be a dictionary")
                counters["relation_counts"][_valid_label(rel.get("name"), f"{context} relation")] += 1
                if include_provenance:
                    counters["original_relation_counts"].update(
                        _relation_original_names(rel, f"{context} relation")
                    )

    result = {
        "num_images": len(data),
        "total_objects": sum(counters["class_counts"].values()),
        "total_attributes": sum(counters["attribute_counts"].values()),
        "total_relations": sum(counters["relation_counts"].values()),
        **counters,
    }
    return result


def compare_vocabularies(train_counts: Counter, val_counts: Counter) -> dict:
    """Legacy two-split vocabulary comparison; preserve prior CSV schema."""
    train_vocab = set(train_counts)
    val_vocab = set(val_counts)
    return {
        "train_unique": len(train_vocab),
        "val_unique": len(val_vocab),
        "common": len(train_vocab & val_vocab),
        "train_only": len(train_vocab - val_vocab),
        "val_only": len(val_vocab - train_vocab),
    }


def save_dataset_summary(results: dict[str, dict], output_dir: Path) -> None:
    fieldnames = [
        "split", "images", "objects", "unique_classes", "attribute_instances",
        "unique_attributes", "relation_instances", "unique_relations",
    ]
    rows = []
    for split, stats in results.items():
        rows.append({
            "split": "validation" if split == "val" else split,
            "images": stats["num_images"],
            "objects": stats["total_objects"],
            "unique_classes": len(stats["class_counts"]),
            "attribute_instances": stats["total_attributes"],
            "unique_attributes": len(stats["attribute_counts"]),
            "relation_instances": stats["total_relations"],
            "unique_relations": len(stats["relation_counts"]),
        })
    write_csv(output_dir / "dataset_summary.csv", fieldnames, rows)


def _vocabularies(results: dict[str, dict], kind: str, original: bool = False) -> dict[str, Counter]:
    key = (ORIGINAL_COUNTER_KEYS if original else COUNTER_KEYS)[kind]
    return {split: stats[key] for split, stats in results.items()}


def save_vocabulary_overlap(results: dict[str, dict], output_dir: Path) -> None:
    splits = tuple(results)
    if len(splits) == 2:
        fieldnames = ["type", "train_unique", "val_unique", "common", "train_only", "val_only"]
        rows = []
        for kind in KINDS:
            counts = _vocabularies(results, kind)
            rows.append({"type": kind, **compare_vocabularies(counts["train"], counts["val"])})
    else:
        fieldnames = ["type", *[f"{s}_unique" for s in splits], "common_all",
                      *[f"{s}_only" for s in splits],
                      "train_val_common", "train_test_common", "val_test_common"]
        rows = []
        for kind in KINDS:
            sets = {s: set(c) for s, c in _vocabularies(results, kind).items()}
            row = {"type": kind}
            row.update({f"{s}_unique": len(sets[s]) for s in splits})
            row["common_all"] = len(set.intersection(*sets.values()))
            row.update({f"{s}_only": len(sets[s] - set.union(*(sets[t] for t in splits if t != s)))
                        for s in splits})
            for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
                row[f"{a}_{b}_common"] = len(sets[a] & sets[b])
            rows.append(row)
    write_csv(output_dir / "vocabulary_overlap.csv", fieldnames, rows)


def save_vocabulary_differences(results: dict[str, dict], output_dir: Path) -> None:
    splits = tuple(results)
    if len(splits) == 2:
        fieldnames = ["type", "label", "presence", "train_count", "val_count"]
        rows = []
        for kind in KINDS:
            counts = _vocabularies(results, kind)
            train_counts, val_counts = counts["train"], counts["val"]
            train_only = sorted(set(train_counts) - set(val_counts), key=lambda x: (-train_counts[x], x))
            val_only = sorted(set(val_counts) - set(train_counts), key=lambda x: (-val_counts[x], x))
            for label in train_only:
                rows.append({"type": kind, "label": label, "presence": "train_only",
                             "train_count": train_counts[label], "val_count": 0})
            for label in val_only:
                rows.append({"type": kind, "label": label, "presence": "val_only",
                             "train_count": 0, "val_count": val_counts[label]})
    else:
        fieldnames = ["type", "label", "presence", *[f"{s}_count" for s in splits]]
        rows = []
        for kind in KINDS:
            counts = _vocabularies(results, kind)
            all_labels = set().union(*(set(c) for c in counts.values()))
            for label in sorted(all_labels, key=lambda x: (-counts["train"][x],
                                                        -counts["val"][x],
                                                        -counts["test"][x], x)):
                present = [s for s in splits if counts[s][label] > 0]
                if len(present) == len(splits):
                    continue
                rows.append({"type": kind, "label": label,
                             "presence": "_".join(present) + "_only",
                             **{f"{s}_count": counts[s][label] for s in splits}})
    write_csv(output_dir / "vocabulary_differences.csv", fieldnames, rows)


def save_frequency_table(counts_by_split: dict[str, Counter], filename: str, output_dir: Path) -> None:
    splits = tuple(counts_by_split)
    all_labels = sorted(
        set().union(*(set(c) for c in counts_by_split.values())),
        key=lambda label: (*(-counts_by_split[s][label] for s in splits), label),
    )
    rows = [{"label": label, **{f"{s}_count": counts_by_split[s][label] for s in splits}}
            for label in all_labels]
    write_csv(output_dir / filename, ["label", *[f"{s}_count" for s in splits]], rows)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    annotation_dir = args.annotations_dir if args.annotations_dir is not None else DATASET_DIRS[args.dataset]
    output_dir = args.output_dir if args.output_dir is not None else OUTPUT_ROOT / args.dataset / "tables"
    splits = DATASET_SPLITS[args.dataset]
    include_provenance = args.dataset == "probing_5k"

    print(f"Loading GQA {args.dataset} scene graphs from {annotation_dir}...")
    results = {}
    for split in splits:
        path = annotation_dir / f"{split}_sceneGraphs.json"
        print(f"Analyzing {split} split...")
        results[split] = analyze_split(load_json(path), include_provenance)

    save_dataset_summary(results, output_dir)
    save_vocabulary_overlap(results, output_dir)
    save_vocabulary_differences(results, output_dir)
    for kind, filename in (
        ("object_classes", "class_frequencies.csv"),
        ("attributes", "attribute_frequencies.csv"),
        ("relations", "relation_frequencies.csv"),
    ):
        save_frequency_table(_vocabularies(results, kind), filename, output_dir)

    if include_provenance:
        for kind, filename in (
            ("object_classes", "original_class_frequencies.csv"),
            ("attributes", "original_attribute_frequencies.csv"),
            ("relations", "original_relation_frequencies.csv"),
        ):
            save_frequency_table(_vocabularies(results, kind, original=True), filename, output_dir)

    print(f"Saved analysis tables to: {output_dir}")


if __name__ == "__main__":
    main()
