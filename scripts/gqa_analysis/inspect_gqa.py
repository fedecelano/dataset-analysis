
"""Generate summary statistics and vocabulary tables for GQA scene graphs.

Analyze the train and validation splits of the original or cleaned GQA
dataset without modifying the source annotations.

Counts represent annotation occurrences, not the number of distinct
images containing a label. Labels are preserved as recorded in the
scene graphs, without normalization.

Outputs are saved to:
    outputs/gqa_analysis/<dataset>/tables/

Usage:
    python scripts/gqa_analysis/inspect_gqa.py --dataset original
    python scripts/gqa_analysis/inspect_gqa.py --dataset cleaned
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIRS = {
    "original": PROJECT_ROOT / "data" / "gqa" / "annotations",
    "cleaned": PROJECT_ROOT / "data" / "gqa_cleaned" / "annotations",
}

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "gqa_analysis"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_DIRS),
        default="original",
        help="Dataset version to analyze (default: original).",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    """Load a GQA scene-graph JSON file."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a dictionary of scene graphs in {path}"
        )

    return data


def analyze_split(data: dict) -> dict:
    """Compute annotation totals and label frequencies for one split.

    Every annotated object, attribute, and relation occurrence is counted.
    Labels are used as-is, without normalization.
    """
    class_counts = Counter()
    attribute_counts = Counter()
    relation_counts = Counter()

    total_objects = 0
    total_attributes = 0
    total_relations = 0

    for image_data in data.values():
        for obj in image_data["objects"].values():
            total_objects += 1
            class_counts[obj["name"]] += 1

            for attribute in obj["attributes"]:
                attribute_counts[attribute] += 1
                total_attributes += 1

            for relation in obj["relations"]:
                relation_counts[relation["name"]] += 1
                total_relations += 1

    return {
        "num_images": len(data),
        "total_objects": total_objects,
        "total_attributes": total_attributes,
        "total_relations": total_relations,
        "class_counts": class_counts,
        "attribute_counts": attribute_counts,
        "relation_counts": relation_counts,
    }


def compare_vocabularies(
    train_counts: Counter,
    val_counts: Counter,
) -> dict:
    """Compute vocabulary sizes and overlap between splits."""
    train_vocab = set(train_counts)
    val_vocab = set(val_counts)

    return {
        "train_unique": len(train_vocab),
        "val_unique": len(val_vocab),
        "common": len(train_vocab & val_vocab),
        "train_only": len(train_vocab - val_vocab),
        "val_only": len(val_vocab - train_vocab),
    }


def vocabulary_types(
    train_stats: dict,
    val_stats: dict,
) -> dict:
    """Associate each label type with its train and validation counters."""
    return {
        "object_classes": (
            train_stats["class_counts"],
            val_stats["class_counts"],
        ),
        "attributes": (
            train_stats["attribute_counts"],
            val_stats["attribute_counts"],
        ),
        "relations": (
            train_stats["relation_counts"],
            val_stats["relation_counts"],
        ),
    }


def save_dataset_summary(
    train_stats: dict,
    val_stats: dict,
    output_dir: Path,
) -> None:
    """Save dataset sizes, annotation totals, and vocabulary sizes."""
    output_path = output_dir / "dataset_summary.csv"

    fieldnames = [
        "split",
        "images",
        "objects",
        "unique_classes",
        "attribute_instances",
        "unique_attributes",
        "relation_instances",
        "unique_relations",
    ]

    rows = []

    for split, stats in (
        ("train", train_stats),
        ("validation", val_stats),
    ):
        rows.append({
            "split": split,
            "images": stats["num_images"],
            "objects": stats["total_objects"],
            "unique_classes": len(stats["class_counts"]),
            "attribute_instances": stats["total_attributes"],
            "unique_attributes": len(stats["attribute_counts"]),
            "relation_instances": stats["total_relations"],
            "unique_relations": len(stats["relation_counts"]),
        })

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_vocabulary_overlap(
    train_stats: dict,
    val_stats: dict,
    output_dir: Path,
) -> None:
    """Save shared and split-exclusive vocabulary sizes."""
    output_path = output_dir / "vocabulary_overlap.csv"

    rows = []

    for label_type, (train_counts, val_counts) in vocabulary_types(
        train_stats, val_stats
    ).items():
        comparison = compare_vocabularies(train_counts, val_counts)
        rows.append({
            "type": label_type,
            **comparison,
        })

    fieldnames = [
        "type",
        "train_unique",
        "val_unique",
        "common",
        "train_only",
        "val_only",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_vocabulary_differences(
    train_stats: dict,
    val_stats: dict,
    output_dir: Path,
) -> None:
    """Save labels exclusive to either split and their frequencies."""
    output_path = output_dir / "vocabulary_differences.csv"

    rows = []

    for label_type, (train_counts, val_counts) in vocabulary_types(
        train_stats, val_stats
    ).items():
        train_vocab = set(train_counts)
        val_vocab = set(val_counts)

        train_only = sorted(
            train_vocab - val_vocab,
            key=lambda label: (-train_counts[label], label),
        )
        val_only = sorted(
            val_vocab - train_vocab,
            key=lambda label: (-val_counts[label], label),
        )

        for label in train_only:
            rows.append({
                "type": label_type,
                "label": label,
                "presence": "train_only",
                "train_count": train_counts[label],
                "val_count": 0,
            })

        for label in val_only:
            rows.append({
                "type": label_type,
                "label": label,
                "presence": "val_only",
                "train_count": 0,
                "val_count": val_counts[label],
            })

    fieldnames = [
        "type",
        "label",
        "presence",
        "train_count",
        "val_count",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_frequency_table(
    train_counts: Counter,
    val_counts: Counter,
    filename: str,
    output_dir: Path,
) -> None:
    """Save label frequencies in deterministic order.

    Labels are sorted by descending training count, descending
    validation count, and then alphabetically.
    """
    output_path = output_dir / filename

    all_labels = sorted(
        set(train_counts) | set(val_counts),
        key=lambda label: (
            -train_counts.get(label, 0),
            -val_counts.get(label, 0),
            label,
        ),
    )

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "train_count", "val_count"])

        for label in all_labels:
            writer.writerow([
                label,
                train_counts.get(label, 0),
                val_counts.get(label, 0),
            ])


def main() -> None:
    """Analyze train and validation splits and export six CSV tables."""
    args = parse_args()

    annotation_dir = DATASET_DIRS[args.dataset]
    output_dir = OUTPUT_ROOT / args.dataset / "tables"

    print(f"Loading GQA {args.dataset} scene graphs...")

    train_data = load_json(annotation_dir / "train_sceneGraphs.json")
    val_data = load_json(annotation_dir / "val_sceneGraphs.json")

    print("Analyzing train split...")
    train_stats = analyze_split(train_data)

    print("Analyzing validation split...")
    val_stats = analyze_split(val_data)

    output_dir.mkdir(parents=True, exist_ok=True)

    save_dataset_summary(train_stats, val_stats, output_dir)
    save_vocabulary_overlap(train_stats, val_stats, output_dir)
    save_vocabulary_differences(train_stats, val_stats, output_dir)

    save_frequency_table(
        train_stats["class_counts"],
        val_stats["class_counts"],
        "class_frequencies.csv",
        output_dir,
    )
    save_frequency_table(
        train_stats["attribute_counts"],
        val_stats["attribute_counts"],
        "attribute_frequencies.csv",
        output_dir,
    )
    save_frequency_table(
        train_stats["relation_counts"],
        val_stats["relation_counts"],
        "relation_frequencies.csv",
        output_dir,
    )

    print(f"Saved analysis tables to: {output_dir}")


if __name__ == "__main__":
    main()
