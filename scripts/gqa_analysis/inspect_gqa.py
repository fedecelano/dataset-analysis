import csv
import json
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

TRAIN_PATH = Path("data/gqa/annotations/train_sceneGraphs.json")
VAL_PATH = Path("data/gqa/annotations/val_sceneGraphs.json")

OUTPUT_DIR = Path("outputs/gqa_analysis/tables")


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_json(path):
    """Load a GQA scene graph JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Split analysis
# ---------------------------------------------------------------------

def analyze_split(data):
    """
    Compute basic statistics and label frequencies for one GQA split.

    Returns:
        - number of images
        - number of object instances
        - number of attribute instances
        - number of relation instances
        - frequency of each object class
        - frequency of each attribute
        - frequency of each relation
    """

    class_counts = Counter()
    attribute_counts = Counter()
    relation_counts = Counter()

    total_objects = 0
    total_attributes = 0
    total_relations = 0

    for image_data in data.values():

        for obj in image_data["objects"].values():

            # Object class
            total_objects += 1
            class_counts[obj["name"]] += 1

            # Object attributes
            for attribute in obj["attributes"]:
                attribute_counts[attribute] += 1
                total_attributes += 1

            # Object relations
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


# ---------------------------------------------------------------------
# Vocabulary comparison
# ---------------------------------------------------------------------

def compare_vocabularies(train_counts, val_counts):
    """Compare the vocabulary of a label type between train and validation."""

    train_vocab = set(train_counts)
    val_vocab = set(val_counts)

    return {
        "train_unique": len(train_vocab),
        "val_unique": len(val_vocab),
        "common": len(train_vocab & val_vocab),
        "train_only": len(train_vocab - val_vocab),
        "val_only": len(val_vocab - train_vocab),
    }


# ---------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------

def save_dataset_summary(train_stats, val_stats):
    """Save high-level statistics for train and validation."""

    output_path = OUTPUT_DIR / "dataset_summary.csv"

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

    rows = [
        {
            "split": "train",
            "images": train_stats["num_images"],
            "objects": train_stats["total_objects"],
            "unique_classes": len(train_stats["class_counts"]),
            "attribute_instances": train_stats["total_attributes"],
            "unique_attributes": len(train_stats["attribute_counts"]),
            "relation_instances": train_stats["total_relations"],
            "unique_relations": len(train_stats["relation_counts"]),
        },
        {
            "split": "validation",
            "images": val_stats["num_images"],
            "objects": val_stats["total_objects"],
            "unique_classes": len(val_stats["class_counts"]),
            "attribute_instances": val_stats["total_attributes"],
            "unique_attributes": len(val_stats["attribute_counts"]),
            "relation_instances": val_stats["total_relations"],
            "unique_relations": len(val_stats["relation_counts"]),
        },
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_vocabulary_overlap(train_stats, val_stats):
    """Save train/validation vocabulary overlap statistics."""

    output_path = OUTPUT_DIR / "vocabulary_overlap.csv"

    vocabulary_types = {
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

    rows = []

    for label_type, (train_counts, val_counts) in vocabulary_types.items():

        comparison = compare_vocabularies(
            train_counts,
            val_counts,
        )

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

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def save_vocabulary_differences(train_stats, val_stats):
    """Save labels that appear only in train or only in validation."""

    output_path = OUTPUT_DIR / "vocabulary_differences.csv"

    vocabulary_types = {
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

    rows = []

    for label_type, (train_counts, val_counts) in vocabulary_types.items():
        train_vocab = set(train_counts)
        val_vocab = set(val_counts)

        train_only = sorted(
            train_vocab - val_vocab,
            key=lambda label: (
                -train_counts[label],
                label,
            ),
        )

        val_only = sorted(
            val_vocab - train_vocab,
            key=lambda label: (
                -val_counts[label],
                label,
            ),
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

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)        


def save_frequency_table(train_counts, val_counts, filename):
    """
    Save train and validation frequency for every label.

    Labels are sorted by decreasing training frequency.
    Validation frequency and alphabetical order are used
    as tie-breakers.
    """

    output_path = OUTPUT_DIR / filename

    all_labels = sorted(
        set(train_counts) | set(val_counts),
        key=lambda label: (
            -train_counts.get(label, 0),
            -val_counts.get(label, 0),
            label,
        ),
    )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "label",
            "train_count",
            "val_count",
        ])

        for label in all_labels:
            writer.writerow([
                label,
                train_counts.get(label, 0),
                val_counts.get(label, 0),
            ])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading GQA scene graphs...")

    train_data = load_json(TRAIN_PATH)
    val_data = load_json(VAL_PATH)

    print("Analyzing train split...")
    train_stats = analyze_split(train_data)

    print("Analyzing validation split...")
    val_stats = analyze_split(val_data)

    save_dataset_summary(
        train_stats,
        val_stats,
    )

    save_vocabulary_overlap(
        train_stats,
        val_stats,
    )

    save_vocabulary_differences(
        train_stats,
        val_stats,
    )

    save_frequency_table(
        train_stats["class_counts"],
        val_stats["class_counts"],
        "class_frequencies.csv",
    )

    save_frequency_table(
        train_stats["attribute_counts"],
        val_stats["attribute_counts"],
        "attribute_frequencies.csv",
    )

    save_frequency_table(
        train_stats["relation_counts"],
        val_stats["relation_counts"],
        "relation_frequencies.csv",
    )

    print(f"Saved analysis tables to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()