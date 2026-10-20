import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

TRAIN_PATH = Path("data/gqa/annotations/train_sceneGraphs.json")
VAL_PATH = Path("data/gqa/annotations/val_sceneGraphs.json")
SELECTION_PATH = Path("outputs/gqa_analysis/tables/object_probe_selection_134.csv")
OUTPUT_DIR = Path("outputs/gqa_analysis/tables")

EXPECTED_FINAL_CLASSES = 100


# ---------------------------------------------------------------------
# Selection loading
# ---------------------------------------------------------------------

def load_object_selection(path: Path):
    """
    Load the object-selection table produced by select_gqa_probe_objects.py.

    Returns
    -------
    source_to_final : dict[str, str]
        Mapping from original GQA object labels to the final probing labels.
        REMOVE rows are excluded.
    final_labels : set[str]
        Final 100-class probing vocabulary.
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in {path}")

    required_columns = {"label", "action", "final_label"}
    missing = required_columns - set(rows[0].keys())
    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {sorted(missing)}"
        )

    source_to_final = {}

    for row in rows:
        source_label = row["label"].strip()
        action = row["action"].strip().upper()
        final_label = row["final_label"].strip()

        if action == "REMOVE":
            continue

        if action not in {"KEEP", "MERGE"}:
            raise ValueError(
                f"Unexpected action '{row['action']}' for label '{source_label}'."
            )

        if not final_label:
            raise ValueError(
                f"Missing final_label for non-removed label '{source_label}'."
            )

        source_to_final[source_label] = final_label

    final_labels = set(source_to_final.values())

    if len(final_labels) != EXPECTED_FINAL_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_FINAL_CLASSES} final classes, "
            f"but found {len(final_labels)} in {path}."
        )

    return source_to_final, final_labels


# ---------------------------------------------------------------------
# GQA analysis
# ---------------------------------------------------------------------

def analyze_split(scene_graph_path: Path, source_to_final: dict[str, str]):
    """
    Analyze attributes attached to the selected object labels in one GQA split.

    Only original labels explicitly present in object_probe_selection_134.csv
    are considered. This intentionally avoids adding lower-frequency object
    variants that have not yet been reviewed.
    """
    with scene_graph_path.open("r", encoding="utf-8") as f:
        scene_graphs = json.load(f)

    object_instances = Counter()
    objects_with_attributes = Counter()
    attribute_annotations_per_object_class = Counter()

    global_attribute_counts = Counter()
    object_attribute_counts = Counter()

    # Number of distinct object instances carrying a given attribute.
    # Using a set per object protects the coverage statistic from accidental
    # duplicate strings inside one object's attribute list.
    object_attribute_object_counts = Counter()

    unique_attributes_per_object = defaultdict(set)
    object_classes_per_attribute = defaultdict(set)

    selected_instances_total = 0
    selected_with_attributes_total = 0
    attribute_annotations_total = 0

    for image_data in scene_graphs.values():
        for obj in image_data.get("objects", {}).values():
            source_label = obj.get("name", "")

            if source_label not in source_to_final:
                continue

            final_label = source_to_final[source_label]
            attributes = obj.get("attributes", []) or []

            object_instances[final_label] += 1
            selected_instances_total += 1

            if attributes:
                objects_with_attributes[final_label] += 1
                selected_with_attributes_total += 1

            attribute_annotations_per_object_class[final_label] += len(attributes)
            attribute_annotations_total += len(attributes)

            for attribute in attributes:
                global_attribute_counts[attribute] += 1
                object_attribute_counts[(final_label, attribute)] += 1
                unique_attributes_per_object[final_label].add(attribute)
                object_classes_per_attribute[attribute].add(final_label)

            for attribute in set(attributes):
                object_attribute_object_counts[(final_label, attribute)] += 1

    return {
        "object_instances": object_instances,
        "objects_with_attributes": objects_with_attributes,
        "attribute_annotations_per_object_class": attribute_annotations_per_object_class,
        "global_attribute_counts": global_attribute_counts,
        "object_attribute_counts": object_attribute_counts,
        "object_attribute_object_counts": object_attribute_object_counts,
        "unique_attributes_per_object": unique_attributes_per_object,
        "object_classes_per_attribute": object_classes_per_attribute,
        "selected_instances_total": selected_instances_total,
        "selected_with_attributes_total": selected_with_attributes_total,
        "attribute_annotations_total": attribute_annotations_total,
    }


# ---------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------

def pct(numerator: int, denominator: int):
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 4)


def build_dataset_summary(train, val, final_labels):
    train_unique_attrs = len(train["global_attribute_counts"])
    val_unique_attrs = len(val["global_attribute_counts"])

    return [
        {
            "metric": "final_object_classes",
            "train": len(final_labels),
            "val": len(final_labels),
        },
        {
            "metric": "selected_object_instances",
            "train": train["selected_instances_total"],
            "val": val["selected_instances_total"],
        },
        {
            "metric": "objects_with_attributes",
            "train": train["selected_with_attributes_total"],
            "val": val["selected_with_attributes_total"],
        },
        {
            "metric": "objects_with_attributes_pct",
            "train": pct(
                train["selected_with_attributes_total"],
                train["selected_instances_total"],
            ),
            "val": pct(
                val["selected_with_attributes_total"],
                val["selected_instances_total"],
            ),
        },
        {
            "metric": "attribute_annotations",
            "train": train["attribute_annotations_total"],
            "val": val["attribute_annotations_total"],
        },
        {
            "metric": "unique_attributes",
            "train": train_unique_attrs,
            "val": val_unique_attrs,
        },
    ]


def build_object_summary(train, val, final_labels):
    rows = []

    for label in final_labels:
        train_objects = train["object_instances"][label]
        val_objects = val["object_instances"][label]

        train_with_attrs = train["objects_with_attributes"][label]
        val_with_attrs = val["objects_with_attributes"][label]

        rows.append(
            {
                "object_label": label,
                "train_objects": train_objects,
                "val_objects": val_objects,
                "train_objects_with_attributes": train_with_attrs,
                "val_objects_with_attributes": val_with_attrs,
                "train_attribute_coverage_pct": pct(train_with_attrs, train_objects),
                "val_attribute_coverage_pct": pct(val_with_attrs, val_objects),
                "train_attribute_annotations": train[
                    "attribute_annotations_per_object_class"
                ][label],
                "val_attribute_annotations": val[
                    "attribute_annotations_per_object_class"
                ][label],
                "train_unique_attributes": len(
                    train["unique_attributes_per_object"].get(label, set())
                ),
                "val_unique_attributes": len(
                    val["unique_attributes_per_object"].get(label, set())
                ),
            }
        )

    rows.sort(key=lambda row: (-row["train_objects"], row["object_label"]))
    return rows


def build_attribute_frequencies(train, val):
    attributes = set(train["global_attribute_counts"]) | set(
        val["global_attribute_counts"]
    )

    rows = []
    for attribute in attributes:
        rows.append(
            {
                "attribute": attribute,
                "train_count": train["global_attribute_counts"][attribute],
                "val_count": val["global_attribute_counts"][attribute],
                "train_object_classes": len(
                    train["object_classes_per_attribute"].get(attribute, set())
                ),
                "val_object_classes": len(
                    val["object_classes_per_attribute"].get(attribute, set())
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            -row["train_count"],
            -row["val_count"],
            row["attribute"],
        )
    )
    return rows


def build_object_attribute_frequencies(train, val, final_labels):
    pairs = set(train["object_attribute_counts"]) | set(
        val["object_attribute_counts"]
    )

    rows = []

    for object_label, attribute in pairs:
        train_count = train["object_attribute_counts"][(object_label, attribute)]
        val_count = val["object_attribute_counts"][(object_label, attribute)]

        train_object_count = train["object_instances"][object_label]
        val_object_count = val["object_instances"][object_label]

        train_objects_with_attribute = train["object_attribute_object_counts"][
            (object_label, attribute)
        ]
        val_objects_with_attribute = val["object_attribute_object_counts"][
            (object_label, attribute)
        ]

        rows.append(
            {
                "object_label": object_label,
                "attribute": attribute,
                "train_count": train_count,
                "val_count": val_count,
                "train_objects_with_attribute": train_objects_with_attribute,
                "val_objects_with_attribute": val_objects_with_attribute,
                "train_object_coverage_pct": pct(
                    train_objects_with_attribute, train_object_count
                ),
                "val_object_coverage_pct": pct(
                    val_objects_with_attribute, val_object_count
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            row["object_label"],
            -row["train_count"],
            -row["val_count"],
            row["attribute"],
        )
    )
    return rows


def save_csv(path: Path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_to_final, final_labels = load_object_selection(SELECTION_PATH)

    train = analyze_split(TRAIN_PATH, source_to_final)
    val = analyze_split(VAL_PATH, source_to_final)

    dataset_summary = build_dataset_summary(train, val, final_labels)
    object_summary = build_object_summary(train, val, final_labels)
    attribute_frequencies = build_attribute_frequencies(train, val)
    object_attribute_frequencies = build_object_attribute_frequencies(
        train, val, final_labels
    )

    summary_path = OUTPUT_DIR / "probe_attribute_dataset_summary.csv"
    object_summary_path = OUTPUT_DIR / "probe_object_attribute_summary.csv"
    attribute_freq_path = OUTPUT_DIR / "probe_attribute_frequencies.csv"
    object_attribute_freq_path = (
        OUTPUT_DIR / "probe_object_attribute_frequencies.csv"
    )

    save_csv(
        summary_path,
        dataset_summary,
        ["metric", "train", "val"],
    )

    save_csv(
        object_summary_path,
        object_summary,
        [
            "object_label",
            "train_objects",
            "val_objects",
            "train_objects_with_attributes",
            "val_objects_with_attributes",
            "train_attribute_coverage_pct",
            "val_attribute_coverage_pct",
            "train_attribute_annotations",
            "val_attribute_annotations",
            "train_unique_attributes",
            "val_unique_attributes",
        ],
    )

    save_csv(
        attribute_freq_path,
        attribute_frequencies,
        [
            "attribute",
            "train_count",
            "val_count",
            "train_object_classes",
            "val_object_classes",
        ],
    )

    save_csv(
        object_attribute_freq_path,
        object_attribute_frequencies,
        [
            "object_label",
            "attribute",
            "train_count",
            "val_count",
            "train_objects_with_attribute",
            "val_objects_with_attribute",
            "train_object_coverage_pct",
            "val_object_coverage_pct",
        ],
    )

    print(f"Final object classes: {len(final_labels)}")
    print(
        "Selected object instances: "
        f"train={train['selected_instances_total']:,}, "
        f"val={val['selected_instances_total']:,}"
    )
    print(
        "Objects with >=1 attribute: "
        f"train={train['selected_with_attributes_total']:,} "
        f"({pct(train['selected_with_attributes_total'], train['selected_instances_total']):.2f}%), "
        f"val={val['selected_with_attributes_total']:,} "
        f"({pct(val['selected_with_attributes_total'], val['selected_instances_total']):.2f}%)"
    )
    print(
        "Attribute annotations: "
        f"train={train['attribute_annotations_total']:,}, "
        f"val={val['attribute_annotations_total']:,}"
    )
    print(
        "Unique attributes: "
        f"train={len(train['global_attribute_counts']):,}, "
        f"val={len(val['global_attribute_counts']):,}"
    )
    print()
    print(f"Saved: {summary_path}")
    print(f"Saved: {object_summary_path}")
    print(f"Saved: {attribute_freq_path}")
    print(f"Saved: {object_attribute_freq_path}")


if __name__ == "__main__":
    main()
