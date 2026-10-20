import csv
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

INPUT_PATH = Path("outputs/gqa_analysis/tables/class_frequencies.csv")
OUTPUT_DIR = Path("outputs/gqa_analysis/tables")

# The rules currently supplied cover ranks 1-134.
# With these 134 source labels, the cleaning below produces exactly
# 100 final object classes.
TOP_K = 134
EXPECTED_FINAL_CLASSES = 100


# ---------------------------------------------------------------------
# Cleaning rules
# ---------------------------------------------------------------------

OBJECT_RELABELING = {
    # People
    "man": "person",
    "woman": "person",
    "boy": "person",
    "girl": "person",
    "people": "person",

    # Singular / plural variants
    "leaves": "leaf",
    "trees": "tree",
    "windows": "window",
    "clouds": "cloud",
    "legs": "leg",
    "shoes": "shoe",
    "flowers": "flower",
    "rocks": "rock",
    "eyes": "eye",
    "wheels": "wheel",
    "hands": "hand",
    "feet": "foot",
    "bushes": "bush",

    # Coarser labels chosen for the probing vocabulary
    "street": "road",
    "post": "pole",
    "cap": "hat",
}


OBJECTS_TO_REMOVE = {
    "ground",
    "sky",
    "sign",
    "water",
    "floor",
    "letter",
    "glass",
    "food",
    "letters",
    "trunk",
    "wing",
    "ceiling",
    "sand",
}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def load_top_classes(path: Path, top_k: int):
    """Load class frequencies and return the top-k classes by train count."""
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in {path}")

    required_columns = {"label", "train_count", "val_count"}
    missing = required_columns - set(rows[0].keys())
    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {sorted(missing)}"
        )

    for row in rows:
        row["train_count"] = int(row["train_count"])
        row["val_count"] = int(row["val_count"])

    # Re-sort explicitly so the script does not depend on the CSV
    # already being ordered.
    rows.sort(
        key=lambda row: (
            -row["train_count"],
            -row["val_count"],
            row["label"],
        )
    )

    if len(rows) < top_k:
        raise ValueError(
            f"Requested top {top_k} classes, but only {len(rows)} are available."
        )

    return rows[:top_k]


def build_selection_table(top_classes):
    """
    Assign KEEP / MERGE / REMOVE to each selected source label.

    Labels not explicitly removed or relabelled are kept unchanged.
    """
    selection = []

    for rank, row in enumerate(top_classes, start=1):
        label = row["label"]

        if label in OBJECTS_TO_REMOVE:
            action = "REMOVE"
            final_label = ""
        elif label in OBJECT_RELABELING:
            action = "MERGE"
            final_label = OBJECT_RELABELING[label]
        else:
            action = "KEEP"
            final_label = label

        selection.append(
            {
                "rank": rank,
                "label": label,
                "train_count": row["train_count"],
                "val_count": row["val_count"],
                "action": action,
                "final_label": final_label,
            }
        )

    return selection


def validate_rules(selection):
    """Check that all configured labels are actually inside the top-k."""
    selected_labels = {row["label"] for row in selection}

    missing_relabel = set(OBJECT_RELABELING) - selected_labels
    missing_remove = OBJECTS_TO_REMOVE - selected_labels

    if missing_relabel:
        raise ValueError(
            "These relabel rules are not present in the selected top-k classes: "
            f"{sorted(missing_relabel)}"
        )

    if missing_remove:
        raise ValueError(
            "These remove rules are not present in the selected top-k classes: "
            f"{sorted(missing_remove)}"
        )


def build_final_vocabulary(selection):
    """
    Aggregate counts after merges and discard removed labels.

    Example:
        man + woman + boy + girl + people + person -> person
    """
    aggregated = defaultdict(
        lambda: {
            "train_count": 0,
            "val_count": 0,
            "source_labels": [],
        }
    )

    for row in selection:
        if row["action"] == "REMOVE":
            continue

        final_label = row["final_label"]

        aggregated[final_label]["train_count"] += row["train_count"]
        aggregated[final_label]["val_count"] += row["val_count"]
        aggregated[final_label]["source_labels"].append(row["label"])

    final_rows = []

    for label, values in aggregated.items():
        final_rows.append(
            {
                "label": label,
                "train_count": values["train_count"],
                "val_count": values["val_count"],
                "source_labels": " | ".join(values["source_labels"]),
            }
        )

    final_rows.sort(
        key=lambda row: (
            -row["train_count"],
            -row["val_count"],
            row["label"],
        )
    )

    for rank, row in enumerate(final_rows, start=1):
        row["rank"] = rank

    return final_rows


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

    top_classes = load_top_classes(INPUT_PATH, TOP_K)
    selection = build_selection_table(top_classes)
    validate_rules(selection)

    final_vocabulary = build_final_vocabulary(selection)

    if len(final_vocabulary) != EXPECTED_FINAL_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_FINAL_CLASSES} final classes, "
            f"but obtained {len(final_vocabulary)}."
        )

    selection_path = OUTPUT_DIR / "object_probe_selection_134.csv"
    vocabulary_path = OUTPUT_DIR / "object_probe_vocabulary_100.csv"

    save_csv(
        selection_path,
        selection,
        [
            "rank",
            "label",
            "train_count",
            "val_count",
            "action",
            "final_label",
        ],
    )

    save_csv(
        vocabulary_path,
        final_vocabulary,
        [
            "rank",
            "label",
            "train_count",
            "val_count",
            "source_labels",
        ],
    )

    n_keep = sum(row["action"] == "KEEP" for row in selection)
    n_merge = sum(row["action"] == "MERGE" for row in selection)
    n_remove = sum(row["action"] == "REMOVE" for row in selection)

    print(f"Selected source labels: {len(selection)}")
    print(f"KEEP:   {n_keep}")
    print(f"MERGE:  {n_merge}")
    print(f"REMOVE: {n_remove}")
    print(f"Final unique classes: {len(final_vocabulary)}")
    print()
    print(f"Saved: {selection_path}")
    print(f"Saved: {vocabulary_path}")


if __name__ == "__main__":
    main()
