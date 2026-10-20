import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

TRAIN_PATH = Path("data/gqa/annotations/train_sceneGraphs.json")
VAL_PATH = Path("data/gqa/annotations/val_sceneGraphs.json")
SELECTION_PATH = Path(
    "outputs/gqa_analysis/tables/object_probe_selection_134.csv"
)
OUTPUT_DIR = Path("outputs/gqa_analysis/tables")

EXPECTED_FINAL_OBJECT_CLASSES = 100


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
        Final object vocabulary used for probing.
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
                f"Unexpected action '{row['action']}' "
                f"for label '{source_label}'."
            )

        if not final_label:
            raise ValueError(
                f"Missing final_label for non-removed label "
                f"'{source_label}'."
            )

        source_to_final[source_label] = final_label

    final_labels = set(source_to_final.values())

    if len(final_labels) != EXPECTED_FINAL_OBJECT_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_FINAL_OBJECT_CLASSES} final object classes, "
            f"but found {len(final_labels)} in {path}."
        )

    return source_to_final, final_labels


# ---------------------------------------------------------------------
# GQA relation analysis
# ---------------------------------------------------------------------

def analyze_split(
    scene_graph_path: Path,
    source_to_final: dict[str, str],
):
    """
    Analyze GQA relations for the selected object vocabulary.

    A relation is included only if BOTH endpoints belong to the selected
    object vocabulary:

        selected source object -- relation --> selected target object

    Object relabeling is applied independently to source and target labels.

    GQA stores relations on the source object. Each relation is expected to
    contain:
        relation["name"]   -> relation label
        relation["object"] -> target object id
    """
    with scene_graph_path.open("r", encoding="utf-8") as f:
        scene_graphs = json.load(f)

    relation_counts = Counter()
    triple_counts = Counter()

    source_classes_per_relation = defaultdict(set)
    target_classes_per_relation = defaultdict(set)
    object_pairs_per_relation = defaultdict(set)

    source_object_instances = Counter()
    target_object_instances = Counter()

    selected_relation_instances = 0
    all_relation_instances_from_selected_sources = 0

    skipped_source_not_selected = 0
    skipped_target_not_selected = 0
    missing_target_object = 0
    missing_relation_name = 0
    missing_target_id = 0

    for image_data in scene_graphs.values():
        objects = image_data.get("objects", {}) or {}

        for source_id, source_obj in objects.items():
            source_label_raw = source_obj.get("name", "")

            relations = source_obj.get("relations", []) or []

            if source_label_raw not in source_to_final:
                skipped_source_not_selected += len(relations)
                continue

            source_label = source_to_final[source_label_raw]
            source_object_instances[source_label] += 1

            for relation in relations:
                all_relation_instances_from_selected_sources += 1

                relation_name = str(relation.get("name", "")).strip()
                target_id = relation.get("object")

                if not relation_name:
                    missing_relation_name += 1
                    continue

                if target_id is None:
                    missing_target_id += 1
                    continue

                # JSON object keys are strings in GQA.
                target_obj = objects.get(str(target_id))

                if target_obj is None:
                    # Keep this fallback in case a different loader preserves
                    # non-string dictionary keys.
                    target_obj = objects.get(target_id)

                if target_obj is None:
                    missing_target_object += 1
                    continue

                target_label_raw = target_obj.get("name", "")

                if target_label_raw not in source_to_final:
                    skipped_target_not_selected += 1
                    continue

                target_label = source_to_final[target_label_raw]

                relation_counts[relation_name] += 1
                triple_counts[
                    (source_label, relation_name, target_label)
                ] += 1

                source_classes_per_relation[relation_name].add(
                    source_label
                )
                target_classes_per_relation[relation_name].add(
                    target_label
                )
                object_pairs_per_relation[relation_name].add(
                    (source_label, target_label)
                )

                target_object_instances[target_label] += 1
                selected_relation_instances += 1

    return {
        "relation_counts": relation_counts,
        "triple_counts": triple_counts,
        "source_classes_per_relation": source_classes_per_relation,
        "target_classes_per_relation": target_classes_per_relation,
        "object_pairs_per_relation": object_pairs_per_relation,
        "source_object_instances": source_object_instances,
        "target_object_instances": target_object_instances,
        "selected_relation_instances": selected_relation_instances,
        "all_relation_instances_from_selected_sources":
            all_relation_instances_from_selected_sources,
        "skipped_source_not_selected": skipped_source_not_selected,
        "skipped_target_not_selected": skipped_target_not_selected,
        "missing_target_object": missing_target_object,
        "missing_relation_name": missing_relation_name,
        "missing_target_id": missing_target_id,
    }


# ---------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------

def pct(numerator: int, denominator: int):
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 4)


def build_dataset_summary(train, val, final_labels):
    return [
        {
            "metric": "final_object_classes",
            "train": len(final_labels),
            "val": len(final_labels),
        },
        {
            "metric": "selected_relation_instances",
            "train": train["selected_relation_instances"],
            "val": val["selected_relation_instances"],
        },
        {
            "metric": "unique_relation_labels",
            "train": len(train["relation_counts"]),
            "val": len(val["relation_counts"]),
        },
        {
            "metric": "relations_from_selected_sources",
            "train": train[
                "all_relation_instances_from_selected_sources"
            ],
            "val": val[
                "all_relation_instances_from_selected_sources"
            ],
        },
        {
            "metric": "selected_relation_pct_of_relations_from_selected_sources",
            "train": pct(
                train["selected_relation_instances"],
                train["all_relation_instances_from_selected_sources"],
            ),
            "val": pct(
                val["selected_relation_instances"],
                val["all_relation_instances_from_selected_sources"],
            ),
        },
        {
            "metric": "skipped_relations_source_not_selected",
            "train": train["skipped_source_not_selected"],
            "val": val["skipped_source_not_selected"],
        },
        {
            "metric": "skipped_relations_target_not_selected",
            "train": train["skipped_target_not_selected"],
            "val": val["skipped_target_not_selected"],
        },
        {
            "metric": "missing_target_object",
            "train": train["missing_target_object"],
            "val": val["missing_target_object"],
        },
        {
            "metric": "missing_relation_name",
            "train": train["missing_relation_name"],
            "val": val["missing_relation_name"],
        },
        {
            "metric": "missing_target_id",
            "train": train["missing_target_id"],
            "val": val["missing_target_id"],
        },
    ]


def build_relation_frequencies(train, val):
    relation_labels = (
        set(train["relation_counts"])
        | set(val["relation_counts"])
    )

    rows = []

    for relation in relation_labels:
        rows.append(
            {
                "relation": relation,
                "train_count": train["relation_counts"][relation],
                "val_count": val["relation_counts"][relation],
                "train_source_classes": len(
                    train["source_classes_per_relation"].get(
                        relation, set()
                    )
                ),
                "val_source_classes": len(
                    val["source_classes_per_relation"].get(
                        relation, set()
                    )
                ),
                "train_target_classes": len(
                    train["target_classes_per_relation"].get(
                        relation, set()
                    )
                ),
                "val_target_classes": len(
                    val["target_classes_per_relation"].get(
                        relation, set()
                    )
                ),
                "train_object_pairs": len(
                    train["object_pairs_per_relation"].get(
                        relation, set()
                    )
                ),
                "val_object_pairs": len(
                    val["object_pairs_per_relation"].get(
                        relation, set()
                    )
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            -row["train_count"],
            -row["val_count"],
            row["relation"],
        )
    )

    return rows


def build_relation_triples(train, val):
    triples = (
        set(train["triple_counts"])
        | set(val["triple_counts"])
    )

    rows = []

    for source_label, relation, target_label in triples:
        rows.append(
            {
                "source_label": source_label,
                "relation": relation,
                "target_label": target_label,
                "train_count": train["triple_counts"][
                    (source_label, relation, target_label)
                ],
                "val_count": val["triple_counts"][
                    (source_label, relation, target_label)
                ],
            }
        )

    rows.sort(
        key=lambda row: (
            -row["train_count"],
            -row["val_count"],
            row["relation"],
            row["source_label"],
            row["target_label"],
        )
    )

    return rows


def build_relation_class_pairs(train, val):
    """
    Aggregate relation counts at the ordered object-pair level.

    This table is useful for detecting relations that are strongly tied to a
    small set of object-class pairs, which may create shortcut learning in
    the probe.
    """
    train_pair_counts = Counter()
    val_pair_counts = Counter()

    for (source, relation, target), count in train[
        "triple_counts"
    ].items():
        train_pair_counts[(source, target)] += count

    for (source, relation, target), count in val[
        "triple_counts"
    ].items():
        val_pair_counts[(source, target)] += count

    pairs = set(train_pair_counts) | set(val_pair_counts)

    rows = []

    for source_label, target_label in pairs:
        train_relations = {
            relation
            for (source, relation, target), count
            in train["triple_counts"].items()
            if source == source_label
            and target == target_label
            and count > 0
        }

        val_relations = {
            relation
            for (source, relation, target), count
            in val["triple_counts"].items()
            if source == source_label
            and target == target_label
            and count > 0
        }

        rows.append(
            {
                "source_label": source_label,
                "target_label": target_label,
                "train_relation_instances":
                    train_pair_counts[(source_label, target_label)],
                "val_relation_instances":
                    val_pair_counts[(source_label, target_label)],
                "train_unique_relations": len(train_relations),
                "val_unique_relations": len(val_relations),
            }
        )

    rows.sort(
        key=lambda row: (
            -row["train_relation_instances"],
            -row["val_relation_instances"],
            row["source_label"],
            row["target_label"],
        )
    )

    return rows


# ---------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------

def save_csv(path: Path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_to_final, final_labels = load_object_selection(
        SELECTION_PATH
    )

    train = analyze_split(
        TRAIN_PATH,
        source_to_final,
    )
    val = analyze_split(
        VAL_PATH,
        source_to_final,
    )

    dataset_summary = build_dataset_summary(
        train,
        val,
        final_labels,
    )
    relation_frequencies = build_relation_frequencies(
        train,
        val,
    )
    relation_triples = build_relation_triples(
        train,
        val,
    )
    relation_class_pairs = build_relation_class_pairs(
        train,
        val,
    )

    summary_path = (
        OUTPUT_DIR / "probe_relation_dataset_summary.csv"
    )
    frequencies_path = (
        OUTPUT_DIR / "probe_relation_frequencies.csv"
    )
    triples_path = (
        OUTPUT_DIR / "probe_relation_triples.csv"
    )
    pairs_path = (
        OUTPUT_DIR / "probe_relation_class_pairs.csv"
    )

    save_csv(
        summary_path,
        dataset_summary,
        ["metric", "train", "val"],
    )

    save_csv(
        frequencies_path,
        relation_frequencies,
        [
            "relation",
            "train_count",
            "val_count",
            "train_source_classes",
            "val_source_classes",
            "train_target_classes",
            "val_target_classes",
            "train_object_pairs",
            "val_object_pairs",
        ],
    )

    save_csv(
        triples_path,
        relation_triples,
        [
            "source_label",
            "relation",
            "target_label",
            "train_count",
            "val_count",
        ],
    )

    save_csv(
        pairs_path,
        relation_class_pairs,
        [
            "source_label",
            "target_label",
            "train_relation_instances",
            "val_relation_instances",
            "train_unique_relations",
            "val_unique_relations",
        ],
    )

    print(f"Final object classes: {len(final_labels)}")
    print(
        "Train selected relation instances: "
        f"{train['selected_relation_instances']}"
    )
    print(
        "Val selected relation instances: "
        f"{val['selected_relation_instances']}"
    )
    print(
        "Train unique relation labels: "
        f"{len(train['relation_counts'])}"
    )
    print(
        "Val unique relation labels: "
        f"{len(val['relation_counts'])}"
    )
    print()
    print(f"Saved: {summary_path}")
    print(f"Saved: {frequencies_path}")
    print(f"Saved: {triples_path}")
    print(f"Saved: {pairs_path}")


if __name__ == "__main__":
    main()
