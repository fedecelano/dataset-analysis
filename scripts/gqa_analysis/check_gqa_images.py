
"""Check GQA image availability and scene-graph image-ID consistency.

Compare locally available image files with the train and validation
scene-graph IDs of the original or cleaned GQA dataset.

For the cleaned dataset, also verify consistency with the image-ID
lists stored in data/gqa_cleaned/image_ids/.

The script does not modify images, annotations, or image-ID lists.
Results are printed to the terminal and exported as CSV files.

Outputs are saved to:
    outputs/gqa_analysis/<dataset>/tables/

Usage:
    python scripts/gqa_analysis/check_gqa_images.py --dataset original
    python scripts/gqa_analysis/check_gqa_images.py --dataset cleaned
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "data" / "gqa" / "images"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

DATASET_DIRS = {
    "original": PROJECT_ROOT / "data" / "gqa",
    "cleaned": PROJECT_ROOT / "data" / "gqa_cleaned",
}

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "gqa_analysis"

SUMMARY_FIELDS = ("metric", "count")
ISSUE_FIELDS = ("issue_type", "split", "image_id", "details")


def parse_args() -> argparse.Namespace:
    """Parse the dataset version to check."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_DIRS),
        default="original",
        help="Dataset version to check (default: original).",
    )

    return parser.parse_args()


def load_json(path: Path) -> dict:
    """Load and validate the top-level GQA scene-graph structure."""
    if not path.is_file():
        raise FileNotFoundError(f"Scene-graph file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a dictionary of scene graphs in: {path}"
        )

    return data


def load_image_ids(path: Path) -> tuple[set[str], set[str]]:
    """Load distinct image IDs and identify IDs repeated in a list.

    Empty lines are ignored. Duplicate IDs are reported once,
    regardless of how many times they occur.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Image-ID list not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        ids = [line.strip() for line in file if line.strip()]

    counts = Counter(ids)

    unique_ids = set(counts)
    duplicate_ids = {
        image_id
        for image_id, count in counts.items()
        if count > 1
    }

    return unique_ids, duplicate_ids


def scan_image_files(
    image_dir: Path,
) -> tuple[list[Path], dict[str, list[Path]]]:
    """Index supported image files directly inside the image directory.

    Supported extensions are .jpg, .jpeg, and .png, matched
    case-insensitively. Subdirectories are not searched.
    """
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}"
        )

    image_files = sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name,
    )

    files_by_id = defaultdict(list)

    for path in image_files:
        files_by_id[path.stem].append(path)

    return image_files, dict(files_by_id)


def check_dataset(
    dataset: str,
) -> tuple[dict[str, int], list[dict[str, str]]]:
    """Collect image-integrity statistics and identifier-level issues."""
    data_dir = DATASET_DIRS[dataset]
    annotations_dir = data_dir / "annotations"

    image_files, files_by_id = scan_image_files(IMAGE_DIR)
    image_ids = set(files_by_id)

    train_data = load_json(
        annotations_dir / "train_sceneGraphs.json"
    )
    val_data = load_json(
        annotations_dir / "val_sceneGraphs.json"
    )

    train_ids = set(train_data)
    val_ids = set(val_data)
    graph_ids = train_ids | val_ids

    summary = {
        "image_files": len(image_files),
        "unique_image_ids": len(image_ids),
        "train_scene_graphs": len(train_ids),
        "validation_scene_graphs": len(val_ids),
        "unique_train_validation_scene_graphs": len(graph_ids),
        "images_with_scene_graph": len(image_ids & graph_ids),
        "images_without_scene_graph": len(image_ids - graph_ids),
        "scene_graphs_without_image": len(graph_ids - image_ids),
        "train_validation_overlap": len(train_ids & val_ids),
        "image_ids_with_multiple_files": sum(
            len(paths) > 1
            for paths in files_by_id.values()
        ),
    }

    issues: list[dict[str, str]] = []

    def add_issues(
        issue_type: str,
        ids: set[str],
        split: str = "",
        details: str = "",
    ) -> None:
        """Append one issue row for each identifier."""
        for image_id in sorted(ids):
            issues.append({
                "issue_type": issue_type,
                "split": split,
                "image_id": image_id,
                "details": details,
            })

    # Scene graphs referencing images that are not locally available.
    add_issues(
        "scene_graph_without_image",
        train_ids - image_ids,
        "train",
    )
    add_issues(
        "scene_graph_without_image",
        val_ids - image_ids,
        "validation",
    )

    # Informational: the image collection may include other GQA splits.
    add_issues(
        "image_without_train_validation_scene_graph",
        image_ids - graph_ids,
    )

    # Images appearing in both annotated splits.
    add_issues(
        "train_validation_overlap",
        train_ids & val_ids,
        "train,validation",
    )

    # Multiple supported image files sharing the same filename stem.
    for image_id, paths in sorted(files_by_id.items()):
        if len(paths) > 1:
            issues.append({
                "issue_type": "multiple_image_files_same_id",
                "split": "",
                "image_id": image_id,
                "details": ", ".join(path.name for path in paths),
            })

    # Cleaned datasets also provide explicit train/validation ID lists.
    if dataset == "cleaned":
        ids_dir = data_dir / "image_ids"

        for split, graph_set, filename in (
            ("train", train_ids, "train.txt"),
            ("validation", val_ids, "val.txt"),
        ):
            list_ids, duplicate_ids = load_image_ids(
                ids_dir / filename
            )

            summary[f"{split}_listed_ids"] = len(list_ids)
            summary[f"{split}_ids_in_graph_not_list"] = len(
                graph_set - list_ids
            )
            summary[f"{split}_ids_in_list_not_graph"] = len(
                list_ids - graph_set
            )
            summary[f"{split}_duplicate_ids_in_list"] = len(
                duplicate_ids
            )

            add_issues(
                "scene_graph_id_missing_from_list",
                graph_set - list_ids,
                split,
            )
            add_issues(
                "listed_id_missing_from_scene_graph",
                list_ids - graph_set,
                split,
            )
            add_issues(
                "duplicate_id_in_list",
                duplicate_ids,
                split,
            )

    issues.sort(
        key=lambda row: (
            row["issue_type"],
            row["split"],
            row["image_id"],
        )
    )

    return summary, issues


def write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict],
) -> None:
    """Write CSV rows, including the header when there are no records."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run image-integrity checks and export the results."""
    args = parse_args()

    summary, issues = check_dataset(args.dataset)

    output_dir = OUTPUT_ROOT / args.dataset / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "image_integrity_summary.csv"
    issues_path = output_dir / "image_integrity_issues.csv"

    write_csv(
        summary_path,
        SUMMARY_FIELDS,
        [
            {"metric": metric, "count": count}
            for metric, count in summary.items()
        ],
    )

    write_csv(
        issues_path,
        ISSUE_FIELDS,
        issues,
    )

    print(f"===== GQA IMAGE CHECK ({args.dataset}) =====")

    for metric, count in summary.items():
        print(f"{metric}: {count}")

    print(
        f"Issue rows (including informational records): {len(issues)}"
    )
    print(f"Saved: {summary_path}")
    print(f"Saved: {issues_path}")


if __name__ == "__main__":
    main()
