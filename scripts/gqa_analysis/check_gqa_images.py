"""Check GQA image availability and consistency of scene-graph image IDs.

The script is read-only with respect to the datasets. It checks image IDs
against train/validation scene graphs and, for the cleaned dataset, against
its train/validation image-ID lists. Results are printed and exported to CSV.

Run from any directory:
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

SUMMARY_FIELDS = ("metric", "count")
ISSUE_FIELDS = ("issue_type", "split", "image_id", "details")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check GQA scene-graph IDs against local images and split ID lists."
    )
    parser.add_argument(
        "--dataset",
        choices=("original", "cleaned"),
        default="original",
        help="Dataset version to check (default: original).",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    """Load a GQA scene-graph JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_image_ids(path: Path) -> tuple[set[str], set[str]]:
    """Read an ID list, returning distinct IDs and IDs repeated in the file."""
    with path.open("r", encoding="utf-8") as file:
        ids = [line.strip() for line in file if line.strip()]
    counts = Counter(ids)
    return set(counts), {image_id for image_id, count in counts.items() if count > 1}


def scan_image_files(image_dir: Path) -> tuple[list[Path], dict[str, list[Path]]]:
    """Find supported image files directly inside the image directory."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_files = sorted(
        (path for path in image_dir.iterdir()
         if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name,
    )
    files_by_id = defaultdict(list)
    for path in image_files:
        files_by_id[path.stem].append(path)
    return image_files, dict(files_by_id)


def check_dataset(dataset: str) -> tuple[dict[str, int], list[dict[str, str]]]:
    """Collect summary counts and record identifier-level integrity issues."""
    data_dir = PROJECT_ROOT / "data" / ("gqa" if dataset == "original" else "gqa_cleaned")
    annotations_dir = data_dir / "annotations"

    image_files, files_by_id = scan_image_files(IMAGE_DIR)
    image_ids = set(files_by_id)
    train_ids = set(load_json(annotations_dir / "train_sceneGraphs.json"))
    val_ids = set(load_json(annotations_dir / "val_sceneGraphs.json"))
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
        "image_ids_with_multiple_files": sum(len(paths) > 1 for paths in files_by_id.values()),
    }
    issues: list[dict[str, str]] = []

    def add_issues(issue_type: str, ids: set[str], split: str = "", details: str = "") -> None:
        for image_id in sorted(ids):
            issues.append({
                "issue_type": issue_type,
                "split": split,
                "image_id": image_id,
                "details": details,
            })

    add_issues("scene_graph_without_image", train_ids - image_ids, "train")
    add_issues("scene_graph_without_image", val_ids - image_ids, "validation")
    # Informational: GQA can contain images not covered by these two scene-graph splits.
    add_issues("image_without_train_validation_scene_graph", image_ids - graph_ids)
    add_issues("train_validation_overlap", train_ids & val_ids, "train,validation")

    for image_id, paths in sorted(files_by_id.items()):
        if len(paths) > 1:
            issues.append({
                "issue_type": "multiple_image_files_same_id",
                "split": "",
                "image_id": image_id,
                "details": ", ".join(path.name for path in paths),
            })

    if dataset == "cleaned":
        ids_dir = data_dir / "image_ids"
        for split, graph_set, filename in (
            ("train", train_ids, "train.txt"),
            ("validation", val_ids, "val.txt"),
        ):
            list_ids, duplicates = load_image_ids(ids_dir / filename)
            summary[f"{split}_listed_ids"] = len(list_ids)
            summary[f"{split}_ids_in_graph_not_list"] = len(graph_set - list_ids)
            summary[f"{split}_ids_in_list_not_graph"] = len(list_ids - graph_set)
            summary[f"{split}_duplicate_ids_in_list"] = len(duplicates)
            add_issues("scene_graph_id_missing_from_list", graph_set - list_ids, split)
            add_issues("listed_id_missing_from_scene_graph", list_ids - graph_set, split)
            add_issues("duplicate_id_in_list", duplicates, split)

    issues.sort(key=lambda row: (row["issue_type"], row["split"], row["image_id"]))
    return summary, issues


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    """Write a CSV file, including its header when no rows exist."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    summary, issues = check_dataset(args.dataset)

    output_dir = PROJECT_ROOT / "outputs" / "gqa_analysis" / args.dataset / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "image_integrity_summary.csv"
    issues_path = output_dir / "image_integrity_issues.csv"

    write_csv(
        summary_path,
        SUMMARY_FIELDS,
        [{"metric": metric, "count": count} for metric, count in summary.items()],
    )
    write_csv(issues_path, ISSUE_FIELDS, issues)

    print(f"===== GQA IMAGE CHECK ({args.dataset}) =====")
    for metric, count in summary.items():
        print(f"{metric}: {count}")
    print(f"Issue rows (including informational records): {len(issues)}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {issues_path}")


if __name__ == "__main__":
    main()
