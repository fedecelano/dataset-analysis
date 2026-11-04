#!/usr/bin/env python3
"""Materialize the frozen GQA probing subset inside its selection directory.

This script NEVER samples or changes the selected image IDs. It reads
train_ids.json, val_ids.json and test_ids.json from --subset-dir, exports
split-specific prepared scene graphs, and copies the corresponding source
images into images/{train,val,test}/.

The new train and validation use prepared GQA train annotations; the new test
uses prepared GQA validation annotations. The entire scene graph record is
preserved, including original-label provenance fields when present.

Existing export directories are never overwritten. The original images,
prepared annotations, frozen selection IDs and sampling statistics remain
untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EXPORT_DIRECTORIES = ("annotations", "images", "image_ids")
REPORT_NAME = "export_report.json"


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")


def safe_id(image_id: str) -> str:
    """Only allow image IDs suitable for a single filename stem."""
    if (not isinstance(image_id, str) or not image_id
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                   for c in image_id)):
        raise ValueError(f"Unsafe image ID: {image_id!r}")
    return image_id


def read_selection(subset_dir: Path) -> dict[str, list[str]]:
    """Read frozen IDs and validate uniqueness and split disjointness."""
    ids = {}
    for split in SPLITS:
        path = subset_dir / f"{split}_ids.json"
        values = read_json(path)
        if not isinstance(values, list):
            raise ValueError(f"{path}: expected a JSON list")
        values = [safe_id(v) for v in values]
        if len(values) != len(set(values)):
            raise ValueError(f"{path}: duplicate image IDs")
        if not values:
            raise ValueError(f"{path}: empty split")
        ids[split] = values

    for index, first in enumerate(SPLITS):
        for second in SPLITS[index + 1:]:
            overlap = set(ids[first]) & set(ids[second])
            if overlap:
                raise ValueError(
                    f"Overlapping IDs between {first} and {second}: "
                    f"{sorted(overlap)[:5]}"
                )
    return ids


def index_images(images_dir: Path) -> dict[str, list[Path]]:
    """Index image files by filename stem, ignoring unsupported extensions."""
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    indexed: dict[str, list[Path]] = {}
    for path in images_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            indexed.setdefault(path.stem, []).append(path)
    return indexed


def resolve_image(image_id: str, image_index: dict[str, list[Path]]) -> Path:
    """Require one physical image per ID; ignore case aliases to the same file."""
    candidates = image_index.get(image_id, [])
    unique: list[Path] = []
    for path in candidates:
        if not any(path.samefile(other) for other in unique):
            unique.append(path)
    if not unique:
        raise FileNotFoundError(f"No source image found for image ID {image_id}")
    if len(unique) != 1:
        raise ValueError(
            f"Multiple distinct source images for ID {image_id}: "
            f"{[str(p) for p in sorted(unique)]}"
        )
    return unique[0]


def export(subset_dir: Path, annotations_dir: Path, images_dir: Path) -> None:
    """Export the selected subset without modifying its frozen selection."""
    if not subset_dir.is_dir():
        raise FileNotFoundError(f"Subset directory not found: {subset_dir}")
    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # The source images must never overlap the export's destination tree.
    destination_images = subset_dir / "images"
    if (images_dir == destination_images or images_dir in destination_images.parents
            or destination_images in images_dir.parents):
        raise ValueError("Source images and export image directory must be disjoint")

    published_paths = [subset_dir / name for name in EXPORT_DIRECTORIES]
    published_paths.append(subset_dir / REPORT_NAME)
    existing = [path for path in published_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Export files already exist; selection files are preserved. "
            f"Remove or archive only the prior export paths before rerunning: {existing}"
        )

    ids = read_selection(subset_dir)
    source_graphs = {
        "train": read_json(annotations_dir / "train_sceneGraphs.json"),
        "val": read_json(annotations_dir / "val_sceneGraphs.json"),
    }
    if not all(isinstance(graphs, dict) for graphs in source_graphs.values()):
        raise ValueError("Prepared scene-graph files must contain JSON objects")
    if set(source_graphs["train"]) & set(source_graphs["val"]):
        raise ValueError("Prepared source train and validation image IDs overlap")

    image_index = index_images(images_dir)
    sources = {}
    for split in SPLITS:
        source_name = "val" if split == "test" else "train"
        graphs = source_graphs[source_name]
        missing = [image_id for image_id in ids[split] if image_id not in graphs]
        if missing:
            raise ValueError(
                f"{split}: {len(missing)} IDs absent from prepared {source_name} "
                f"annotations (examples: {missing[:5]})"
            )
        sources[split] = {
            image_id: resolve_image(image_id, image_index) for image_id in ids[split]
        }

    # Construct the export in a temporary directory within the subset parent.
    # Publish only after all copies and annotation files are complete.
    with tempfile.TemporaryDirectory(
        prefix=".probing_export_staging_", dir=subset_dir
    ) as temporary:
        staging = Path(temporary)
        report = {
            "description": "Frozen GQA probing subset (no resampling)",
            "source_annotations_dir": str(annotations_dir),
            "source_images_dir": str(images_dir),
            "splits": {},
        }
        for split in SPLITS:
            source_name = "val" if split == "test" else "train"
            selected = {
                image_id: source_graphs[source_name][image_id]
                for image_id in ids[split]
            }
            for image_id, graph in selected.items():
                if not isinstance(graph, dict) or not isinstance(graph.get("objects"), dict):
                    raise ValueError(f"Invalid scene graph for {split}/{image_id}")

            write_json(
                staging / "annotations" / f"{split}_sceneGraphs.json", selected
            )
            id_path = staging / "image_ids" / f"{split}.txt"
            id_path.parent.mkdir(parents=True, exist_ok=True)
            id_path.write_text(
                "".join(f"{image_id}\n" for image_id in ids[split]),
                encoding="utf-8",
            )

            image_destination = staging / "images" / split
            image_destination.mkdir(parents=True, exist_ok=True)
            extensions = Counter()
            for image_id in ids[split]:
                source_image = sources[split][image_id]
                shutil.copy2(source_image, image_destination / source_image.name)
                extensions[source_image.suffix.lower()] += 1

            report["splits"][split] = {
                "images": len(ids[split]),
                "scene_graphs": len(selected),
                "objects": sum(len(graph["objects"]) for graph in selected.values()),
                "annotation_source": f"{source_name}_sceneGraphs.json",
                "image_extensions": dict(sorted(extensions.items())),
            }
            print(f"{split}: exported {len(ids[split]):,} images and scene graphs")

        write_json(staging / REPORT_NAME, report)

        # Windows can refuse directory renames when an external process holds
        # a handle to a recently written file (WinError 5). A copy-based
        # fallback publishes the already verified staging data without
        # changing the frozen selection. The report is published LAST.
        published: list[Path] = []
        active_target: Path | None = None
        try:
            for name in EXPORT_DIRECTORIES:
                active_target = subset_dir / name
                if active_target.exists():
                    raise FileExistsError(
                        f"Destination appeared during export: {active_target}"
                    )
                staged_path = staging / name
                try:
                    staged_path.rename(active_target)
                except PermissionError:
                    if active_target.exists():
                        raise FileExistsError(
                            f"Destination appeared during export: {active_target}"
                        )
                    print(f"Windows blocked rename for {name}; copying instead.")
                    shutil.copytree(staged_path, active_target)
                published.append(active_target)
                active_target = None

            active_target = subset_dir / REPORT_NAME
            if active_target.exists():
                raise FileExistsError(
                    f"Destination appeared during export: {active_target}"
                )
            # Copying a small JSON report avoids a final Windows rename.
            shutil.copy2(staging / REPORT_NAME, active_target)
            published.append(active_target)
            active_target = None
        except Exception:
            # Undo only output paths created by this invocation. Never touch
            # frozen *_ids.json or sampling_stats.json.
            if active_target is not None and active_target.exists():
                if active_target.is_dir():
                    shutil.rmtree(active_target)
                else:
                    active_target.unlink()
            for path in reversed(published):
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            raise

    print(f"Export completed in: {subset_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset-dir", type=Path,
        default=ROOT / "data" / "gqa_probing" / "subsets" / "probing_5k",
        help="Directory with frozen *_ids.json; exported files are written here.",
    )
    parser.add_argument(
        "--annotations-dir", type=Path,
        default=ROOT / "data" / "gqa_probing" / "annotations",
    )
    parser.add_argument(
        "--images-dir", type=Path,
        default=ROOT / "data" / "gqa" / "images",
    )
    args = parser.parse_args()
    export(
        args.subset_dir.resolve(),
        args.annotations_dir.resolve(),
        args.images_dir.resolve(),
    )


if __name__ == "__main__":
    main()
