#!/usr/bin/env python3
"""Export the frozen GQA probing 5k subset for transfer to a compute cluster.

Reads existing train/val/test ID lists; NEVER resamples. Copies the original
image bytes and writes split-specific filtered scene-graph JSON files. The new
train and val use preprocessed GQA train annotations; test uses GQA val.

Does not modify original images, GQA annotations, subset IDs, or cleaning data.
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
SUFFIXES = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")


def safe_id(image_id: str) -> str:
    """Avoid writing unexpected paths from external JSON."""
    if not isinstance(image_id, str) or not image_id or image_id in (".", ".."):
        raise ValueError(f"Invalid image ID: {image_id!r}")
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in image_id):
        raise ValueError(f"Unsafe image ID: {image_id!r}")
    return image_id


def find_image(image_dir: Path, image_id: str) -> Path:
    """Find one physical image; Windows extension casing is case-insensitive.

    On Windows, both ``123.jpg`` and ``123.JPG`` can point to the same file.
    Only raise for genuinely distinct files with the same image ID.
    """
    candidates = [image_dir / (image_id + ext) for ext in SUFFIXES]
    matches: list[Path] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        if any(candidate.samefile(existing) for existing in matches):
            continue
        matches.append(candidate)
    if not matches:
        raise FileNotFoundError(
            f"No image for ID {image_id} in {image_dir} (jpg/jpeg/png)"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple DISTINCT images for ID {image_id}: {matches}. "
            "Resolve duplicates before exporting."
        )
    return matches[0]


def export(subset_dir: Path, annotations_dir: Path, images_dir: Path, output_dir: Path):
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}. Choose a new output path.")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Original image directory missing: {images_dir}")
    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"Preprocessed annotation directory missing: {annotations_dir}")

    source_graphs = {
        "train": read_json(annotations_dir / "train_sceneGraphs.json"),
        "val": read_json(annotations_dir / "val_sceneGraphs.json"),
    }
    ids = {}
    sources = {}
    for split in SPLITS:
        values = read_json(subset_dir / f"{split}_ids.json")
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ValueError(f"{split}_ids.json must contain a JSON list of string IDs")
        values = [safe_id(v) for v in values]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate image IDs in {split}_ids.json")
        ids[split] = values
        sources[split] = "val" if split == "test" else "train"

    for a in SPLITS:
        for b in SPLITS:
            if a < b and set(ids[a]) & set(ids[b]):
                raise ValueError(f"Overlapping image IDs between {a} and {b}")
    if set(source_graphs["train"]) & set(source_graphs["val"]):
        raise ValueError("Overlap between source GQA train and val scene graph IDs")

    # Preflight everything before copying to avoid incomplete exports.
    images = {}
    for split in SPLITS:
        graphs = source_graphs[sources[split]]
        missing = [image_id for image_id in ids[split] if image_id not in graphs]
        if missing:
            raise ValueError(f"{split}: {len(missing)} IDs missing in GQA {sources[split]} annotations; examples: {missing[:5]}")
        images[split] = {image_id: find_image(images_dir, image_id) for image_id in ids[split]}

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output_dir.name + ".staging.", dir=output_dir.parent) as staging_str:
        staging = Path(staging_str)
        report = {"description": "Frozen 3k/1k/1k GQA probing export", "splits": {}}
        for split in SPLITS:
            graphs = source_graphs[sources[split]]
            selected_graphs = {image_id: graphs[image_id] for image_id in ids[split]}
            write_json(staging / "annotations" / f"{split}_sceneGraphs.json", selected_graphs)
            write_json(staging / "image_ids" / f"{split}_ids.json", ids[split])
            image_root = staging / "images" / split
            image_root.mkdir(parents=True, exist_ok=True)
            extensions = Counter()
            for image_id in ids[split]:
                source = images[split][image_id]
                shutil.copy2(source, image_root / source.name)
                extensions[source.suffix.lower()] += 1
            report["splits"][split] = {
                "images": len(ids[split]),
                "scene_graphs": len(selected_graphs),
                "annotation_source": f"{sources[split]}_sceneGraphs.json",
                "image_extensions": dict(extensions),
                "objects": sum(len(scene["objects"]) for scene in selected_graphs.values()),
            }
            print(f"{split}: {len(ids[split]):,} original images copied; {len(selected_graphs):,} scene graphs")
        write_json(staging / "export_report.json", report)
        # Publish only after all data have been written. Windows sometimes
        # rejects directory rename (WinError 5) because a process temporarily
        # holds a handle inside the staging directory. Fall back to copying.
        try:
            staging.rename(output_dir)
        except PermissionError:
            if output_dir.exists():
                raise FileExistsError(
                    f"Cannot publish export: destination already exists: {output_dir}"
                )
            print("Directory rename blocked by Windows; publishing via copytree instead.")
            try:
                shutil.copytree(staging, output_dir)
            except Exception:
                # Do not leave a partial export looking like a completed one.
                shutil.rmtree(output_dir, ignore_errors=True)
                raise
            # Keep the complete report inside the published directory; the
            # TemporaryDirectory context will clean up staging automatically.
    print(f"Exported to: {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-dir", type=Path, default=ROOT / "data/gqa_probing/subsets/probing_5k")
    parser.add_argument("--annotations-dir", type=Path, default=ROOT / "data/gqa_probing/annotations")
    parser.add_argument("--images-dir", type=Path, default=ROOT / "data/gqa/images")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/gqa_probing/exports/probing_5k")
    args = parser.parse_args()
    export(args.subset_dir.resolve(), args.annotations_dir.resolve(), args.images_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
