#!/usr/bin/env python3
"""Prepare GQA scene graphs with 14 object, 5 attribute, and 7 relation labels.

Semantic object filtering precedes bounding-box validation. Selected objects
with invalid boxes are removed; partially out-of-frame boxes are clipped.
Attributes and relations are then filtered using the supplied mapping rules.
Source scene graphs and image files are never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path

VOCABS = ("object", "attribute", "relation")
EXPECTED_VOCAB_SIZES = {"object": 14, "attribute": 5, "relation": 7}


def load_rules(config_dir: Path) -> dict:
    """Load KEEP/MERGE/REMOVE mappings and validate their consistency."""
    rules = {}
    for kind in VOCABS:
        path = config_dir / f"{kind}_mapping_macro.json"
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict) or set(data) != {"keep", "merge", "remove"}:
            raise ValueError(f"Invalid mapping format: {path}")
        if not (isinstance(data["keep"], list) and isinstance(data["merge"], dict)
                and isinstance(data["remove"], list)):
            raise ValueError(f"Invalid mapping entry types: {path}")
        keep_list, remove_list, merge = data["keep"], data["remove"], data["merge"]
        labels = [*keep_list, *remove_list, *merge.keys(), *merge.values()]
        if not all(isinstance(label, str) and label for label in labels):
            raise ValueError(f"Labels must be nonempty strings: {path}")
        if len(set(keep_list)) != EXPECTED_VOCAB_SIZES[kind] or len(keep_list) != EXPECTED_VOCAB_SIZES[kind]:
            raise ValueError(f"Expected {EXPECTED_VOCAB_SIZES[kind]} unique KEEP labels: {path}")
        if len(set(remove_list)) != len(remove_list):
            raise ValueError(f"Duplicate REMOVE labels: {path}")
        keep, remove = set(keep_list), set(remove_list)
        if (keep & remove) or (keep & set(merge)) or (remove & set(merge)):
            raise ValueError(f"Conflicting actions: {path}")
        if not set(merge.values()) <= keep:
            raise ValueError(f"MERGE targets outside KEEP vocabulary: {path}")
        rules[kind] = {"keep": keep, "merge": merge, "remove": remove}
    return rules


def resolve(label, rules: dict, counts: Counter, kind: str) -> str | None:
    """Map a label to its canonical form, or None if it should be removed."""
    if not isinstance(label, str):
        counts[f"{kind}_invalid_label"] += 1
        return None
    if label in rules["keep"]:
        counts[f"{kind}_keep"] += 1
        return label
    if label in rules["merge"]:
        counts[f"{kind}_merge"] += 1
        return rules["merge"][label]
    if label in rules["remove"]:
        counts[f"{kind}_remove_explicit"] += 1
        return None
    counts[f"{kind}_remove_unlisted"] += 1
    return None


def positive_finite(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def finite_number(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def clip_bbox(obj: dict, image_width: float, image_height: float) -> tuple[str, dict | None]:
    """Return (valid|clippable|invalid, copied box or None).

    All coordinates use the GQA x/y/w/h format. A box is usable iff its
    intersection with [0,W] x [0,H] has strictly positive width and height.
    """
    x, y, w, h = (obj.get(key) for key in ("x", "y", "w", "h"))
    if not (finite_number(x) and finite_number(y)
            and positive_finite(w) and positive_finite(h)):
        return "invalid", None
    left, top = max(0, x), max(0, y)
    right, bottom = min(image_width, x + w), min(image_height, y + h)
    if right <= left or bottom <= top:
        return "invalid", None
    result = deepcopy(obj)
    if x >= 0 and y >= 0 and x + w <= image_width and y + h <= image_height:
        return "valid", result
    result.update(x=left, y=top, w=right - left, h=bottom - top)
    return "clippable", result


def preprocess_graph(image_graph: dict, rules: dict, counts: Counter) -> dict | None:
    """Filter one graph; keep only selected objects with usable bounding boxes."""
    if not isinstance(image_graph, dict) or not isinstance(image_graph.get("objects"), dict):
        raise ValueError("Expected scene graph with an objects dictionary")
    width, height = image_graph.get("width"), image_graph.get("height")
    if not positive_finite(width) or not positive_finite(height):
        raise ValueError(f"Invalid image dimensions: width={width!r}, height={height!r}")

    objects = image_graph["objects"]
    counts["objects_before"] += len(objects)
    # Gather true pre-filter annotation totals for auditing.
    for oid, source in objects.items():
        if not isinstance(source, dict):
            raise ValueError(f"Object {oid}: expected dictionary")
        attrs = source.get("attributes", [])
        relations = source.get("relations", [])
        if not isinstance(attrs, list) or not isinstance(relations, list):
            raise ValueError(f"Object {oid}: attributes and relations must be lists")
        counts["attributes_before"] += len(attrs)
        counts["relations_before"] += len(relations)

    kept = {}
    for oid, source in objects.items():
        new_name = resolve(source.get("name"), rules["object"], counts, "object")
        if new_name is None:
            counts["objects_removed_semantic"] += 1
            continue
        counts["objects_selected_semantic"] += 1
        status, obj = clip_bbox(source, width, height)
        counts[f"bbox_{status}"] += 1
        if status == "invalid":
            counts["objects_removed_invalid_bbox"] += 1
            continue
        obj["original_name"] = source.get("name")
        obj["name"] = new_name
        # Attribute filtering is performed only on surviving objects.
        original_attrs = source.get("attributes", [])
        counts["attributes_on_retained_objects_before"] += len(original_attrs)
        obj["original_attributes"] = deepcopy(original_attrs)
        obj["attributes"] = []
        for attr in original_attrs:
            mapped = resolve(attr, rules["attribute"], counts, "attribute")
            if mapped is not None:
                obj["attributes"].append(mapped)
                counts["attributes_after"] += 1
        kept[str(oid)] = obj

    counts["objects_after"] += len(kept)
    if not kept:
        return None

    for oid, obj in kept.items():
        source_relations = obj.get("relations", [])
        counts["relations_on_retained_objects_before"] += len(source_relations)
        obj["relations"] = []
        for rel in source_relations:
            if not isinstance(rel, dict):
                raise ValueError(f"Object {oid}: relation must be a dictionary")
            name = resolve(rel.get("name"), rules["relation"], counts, "relation")
            if name is None:
                continue
            if str(rel.get("object")) not in kept:
                counts["relations_removed_missing_target"] += 1
                continue
            new_rel = deepcopy(rel)
            new_rel["original_name"] = rel.get("name")
            new_rel["name"] = name
            obj["relations"].append(new_rel)
            counts["relations_after"] += 1

    result = dict(image_graph)
    result["objects"] = kept
    return result


def validate_filtered_graphs(graphs: dict, rules: dict) -> dict:
    """Check output labels, relation references, and in-frame positive boxes."""
    checked = Counter()
    errors = []
    for image_id, graph in graphs.items():
        objects = graph["objects"]
        width, height = graph["width"], graph["height"]
        if not objects:
            errors.append(f"{image_id}: no objects")
        for oid, obj in objects.items():
            checked["objects_checked"] += 1
            if obj.get("name") not in rules["object"]["keep"]:
                errors.append(f"{image_id}/{oid}: invalid object name")
            status, _ = clip_bbox(obj, width, height)
            if status != "valid":
                errors.append(f"{image_id}/{oid}: non-valid output bounding box")
            for attr in obj.get("attributes", []):
                checked["attributes_checked"] += 1
                if attr not in rules["attribute"]["keep"]:
                    errors.append(f"{image_id}/{oid}: invalid attribute {attr!r}")
            for rel in obj.get("relations", []):
                checked["relations_checked"] += 1
                if rel.get("name") not in rules["relation"]["keep"]:
                    errors.append(f"{image_id}/{oid}: invalid relation {rel.get('name')!r}")
                if str(rel.get("object")) not in objects:
                    errors.append(f"{image_id}/{oid}: dangling target {rel.get('object')!r}")
            if len(errors) >= 30:
                raise ValueError("Filtered graph validation failed:\n" + "\n".join(errors))
    if errors:
        raise ValueError("Filtered graph validation failed:\n" + "\n".join(errors))
    return {"status": "passed", **dict(checked)}


def process_file(source: Path, dest: Path, rules: dict) -> dict:
    """Write filtered graphs and split IDs; return preprocessing statistics."""
    with source.open(encoding="utf-8") as stream:
        graphs = json.load(stream)
    if not isinstance(graphs, dict):
        raise ValueError(f"{source}: expected image-ID-to-graph dictionary")
    counts = Counter()
    output = {}
    for image_id, scene in graphs.items():
        counts["images_before"] += 1
        try:
            filtered = preprocess_graph(scene, rules, counts)
        except ValueError as exc:
            raise ValueError(f"{source} image {image_id}: {exc}") from exc
        if filtered is None:
            counts["images_removed_without_selected_objects"] += 1
        else:
            output[image_id] = filtered
            counts["images_after"] += 1
    validation = validate_filtered_graphs(output, rules)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, separators=(",", ":"))
    ids_path = dest.parent.parent / "image_ids" / f"{dest.stem.removesuffix('_sceneGraphs')}.txt"
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.write_text("".join(f"{image_id}\n" for image_id in output), encoding="utf-8")
    result = dict(sorted(counts.items()))
    result["validation"] = validation
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=Path(__file__).resolve().parent / "config")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()
    input_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    if input_dir == output_dir or input_dir in output_dir.parents or output_dir in input_dir.parents:
        raise ValueError("Input and output directories must be disjoint")
    if output_dir.name == "gqa_cleaned":
        raise ValueError("Refusing to overwrite the legacy gqa_cleaned dataset")
    if len(set(args.splits)) != len(args.splits):
        raise ValueError("Duplicate splits requested")
    rules = load_rules(args.config_dir)
    summary = {}
    for split in args.splits:
        if not split.replace("_", "").isalnum():
            raise ValueError(f"Invalid split: {split}")
        source = input_dir / f"{split}_sceneGraphs.json"
        if not source.is_file():
            raise FileNotFoundError(source)
        dest = output_dir / "annotations" / source.name
        summary[split] = process_file(source, dest, rules)
        print(f"{split}: {summary[split].get('images_after', 0)} images retained")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preprocessing_stats.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
