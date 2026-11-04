#!/usr/bin/env python3
"""Prepare GQA probing annotations with 14/5/7 macro-label vocabularies.

Apply mappings directly to *original* GQA scene graphs, not to the
100/100/100 cleaned dataset. Filter objects before validating/clipping boxes;
then filter attributes and instance-level relations. Deduplicate canonical
attributes per object and canonical (source, relation, target) triples.

Retain source-label provenance in original_name, original_attributes, and
relation original_names. Source JSON files and images are never modified.

Example (run from the repository root):
    python scripts/gqa_probing/prepare_gqa_probing.py \
        --input-dir data/gqa/annotations --output-dir data/gqa_probing
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import tempfile

VOCABS = ("object", "attribute", "relation")
EXPECTED_VOCAB_SIZES = {"object": 14, "attribute": 5, "relation": 7}


def load_rules(config_dir: Path) -> dict:
    """Load KEEP/MERGE/REMOVE rules with mutually exclusive actions."""
    rules = {}
    for kind in VOCABS:
        path = config_dir / f"{kind}_mapping_macro.json"
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict) or set(data) != {"keep", "merge", "remove"}:
            raise ValueError(f"Invalid mapping format: {path}")
        if not (isinstance(data["keep"], list)
                and isinstance(data["merge"], dict)
                and isinstance(data["remove"], list)):
            raise ValueError(f"Invalid mapping entry types: {path}")
        keep_list, merge, remove_list = data["keep"], data["merge"], data["remove"]
        labels = [*keep_list, *merge.keys(), *merge.values(), *remove_list]
        if not all(isinstance(label, str) and label for label in labels):
            raise ValueError(f"Labels must be nonempty strings: {path}")
        if (len(keep_list) != EXPECTED_VOCAB_SIZES[kind]
                or len(set(keep_list)) != len(keep_list)):
            raise ValueError(f"Expected {EXPECTED_VOCAB_SIZES[kind]} unique KEEP labels: {path}")
        if len(set(remove_list)) != len(remove_list):
            raise ValueError(f"Duplicate REMOVE labels: {path}")
        keep, remove, merge_sources = set(keep_list), set(remove_list), set(merge)
        if keep & remove or keep & merge_sources or remove & merge_sources:
            raise ValueError(f"Conflicting mapping actions: {path}")
        if not set(merge.values()) <= keep:
            raise ValueError(f"MERGE targets outside KEEP vocabulary: {path}")
        rules[kind] = {"keep": keep, "merge": merge, "remove": remove}
    return rules


def resolve(label: object, rule: dict, counts: Counter, kind: str) -> str | None:
    """Map a label to a macro label (or discard it), recording the action."""
    if not isinstance(label, str):
        counts[f"{kind}_invalid_label"] += 1
        return None
    if label in rule["keep"]:
        counts[f"{kind}_keep"] += 1
        return label
    if label in rule["merge"]:
        counts[f"{kind}_merge"] += 1
        return rule["merge"][label]
    if label in rule["remove"]:
        counts[f"{kind}_remove_explicit"] += 1
        return None
    counts[f"{kind}_remove_unlisted"] += 1
    return None


def finite_number(value: object) -> bool:
    """Exclude nonnumeric, boolean, infinite, and NaN inputs."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def positive_finite(value: object) -> bool:
    return finite_number(value) and value > 0


def clip_bbox(obj: dict, image_width: float, image_height: float) -> tuple[str, dict | None]:
    """Return (valid|clippable|invalid, copied box or None)."""
    x, y, w, h = (obj.get(key) for key in ("x", "y", "w", "h"))
    if not (finite_number(x) and finite_number(y)
            and positive_finite(w) and positive_finite(h)
            and positive_finite(image_width) and positive_finite(image_height)):
        return "invalid", None
    try:
        right_edge, bottom_edge = x + w, y + h
    except OverflowError:
        return "invalid", None
    if not finite_number(right_edge) or not finite_number(bottom_edge):
        return "invalid", None

    left, top = max(0, x), max(0, y)
    right, bottom = min(image_width, right_edge), min(image_height, bottom_edge)
    if right <= left or bottom <= top:
        return "invalid", None
    result = deepcopy(obj)
    if x >= 0 and y >= 0 and right_edge <= image_width and bottom_edge <= image_height:
        return "valid", result
    result.update(x=left, y=top, w=right - left, h=bottom - top)
    return "clippable", result


def preprocess_graph(image_graph: dict, rules: dict, counts: Counter) -> dict | None:
    """Filter one original scene graph, retaining source-label provenance."""
    if not isinstance(image_graph, dict) or not isinstance(image_graph.get("objects"), dict):
        raise ValueError("Expected scene graph with an objects dictionary")
    width, height = image_graph.get("width"), image_graph.get("height")
    if not positive_finite(width) or not positive_finite(height):
        raise ValueError(f"Invalid image dimensions: width={width!r}, height={height!r}")

    objects = image_graph["objects"]
    counts["objects_before"] += len(objects)
    for oid, source in objects.items():
        if not isinstance(source, dict):
            raise ValueError(f"Object {oid}: expected dictionary")
        attrs, relations = source.get("attributes", []), source.get("relations", [])
        if not isinstance(attrs, list) or not isinstance(relations, list):
            raise ValueError(f"Object {oid}: attributes and relations must be lists")
        # Input must be the original graph. Prevent silent loss of earlier provenance.
        if "original_name" in source or "original_attributes" in source:
            raise ValueError(f"Object {oid}: provenance fields already exist; use original GQA input")
        for rel in relations:
            if not isinstance(rel, dict):
                raise ValueError(f"Object {oid}: relation must be a dictionary")
            if "original_name" in rel or "original_names" in rel:
                raise ValueError(f"Object {oid}: relation provenance already exists; use original GQA input")
        counts["attributes_before"] += len(attrs)
        counts["relations_before"] += len(relations)

    kept: dict[str, dict] = {}
    for oid, source in objects.items():
        original_name = source.get("name")
        new_name = resolve(original_name, rules["object"], counts, "object")
        if new_name is None:
            counts["objects_removed_semantic"] += 1
            continue
        counts["objects_selected_semantic"] += 1
        status, obj = clip_bbox(source, width, height)
        counts[f"bbox_{status}"] += 1
        if status == "invalid":
            counts["objects_removed_invalid_bbox"] += 1
            continue

        obj["name"] = new_name
        obj["original_name"] = original_name
        original_attrs = source.get("attributes", [])
        obj["original_attributes"] = deepcopy(original_attrs)
        counts["attributes_on_retained_objects_before"] += len(original_attrs)
        mapped_attrs = []
        seen_attrs = set()
        for attr in original_attrs:
            mapped = resolve(attr, rules["attribute"], counts, "attribute")
            if mapped is None:
                continue
            if mapped in seen_attrs:
                counts["attributes_removed_duplicate_after_mapping"] += 1
                continue
            seen_attrs.add(mapped)
            mapped_attrs.append(mapped)
            counts["attributes_after"] += 1
        obj["attributes"] = mapped_attrs
        kept[str(oid)] = obj

    counts["objects_after"] += len(kept)
    if not kept:
        return None

    for oid, obj in kept.items():
        original_relations = obj.get("relations", [])
        counts["relations_on_retained_objects_before"] += len(original_relations)
        mapped_rels = []
        seen_rels: dict[tuple[str, str], dict] = {}
        for rel in original_relations:
            original_rel_name = rel.get("name")
            mapped = resolve(original_rel_name, rules["relation"], counts, "relation")
            if mapped is None:
                continue
            target_id = str(rel.get("object"))
            if target_id not in kept:
                counts["relations_removed_missing_target"] += 1
                continue
            key = (mapped, target_id)  # Source oid is fixed in this loop.
            if key in seen_rels:
                seen_rels[key]["original_names"].append(original_rel_name)
                counts["relations_removed_duplicate_after_mapping"] += 1
                continue
            new_rel = deepcopy(rel)
            new_rel["name"] = mapped
            new_rel["object"] = target_id
            new_rel["original_name"] = original_rel_name
            new_rel["original_names"] = [original_rel_name]
            mapped_rels.append(new_rel)
            seen_rels[key] = new_rel
            counts["relations_after"] += 1
        obj["relations"] = mapped_rels

    result = dict(image_graph)
    result["objects"] = kept
    return result


def validate_filtered_graphs(graphs: dict, rules: dict) -> dict:
    """Validate canonical labels, unique triples, geometry, and provenance."""
    checked = Counter()
    errors = []
    for image_id, graph in graphs.items():
        objects = graph["objects"]
        width, height = graph["width"], graph["height"]
        if not objects:
            errors.append(f"{image_id}: empty graph")
        for oid, obj in objects.items():
            checked["objects_checked"] += 1
            if obj.get("name") not in rules["object"]["keep"]:
                errors.append(f"{image_id}/{oid}: invalid macro object")
            original_name = obj.get("original_name")
            if not isinstance(original_name, str) or (original_name not in rules["object"]["keep"]
                    and rules["object"]["merge"].get(original_name) != obj.get("name")):
                errors.append(f"{image_id}/{oid}: invalid original object provenance")
            status, _ = clip_bbox(obj, width, height)
            if status != "valid":
                errors.append(f"{image_id}/{oid}: output box is not valid")

            attrs = obj.get("attributes")
            original_attrs = obj.get("original_attributes")
            if not isinstance(attrs, list) or not isinstance(original_attrs, list):
                errors.append(f"{image_id}/{oid}: missing attribute lists")
                attrs = attrs if isinstance(attrs, list) else []
                original_attrs = original_attrs if isinstance(original_attrs, list) else []
            if len(attrs) != len(set(attrs)):
                errors.append(f"{image_id}/{oid}: duplicate macro attributes")
            expected_attrs = []
            for original_attr in original_attrs:
                canonical = (original_attr if isinstance(original_attr, str)
                             and original_attr in rules["attribute"]["keep"]
                             else rules["attribute"]["merge"].get(original_attr)
                             if isinstance(original_attr, str) else None)
                if canonical is not None and canonical not in expected_attrs:
                    expected_attrs.append(canonical)
            if attrs != expected_attrs:
                errors.append(f"{image_id}/{oid}: attribute provenance mismatch")
            checked["attributes_checked"] += len(attrs)

            relations = obj.get("relations")
            if not isinstance(relations, list):
                errors.append(f"{image_id}/{oid}: invalid relations list")
                relations = []
            seen_relations = set()
            for rel in relations:
                if not isinstance(rel, dict):
                    errors.append(f"{image_id}/{oid}: invalid relation structure")
                    continue
                checked["relations_checked"] += 1
                name = rel.get("name")
                target = str(rel.get("object"))
                if name not in rules["relation"]["keep"]:
                    errors.append(f"{image_id}/{oid}: invalid macro relation")
                if target not in objects:
                    errors.append(f"{image_id}/{oid}: dangling relation target {target}")
                key = (name, target)
                if key in seen_relations:
                    errors.append(f"{image_id}/{oid}: duplicate macro triple {key}")
                seen_relations.add(key)
                names = rel.get("original_names")
                if not isinstance(names, list) or not names or rel.get("original_name") != names[0]:
                    errors.append(f"{image_id}/{oid}: invalid relation provenance")
                else:
                    for original in names:
                        canonical = (original if isinstance(original, str)
                                     and original in rules["relation"]["keep"]
                                     else rules["relation"]["merge"].get(original)
                                     if isinstance(original, str) else None)
                        if canonical != name:
                            errors.append(f"{image_id}/{oid}: relation provenance mismatch")
            if len(errors) >= 30:
                raise ValueError("Probing graph validation failed:\n" + "\n".join(errors))
    if errors:
        raise ValueError("Probing graph validation failed:\n" + "\n".join(errors))
    return {"status": "passed", **dict(checked)}


def validate_statistics(counts: Counter, split: str) -> None:
    """Verify accounting identities, including both deduplication stages."""
    checks = {
        "images": (counts["images_before"],
                   counts["images_after"] + counts["images_removed_without_selected_objects"]),
        "objects": (counts["objects_before"], counts["objects_removed_semantic"]
                    + counts["objects_removed_invalid_bbox"] + counts["objects_after"]),
        "selected boxes": (counts["objects_selected_semantic"], counts["bbox_valid"]
                           + counts["bbox_clippable"] + counts["bbox_invalid"]),
        "attributes evaluated": (counts["attributes_on_retained_objects_before"],
                                 sum(counts[f"attribute_{action}"] for action in
                                     ("keep", "merge", "remove_explicit", "remove_unlisted", "invalid_label"))),
        "attributes retained": (counts["attribute_keep"] + counts["attribute_merge"],
                                counts["attributes_after"]
                                + counts["attributes_removed_duplicate_after_mapping"]),
        "relations evaluated": (counts["relations_on_retained_objects_before"],
                                sum(counts[f"relation_{action}"] for action in
                                    ("keep", "merge", "remove_explicit", "remove_unlisted", "invalid_label"))),
        "relations retained": (counts["relation_keep"] + counts["relation_merge"],
                               counts["relations_removed_missing_target"]
                               + counts["relations_removed_duplicate_after_mapping"]
                               + counts["relations_after"]),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"{split}: {label} counts disagree ({actual} != {expected})")


def write_text_atomic(path: Path, content: str) -> None:
    """Replace one output file only after its entire content is written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as stream:
            tmp_path = Path(stream.name)
            stream.write(content)
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def process_file(source: Path, dest: Path, rules: dict) -> dict:
    """Process one split and write annotations plus its retained image IDs."""
    with source.open("r", encoding="utf-8") as stream:
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
            raise ValueError(f"{source}, image {image_id}: {exc}") from exc
        if filtered is None:
            counts["images_removed_without_selected_objects"] += 1
        else:
            output[image_id] = filtered
            counts["images_after"] += 1

    validation = validate_filtered_graphs(output, rules)
    split = source.stem.removesuffix("_sceneGraphs")
    validate_statistics(counts, split)
    ids_path = dest.parent.parent / "image_ids" / f"{split}.txt"
    write_text_atomic(dest, json.dumps(output, ensure_ascii=False, separators=(",", ":")))
    write_text_atomic(ids_path, "".join(f"{image_id}\n" for image_id in output))
    result = dict(sorted(counts.items()))
    result["validation"] = validation
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Original GQA scene-graph directory (data/gqa/annotations).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Root directory for the macro probing dataset.")
    parser.add_argument("--config-dir", type=Path,
                        default=Path(__file__).resolve().parent / "config")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()
    input_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    if (input_dir == output_dir or input_dir in output_dir.parents
            or output_dir in input_dir.parents):
        parser.error("Input and output directories must be disjoint")
    if output_dir.name == "gqa_cleaned":
        parser.error("Refusing to overwrite the 100/100/100 cleaned dataset")
    if len(set(args.splits)) != len(args.splits):
        parser.error("Duplicate splits requested")
    if any(not split or not split.replace("_", "").isalnum() for split in args.splits):
        parser.error("Invalid split name")
    rules = load_rules(args.config_dir)
    for split in args.splits:
        source = input_dir / f"{split}_sceneGraphs.json"
        if not source.is_file():
            raise FileNotFoundError(source)

    summary = {}
    for split in args.splits:
        source = input_dir / f"{split}_sceneGraphs.json"
        dest = output_dir / "annotations" / source.name
        summary[split] = process_file(source, dest, rules)
        print(f"{split}: {summary[split].get('images_after', 0)} images retained")
    write_text_atomic(output_dir / "preprocessing_stats.json",
                      json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
