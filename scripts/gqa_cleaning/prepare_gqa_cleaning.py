
#!/usr/bin/env python3
"""Filter GQA scene graphs using canonical 100-label vocabularies.

Objects are filtered semantically before bounding-box validation.
Invalid boxes are removed and partially out-of-frame boxes are clipped.

Attributes and relations are normalized using KEEP/MERGE/REMOVE rules.
Duplicate canonical attributes are removed within each retained object.

Original labels are preserved for retained annotations:
- original_name for objects;
- original_attributes for objects;
- original_name for retained relations.

Source annotations and images are never modified.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from copy import deepcopy
from pathlib import Path


VOCABS = ("object", "attribute", "relation")
EXPECTED_VOCAB_SIZE = 100


def load_rules(config_dir: Path) -> dict:
    """Load and validate KEEP/MERGE/REMOVE mapping rules."""
    rules = {}

    for kind in VOCABS:
        path = config_dir / f"{kind}_mapping.json"

        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)

        if (
            not isinstance(data, dict)
            or set(data) != {"keep", "merge", "remove"}
        ):
            raise ValueError(f"Invalid mapping format: {path}")

        if not (
            isinstance(data["keep"], list)
            and isinstance(data["merge"], dict)
            and isinstance(data["remove"], list)
        ):
            raise ValueError(f"Invalid mapping entry types: {path}")

        keep_list = data["keep"]
        remove_list = data["remove"]
        merge = data["merge"]

        labels = [
            *keep_list,
            *remove_list,
            *merge.keys(),
            *merge.values(),
        ]

        if not all(
            isinstance(label, str) and label
            for label in labels
        ):
            raise ValueError(
                f"Labels must be nonempty strings: {path}"
            )

        if (
            len(keep_list) != EXPECTED_VOCAB_SIZE
            or len(set(keep_list)) != EXPECTED_VOCAB_SIZE
        ):
            raise ValueError(
                f"Expected {EXPECTED_VOCAB_SIZE} unique "
                f"KEEP labels: {path}"
            )

        if len(set(remove_list)) != len(remove_list):
            raise ValueError(f"Duplicate REMOVE labels: {path}")

        keep = set(keep_list)
        remove = set(remove_list)
        merge_sources = set(merge)

        if (
            keep & remove
            or keep & merge_sources
            or remove & merge_sources
        ):
            raise ValueError(f"Conflicting actions: {path}")

        if not set(merge.values()) <= keep:
            raise ValueError(
                f"MERGE targets outside KEEP vocabulary: {path}"
            )

        rules[kind] = {
            "keep": keep,
            "merge": merge,
            "remove": remove,
        }

    return rules


def resolve(
    label: object,
    rules: dict,
    counts: Counter,
    kind: str,
) -> str | None:
    """Resolve a label or return None if it should be removed."""
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


def finite_number(value: object) -> bool:
    """Check whether a value is a finite real number, excluding booleans."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False

    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def positive_finite(value: object) -> bool:
    """Check whether a value is finite and strictly positive."""
    return finite_number(value) and value > 0


def clip_bbox(
    obj: dict,
    image_width: float,
    image_height: float,
) -> tuple[str, dict | None]:
    """Validate and optionally clip an (x, y, w, h) bounding box.

    Return:
        ("valid", copied object)
        ("clippable", copied object with clipped coordinates)
        ("invalid", None)
    """
    x, y, w, h = (
        obj.get(key)
        for key in ("x", "y", "w", "h")
    )

    if not (
        finite_number(x)
        and finite_number(y)
        and positive_finite(w)
        and positive_finite(h)
        and positive_finite(image_width)
        and positive_finite(image_height)
    ):
        return "invalid", None

    try:
        right_edge = x + w
        bottom_edge = y + h
    except OverflowError:
        return "invalid", None

    if not (
        finite_number(right_edge)
        and finite_number(bottom_edge)
    ):
        return "invalid", None

    left = max(0, x)
    top = max(0, y)
    right = min(image_width, right_edge)
    bottom = min(image_height, bottom_edge)

    if right <= left or bottom <= top:
        return "invalid", None

    result = deepcopy(obj)

    if (
        x >= 0
        and y >= 0
        and right_edge <= image_width
        and bottom_edge <= image_height
    ):
        return "valid", result

    result.update(
        x=left,
        y=top,
        w=right - left,
        h=bottom - top,
    )

    return "clippable", result


def preprocess_graph(
    image_graph: dict,
    rules: dict,
    counts: Counter,
) -> dict | None:
    """Filter one scene graph while preserving retained original labels."""
    if (
        not isinstance(image_graph, dict)
        or not isinstance(image_graph.get("objects"), dict)
    ):
        raise ValueError(
            "Expected scene graph with an objects dictionary"
        )

    width = image_graph.get("width")
    height = image_graph.get("height")

    if not positive_finite(width) or not positive_finite(height):
        raise ValueError(
            f"Invalid image dimensions: "
            f"width={width!r}, height={height!r}"
        )

    objects = image_graph["objects"]
    counts["objects_before"] += len(objects)

    # Count annotations before any filtering.
    for oid, source in objects.items():
        if not isinstance(source, dict):
            raise ValueError(
                f"Object {oid}: expected dictionary"
            )

        attributes = source.get("attributes", [])
        relations = source.get("relations", [])

        if (
            not isinstance(attributes, list)
            or not isinstance(relations, list)
        ):
            raise ValueError(
                f"Object {oid}: attributes and relations must be lists"
            )

        # Avoid silently overwriting existing provenance fields.
        if (
            "original_name" in source
            or "original_attributes" in source
        ):
            raise ValueError(
                f"Object {oid}: reserved original-label "
                "fields already exist"
            )

        for relation in relations:
            if (
                isinstance(relation, dict)
                and "original_name" in relation
            ):
                raise ValueError(
                    f"Object {oid}: relation already contains "
                    "reserved original_name field"
                )

        counts["attributes_before"] += len(attributes)
        counts["relations_before"] += len(relations)

    kept = {}

    # Step 1: semantic object filtering and bounding-box validation.
    for oid, source in objects.items():
        original_name = source.get("name")

        canonical_name = resolve(
            original_name,
            rules["object"],
            counts,
            "object",
        )

        if canonical_name is None:
            counts["objects_removed_semantic"] += 1
            continue

        counts["objects_selected_semantic"] += 1

        status, obj = clip_bbox(source, width, height)
        counts[f"bbox_{status}"] += 1

        if status == "invalid":
            counts["objects_removed_invalid_bbox"] += 1
            continue

        obj["original_name"] = original_name
        obj["name"] = canonical_name

        # Step 2: attribute filtering and canonical deduplication.
        original_attributes = source.get("attributes", [])
        obj["original_attributes"] = deepcopy(original_attributes)

        counts["attributes_on_retained_objects_before"] += len(
            original_attributes
        )

        canonical_attributes = []
        seen_attributes = set()

        for attribute in original_attributes:
            canonical_attribute = resolve(
                attribute,
                rules["attribute"],
                counts,
                "attribute",
            )

            if canonical_attribute is None:
                continue

            if canonical_attribute in seen_attributes:
                counts["attributes_removed_duplicate_after_mapping"] += 1
                continue

            seen_attributes.add(canonical_attribute)
            canonical_attributes.append(canonical_attribute)
            counts["attributes_after"] += 1

        obj["attributes"] = canonical_attributes
        kept[str(oid)] = obj

    counts["objects_after"] += len(kept)

    if not kept:
        return None

    # Step 3: relation filtering with target-reference validation.
    for oid, obj in kept.items():
        source_relations = obj.get("relations", [])

        counts["relations_on_retained_objects_before"] += len(
            source_relations
        )

        filtered_relations = []

        for relation in source_relations:
            if not isinstance(relation, dict):
                raise ValueError(
                    f"Object {oid}: relation must be a dictionary"
                )

            original_relation_name = relation.get("name")

            canonical_relation_name = resolve(
                original_relation_name,
                rules["relation"],
                counts,
                "relation",
            )

            if canonical_relation_name is None:
                continue

            target_id = str(relation.get("object"))

            if target_id not in kept:
                counts["relations_removed_missing_target"] += 1
                continue

            new_relation = deepcopy(relation)
            new_relation["original_name"] = original_relation_name
            new_relation["name"] = canonical_relation_name

            filtered_relations.append(new_relation)
            counts["relations_after"] += 1

        obj["relations"] = filtered_relations

    result = dict(image_graph)
    result["objects"] = kept

    return result


def validate_filtered_graphs(
    graphs: dict,
    rules: dict,
) -> dict:
    """Validate canonical annotations, provenance, and geometry."""
    checked = Counter()
    errors = []

    for image_id, graph in graphs.items():
        objects = graph["objects"]
        width = graph["width"]
        height = graph["height"]

        if not objects:
            errors.append(f"{image_id}: no objects")

        for oid, obj in objects.items():
            checked["objects_checked"] += 1

            if obj.get("name") not in rules["object"]["keep"]:
                errors.append(
                    f"{image_id}/{oid}: invalid object name"
                )

            if not isinstance(obj.get("original_name"), str):
                errors.append(
                    f"{image_id}/{oid}: missing original object name"
                )

            status, _ = clip_bbox(obj, width, height)

            if status != "valid":
                errors.append(
                    f"{image_id}/{oid}: non-valid output bounding box"
                )

            attributes = obj.get("attributes")
            original_attributes = obj.get("original_attributes")

            if not isinstance(attributes, list):
                errors.append(
                    f"{image_id}/{oid}: invalid attributes list"
                )
                attributes = []

            if not isinstance(original_attributes, list):
                errors.append(
                    f"{image_id}/{oid}: missing original attributes"
                )

            if len(attributes) != len(set(attributes)):
                errors.append(
                    f"{image_id}/{oid}: duplicate canonical attributes"
                )

            for attribute in attributes:
                checked["attributes_checked"] += 1

                if attribute not in rules["attribute"]["keep"]:
                    errors.append(
                        f"{image_id}/{oid}: "
                        f"invalid attribute {attribute!r}"
                    )

            relations = obj.get("relations")

            if not isinstance(relations, list):
                errors.append(
                    f"{image_id}/{oid}: invalid relations list"
                )
                relations = []

            for relation in relations:
                checked["relations_checked"] += 1

                if not isinstance(relation, dict):
                    errors.append(
                        f"{image_id}/{oid}: invalid relation structure"
                    )
                    continue

                if relation.get("name") not in rules["relation"]["keep"]:
                    errors.append(
                        f"{image_id}/{oid}: "
                        f"invalid relation {relation.get('name')!r}"
                    )

                if not isinstance(relation.get("original_name"), str):
                    errors.append(
                        f"{image_id}/{oid}: "
                        "missing original relation name"
                    )

                if str(relation.get("object")) not in objects:
                    errors.append(
                        f"{image_id}/{oid}: "
                        f"dangling target {relation.get('object')!r}"
                    )

            if len(errors) >= 30:
                raise ValueError(
                    "Filtered graph validation failed:\n"
                    + "\n".join(errors)
                )

    if errors:
        raise ValueError(
            "Filtered graph validation failed:\n"
            + "\n".join(errors)
        )

    return {
        "status": "passed",
        **dict(checked),
    }


def validate_statistics(counts: Counter, split: str) -> None:
    """Verify fundamental preprocessing accounting identities."""
    object_balance = (
        counts["objects_removed_semantic"]
        + counts["objects_removed_invalid_bbox"]
        + counts["objects_after"]
    )

    if counts["objects_before"] != object_balance:
        raise ValueError(
            f"{split}: inconsistent object filtering statistics"
        )

    bbox_balance = (
        counts["bbox_valid"]
        + counts["bbox_clippable"]
        + counts["bbox_invalid"]
    )

    if counts["objects_selected_semantic"] != bbox_balance:
        raise ValueError(
            f"{split}: inconsistent bounding-box statistics"
        )

    if counts["bbox_invalid"] != counts["objects_removed_invalid_bbox"]:
        raise ValueError(
            f"{split}: inconsistent invalid-box statistics"
        )

    if counts["images_before"] != (
        counts["images_after"]
        + counts["images_removed_without_selected_objects"]
    ):
        raise ValueError(
            f"{split}: inconsistent image filtering statistics"
        )

    attribute_balance = (
        counts["attribute_keep"]
        + counts["attribute_merge"]
        + counts["attribute_remove_explicit"]
        + counts["attribute_remove_unlisted"]
        + counts["attribute_invalid_label"]
    )

    if (
        counts["attributes_on_retained_objects_before"]
        != attribute_balance
    ):
        raise ValueError(
            f"{split}: inconsistent attribute mapping statistics"
        )

    if (
        counts["attribute_keep"]
        + counts["attribute_merge"]
        != counts["attributes_after"]
        + counts["attributes_removed_duplicate_after_mapping"]
    ):
        raise ValueError(
            f"{split}: inconsistent attribute deduplication statistics"
        )

    relation_balance = (
        counts["relation_keep"]
        + counts["relation_merge"]
        + counts["relation_remove_explicit"]
        + counts["relation_remove_unlisted"]
        + counts["relation_invalid_label"]
    )

    if (
        counts["relations_on_retained_objects_before"]
        != relation_balance
    ):
        raise ValueError(
            f"{split}: inconsistent relation mapping statistics"
        )

    if (
        counts["relation_keep"]
        + counts["relation_merge"]
        != counts["relations_after"]
        + counts["relations_removed_missing_target"]
    ):
        raise ValueError(
            f"{split}: inconsistent relation filtering statistics"
        )


def process_file(
    source: Path,
    dest: Path,
    rules: dict,
) -> dict:
    """Filter one split, validate it, and export annotations and IDs."""
    with source.open("r", encoding="utf-8") as stream:
        graphs = json.load(stream)

    if not isinstance(graphs, dict):
        raise ValueError(
            f"{source}: expected image-ID-to-graph dictionary"
        )

    split = source.stem.removesuffix("_sceneGraphs")

    counts = Counter()
    output = {}

    for image_id, scene in graphs.items():
        counts["images_before"] += 1

        try:
            filtered = preprocess_graph(scene, rules, counts)
        except ValueError as exc:
            raise ValueError(
                f"{source} image {image_id}: {exc}"
            ) from exc

        if filtered is None:
            counts["images_removed_without_selected_objects"] += 1
        else:
            output[image_id] = filtered
            counts["images_after"] += 1

    validation = validate_filtered_graphs(output, rules)
    validate_statistics(counts, split)

    dest.parent.mkdir(parents=True, exist_ok=True)

    with dest.open("w", encoding="utf-8") as stream:
        json.dump(
            output,
            stream,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    ids_path = (
        dest.parent.parent
        / "image_ids"
        / f"{split}.txt"
    )

    ids_path.parent.mkdir(parents=True, exist_ok=True)

    ids_path.write_text(
        "".join(
            f"{image_id}\n"
            for image_id in output
        ),
        encoding="utf-8",
    )

    result = dict(sorted(counts.items()))
    result["validation"] = validation

    return result


def main() -> None:
    """Run GQA preprocessing for the requested splits."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "config",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
    )

    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if (
        input_dir == output_dir
        or input_dir in output_dir.parents
        or output_dir in input_dir.parents
    ):
        raise ValueError(
            "Input and output directories must be disjoint"
        )

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

        dest = (
            output_dir
            / "annotations"
            / source.name
        )

        summary[split] = process_file(
            source,
            dest,
            rules,
        )

        print(
            f"{split}: "
            f"{summary[split].get('images_after', 0)} images retained"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = output_dir / "preprocessing_stats.json"

    stats_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
