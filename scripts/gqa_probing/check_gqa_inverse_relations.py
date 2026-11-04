#!/usr/bin/env python3
"""Audit instance-level reciprocity of GQA left/right relations.

For each A --to the left of--> B, check whether the same image contains
B --to the right of--> A, and vice versa. Count directed annotations,
repeated annotations, and reciprocal pairs; report unmatched examples.

This is a *descriptive* audit: missing inverse annotations are not assumed
to be errors, and the script does not add or remove relations.

Example (from the repository root):
    python scripts/gqa_probing/check_gqa_inverse_relations.py

Inputs: data/gqa_probing/annotations/{train,val}_sceneGraphs.json
Output: outputs/gqa_probing/tables/inverse_relations_report.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

LEFT = "to the left of"
RIGHT = "to the right of"
INVERSE = {LEFT: RIGHT, RIGHT: LEFT}


def inspect_graphs(path: Path, max_examples: int = 15) -> dict:
    """Count directed left/right annotations and reciprocal instance pairs.

    Repeated identical annotations are counted individually. A reciprocal
    match is limited by the smaller multiplicity of the two directions.
    """
    if max_examples < 0:
        raise ValueError("max_examples must be nonnegative")

    with path.open("r", encoding="utf-8") as handle:
        graphs = json.load(handle)
    if not isinstance(graphs, dict):
        raise ValueError(f"Expected image-ID-to-graph dictionary: {path}")

    tally = Counter()
    examples = []

    for image_id, graph in graphs.items():
        if not isinstance(graph, dict) or not isinstance(graph.get("objects"), dict):
            raise ValueError(f"Invalid objects dictionary in image {image_id}")
        objects = graph["objects"]
        tally["images_checked"] += 1
        edges = Counter()
        provenance = {}

        for source_id, obj in objects.items():
            if not isinstance(obj, dict) or not isinstance(obj.get("relations", []), list):
                raise ValueError(f"Invalid object/relations at {image_id}/{source_id}")
            for rel in obj.get("relations", []):
                if not isinstance(rel, dict):
                    raise ValueError(f"Invalid relation at {image_id}/{source_id}")
                label = rel.get("name")
                if label not in INVERSE:
                    continue
                if "object" not in rel:
                    raise ValueError(f"Missing target at {image_id}/{source_id}")
                target_id = str(rel["object"])
                if target_id not in objects:
                    raise ValueError(
                        f"Missing target {target_id!r} at {image_id}/{source_id}"
                    )
                key = (str(source_id), label, target_id)
                edges[key] += 1
                provenance.setdefault(key, rel.get("original_names", [rel.get("original_name", label)]))
                tally["left_annotations" if label == LEFT else "right_annotations"] += 1
                if str(source_id) == target_id:
                    tally["self_relations"] += 1

        if edges:
            tally["images_with_directional_relations"] += 1
        tally["unique_directional_edges"] += len(edges)

        for (src, label, dst), count in edges.items():
            inverse_count = edges.get((dst, INVERSE[label], src), 0)
            matched = min(count, inverse_count)
            missing = count - matched
            prefix = "left" if label == LEFT else "right"
            tally[f"{prefix}_matched_annotations"] += matched
            tally[f"{prefix}_unmatched_annotations"] += missing
            tally[f"{prefix}_unique_edges"] += 1
            if inverse_count:
                tally[f"{prefix}_edges_with_inverse"] += 1
            if count > 1:
                tally[f"{prefix}_repeated_annotations"] += count - 1
                tally[f"{prefix}_repeated_edges"] += 1
            if missing and len(examples) < max_examples:
                original_labels = provenance[key]
                if isinstance(original_labels, str):
                    original_labels = [original_labels]
                examples.append({
                    "image_id": str(image_id),
                    "source_id": src,
                    "source_class": objects[src].get("name"),
                    "source_original_class": objects[src].get("original_name"),
                    "relation": label,
                    "relation_original_names": original_labels,
                    "target_id": dst,
                    "target_class": objects[dst].get("name"),
                    "target_original_class": objects[dst].get("original_name"),
                    "count": count,
                    "inverse_count": inverse_count,
                    "unmatched_count": missing,
                })

    result = dict(sorted(tally.items()))
    for key in (
        "images_checked", "images_with_directional_relations",
        "left_annotations", "right_annotations",
        "left_matched_annotations", "right_matched_annotations",
        "left_unmatched_annotations", "right_unmatched_annotations",
        "left_repeated_annotations", "right_repeated_annotations",
        "left_repeated_edges", "right_repeated_edges",
        "left_unique_edges", "right_unique_edges",
        "left_edges_with_inverse", "right_edges_with_inverse",
        "unique_directional_edges", "self_relations",
    ):
        result.setdefault(key, 0)

    if result["left_matched_annotations"] != result["right_matched_annotations"]:
        raise AssertionError("Matched left/right counts must agree")
    for prefix in ("left", "right"):
        if (result[f"{prefix}_matched_annotations"]
                + result[f"{prefix}_unmatched_annotations"]
                != result[f"{prefix}_annotations"]):
            raise AssertionError(f"Inconsistent {prefix} annotation counts")

    for prefix in ("left", "right"):
        annotations = result[f"{prefix}_annotations"]
        unique_edges = result[f"{prefix}_unique_edges"]
        result[f"{prefix}_reciprocity_pct"] = (
            round(100 * result[f"{prefix}_matched_annotations"] / annotations, 4)
            if annotations else None
        )
        result[f"{prefix}_unique_edge_reciprocity_pct"] = (
            round(100 * result[f"{prefix}_edges_with_inverse"] / unique_edges, 4)
            if unique_edges else None
        )

    result["reciprocal_pair_count"] = result["left_matched_annotations"]
    result["unmatched_examples"] = examples
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=root / "data" / "gqa_probing" / "annotations",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs" / "gqa_probing" / "tables" / "inverse_relations_report.json",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--max-examples", type=int, default=15)
    args = parser.parse_args()

    if args.max_examples < 0:
        parser.error("--max-examples must be nonnegative")
    if len(args.splits) != len(set(args.splits)):
        parser.error("Duplicate splits requested")
    for split in args.splits:
        if not split or not split.replace("_", "").isalnum():
            parser.error(f"Invalid split {split!r}")

    report = {}
    for split in args.splits:
        source = args.input_dir / f"{split}_sceneGraphs.json"
        if not source.is_file():
            parser.error(f"Input file not found: {source}")
        stats = inspect_graphs(source, args.max_examples)
        report[split] = stats
        print(
            f"{split}: left={stats['left_annotations']:,} "
            f"(reciprocal {stats['left_reciprocity_pct']}%), "
            f"right={stats['right_annotations']:,} "
            f"(reciprocal {stats['right_reciprocity_pct']}%), "
            f"unmatched={stats['left_unmatched_annotations'] + stats['right_unmatched_annotations']:,}, "
            f"repeated={stats['left_repeated_annotations'] + stats['right_repeated_annotations']:,}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
