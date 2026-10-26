#!/usr/bin/env python3
"""Check reciprocal left/right GQA relationships at *object-instance* level.

Read-only: scans prepared GQA probing train/val scene graphs and writes a JSON
report. For every directed A --left--> B, looks for B --right--> A (and vice
versa) in the SAME image. Counts repeated edges separately and reports a few
non-reciprocal examples for inspection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

LEFT = "to the left of"
RIGHT = "to the right of"


def inspect_graphs(path: Path, max_examples: int = 15) -> dict:
    with path.open(encoding="utf-8") as handle:
        graphs = json.load(handle)
    if not isinstance(graphs, dict):
        raise ValueError(f"Expected image-ID -> graph dictionary in {path}")

    tally = Counter()
    examples = []
    for image_id, graph in graphs.items():
        objects = graph.get("objects")
        if not isinstance(objects, dict):
            raise ValueError(f"Invalid objects in image {image_id}")
        tally["images_checked"] += 1
        # Directed edges counted independently. Only left/right are needed.
        edges = Counter()
        for source_id, obj in objects.items():
            relations = obj.get("relations", [])
            if not isinstance(relations, list):
                raise ValueError(f"Invalid relations at {image_id}/{source_id}")
            for rel in relations:
                name = rel.get("name")
                if name not in (LEFT, RIGHT):
                    continue
                target_id = str(rel.get("object"))
                if target_id not in objects:
                    raise ValueError(f"Missing target {target_id} at {image_id}/{source_id}")
                edges[(str(source_id), name, target_id)] += 1
                tally["left_annotations" if name == LEFT else "right_annotations"] += 1
                if str(source_id) == target_id:
                    tally["self_relations"] += 1
        if edges:
            tally["images_with_directional_relations"] += 1
        # Match each forward edge to reverse edge of opposite label. An
        # exact reciprocal pair is visited twice below, once per direction.
        for (src, label, dst), n in edges.items():
            inverse = RIGHT if label == LEFT else LEFT
            reciprocal_n = edges.get((dst, inverse, src), 0)
            matched = min(n, reciprocal_n)
            missing = n - matched
            prefix = "left" if label == LEFT else "right"
            tally[f"{prefix}_matched_annotations"] += matched
            tally[f"{prefix}_unmatched_annotations"] += missing
            if n > 1:
                tally[f"{prefix}_repeated_annotations"] += n - 1
                tally[f"{prefix}_repeated_edges"] += 1
            if missing and len(examples) < max_examples:
                examples.append({
                    "image_id": str(image_id),
                    "source_id": src,
                    "source_class": objects[src].get("name"),
                    "relation": label,
                    "target_id": dst,
                    "target_class": objects[dst].get("name"),
                    "count": n,
                    "inverse_count": reciprocal_n,
                    "unmatched_count": missing,
                })
    left = tally["left_annotations"]
    right = tally["right_annotations"]
    matched_left = tally["left_matched_annotations"]
    matched_right = tally["right_matched_annotations"]
    result = dict(sorted(tally.items()))
    for key in (
        "left_annotations", "right_annotations", "left_matched_annotations",
        "right_matched_annotations", "left_unmatched_annotations",
        "right_unmatched_annotations", "left_repeated_annotations",
        "right_repeated_annotations", "self_relations",
    ):
        result.setdefault(key, 0)
    result["left_reciprocity_pct"] = round(100 * matched_left / left, 4) if left else None
    result["right_reciprocity_pct"] = round(100 * matched_right / right, 4) if right else None
    result["reciprocal_pair_count"] = matched_left  # one pair has two directed annotations
    result["unmatched_examples"] = examples
    if matched_left != matched_right:
        raise AssertionError("Matched left/right annotation counts must agree")
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root / "data/gqa_probing/annotations")
    parser.add_argument("--output", type=Path, default=root / "outputs/gqa_probing/tables/inverse_relations_report.json")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--max-examples", type=int, default=15)
    args = parser.parse_args()
    if args.max_examples < 0:
        parser.error("--max-examples must be nonnegative")
    report = {}
    for split in args.splits:
        if not split.replace("_", "").isalnum():
            parser.error(f"Invalid split {split!r}")
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
