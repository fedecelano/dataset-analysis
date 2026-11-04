#!/usr/bin/env python3
"""Analyze directed object–relation–object associations in frozen GQA probing_5k.

Uses retained, deduplicated, directed edges from prepared scene graphs; does
not infer absent edges or create inverse edges. Computes edge-conditioned
probabilities and lift for canonical relations with macro or original object
endpoints. All reported image supports count distinct images. This describes
annotation associations, NOT model bias or visual ground-truth negatives.

Run from repository root:
    python scripts/gqa_bias_analysis/analyze_relational_bias.py
    python scripts/gqa_bias_analysis/analyze_relational_bias.py --min-triple-edges 30
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test")
LEVELS = ("macro_object", "original_object")
RELATIONS = ("to the left of", "to the right of")

TRIPLE_FIELDS = [
    "split", "level", "subject", "relation", "object", "triple_edges",
    "triple_images", "subject_object_edges", "subject_edges", "object_edges",
    "relation_edges", "total_edges", "total_images", "p_relation_given_pair_pct",
    "p_relation_given_subject_pct", "p_relation_given_object_pct",
    "p_relation_marginal_pct", "lift_relation_given_pair",
    "lift_relation_given_subject", "lift_relation_given_object",
    "triple_image_prevalence_pct",
]
DIAG_FIELDS = [
    "split", "relation", "edges", "images", "matched_inverse_edges",
    "unmatched_inverse_edges", "matched_inverse_fraction_pct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-dir", type=Path,
        default=ROOT / "data/gqa_probing/subsets/probing_5k/annotations",
        help="Directory with train/val/test_sceneGraphs.json.",
    )
    parser.add_argument(
        "--config-dir", type=Path,
        default=ROOT / "scripts/gqa_probing/config",
        help="Directory with object_mapping_macro.json and relation_mapping_macro.json.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs/gqa_bias_analysis/probing_5k/tables/relational",
    )
    parser.add_argument("--min-triple-edges", type=int, default=20)
    parser.add_argument("--min-triple-images", type=int, default=10)
    parser.add_argument("--min-pair-edges", type=int, default=50)
    args = parser.parse_args()
    if min(args.min_triple_edges, args.min_triple_images, args.min_pair_edges) < 1:
        parser.error("All minimum-support thresholds must be positive integers.")
    return args


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON dictionary")
    return data


def load_mapping(path: Path, expected: int) -> dict:
    mapping = read_json(path)
    if set(mapping) != {"keep", "merge", "remove"}:
        raise ValueError(f"{path}: expected keep, merge, remove")
    keep, merge, remove = (mapping[key] for key in ("keep", "merge", "remove"))
    if not isinstance(keep, list) or not isinstance(merge, dict) or not isinstance(remove, list):
        raise ValueError(f"{path}: invalid mapping types")
    values = [*keep, *merge.keys(), *merge.values(), *remove]
    if (len(keep) != expected or len(set(keep)) != expected
            or any(not isinstance(x, str) or not x for x in values)
            or set(keep) & set(merge) or set(keep) & set(remove)
            or set(merge) & set(remove) or not set(merge.values()) <= set(keep)):
        raise ValueError(f"{path}: invalid or conflicting mapping")
    return {"keep": set(keep), "merge": merge}


def mapped(original: str, canonical: str, mapping: dict) -> bool:
    return (original == canonical and original in mapping["keep"]) or mapping["merge"].get(original) == canonical


def pct(n: int, d: int) -> float | None:
    return round(100.0 * n / d, 6) if d else None


def ratio(n: int, d: int) -> float | None:
    return round(n / d, 6) if d else None


def new_level() -> dict:
    return {
        "triple": Counter(), "triple_images": Counter(),
        "pair": Counter(), "subject": Counter(), "object": Counter(),
        "subject_relation": Counter(), "object_relation": Counter(),
    }


def analyze_split(path: Path, mappings: dict) -> dict:
    graphs = read_json(path)
    stats = {
        "images": len(graphs), "objects": 0, "edges": 0,
        "relation": Counter(), "relation_images": Counter(),
        "levels": {level: new_level() for level in LEVELS},
        "directional": {rel: {"edges": 0, "images": 0, "matched": 0} for rel in RELATIONS},
    }
    for image_id, graph in graphs.items():
        objects = graph.get("objects") if isinstance(graph, dict) else None
        if not isinstance(objects, dict) or not objects:
            raise ValueError(f"{path}: invalid or empty graph for {image_id}")
        for oid, obj in objects.items():
            where = f"{path}: {image_id}/{oid}"
            if not isinstance(obj, dict):
                raise ValueError(f"{where}: invalid object")
            canon, original = obj.get("name"), obj.get("original_name")
            if not isinstance(canon, str) or not isinstance(original, str) or not mapped(original, canon, mappings["object"]):
                raise ValueError(f"{where}: missing or invalid original/canonical object name")
        stats["objects"] += len(objects)
        image_triples = {level: set() for level in LEVELS}
        image_relations: set[str] = set()
        edge_set: set[tuple[str, str, str]] = set()
        for source_id, source in objects.items():
            rels = source.get("relations")
            where = f"{path}: {image_id}/{source_id}"
            if not isinstance(rels, list):
                raise ValueError(f"{where}: invalid relations")
            for rel_record in rels:
                if not isinstance(rel_record, dict):
                    raise ValueError(f"{where}: invalid relation record")
                relation = rel_record.get("name")
                target_id = str(rel_record.get("object"))
                if relation not in mappings["relation"]["keep"] or target_id not in objects:
                    raise ValueError(f"{where}: invalid relation or target")
                originals = rel_record.get("original_names")
                if (not isinstance(originals, list) or not originals
                        or any(not isinstance(x, str) or not mapped(x, relation, mappings["relation"]) for x in originals)
                        or rel_record.get("original_name") != originals[0]):
                    raise ValueError(f"{where}: invalid relation provenance")
                edge = (str(source_id), relation, target_id)
                if edge in edge_set:
                    raise ValueError(f"{where}: duplicate directed canonical edge")
                edge_set.add(edge)
                stats["edges"] += 1
                stats["relation"][relation] += 1
                image_relations.add(relation)
                target = objects[target_id]
                for level, source_key, target_key in (
                    ("macro_object", "name", "name"),
                    ("original_object", "original_name", "original_name"),
                ):
                    s, o = source[source_key], target[target_key]
                    current = stats["levels"][level]
                    triple = (s, relation, o)
                    current["triple"][triple] += 1
                    current["pair"][(s, o)] += 1
                    current["subject"][s] += 1
                    current["object"][o] += 1
                    current["subject_relation"][(s, relation)] += 1
                    current["object_relation"][(o, relation)] += 1
                    image_triples[level].add(triple)
        for level in LEVELS:
            stats["levels"][level]["triple_images"].update(image_triples[level])
        stats["relation_images"].update(image_relations)
        for relation in RELATIONS:
            directional_edges = [edge for edge in edge_set if edge[1] == relation]
            inverse = RELATIONS[1] if relation == RELATIONS[0] else RELATIONS[0]
            entry = stats["directional"][relation]
            entry["edges"] += len(directional_edges)
            entry["images"] += bool(directional_edges)
            entry["matched"] += sum((t, inverse, s) in edge_set for s, _, t in directional_edges)
    return stats


def build_rows(split: str, level: str, stats: dict) -> list[dict]:
    result = []
    data = stats["levels"][level]
    for (subject, relation, obj), count in sorted(data["triple"].items()):
        pair_edges = data["pair"][(subject, obj)]
        sub_edges = data["subject"][subject]
        obj_edges = data["object"][obj]
        relation_edges = stats["relation"][relation]
        # All denominators are directed, retained annotation EDGES, not all
        # possible object-instance pairs (unannotated pairs are unknown).
        p_pair = count / pair_edges
        p_sub = data["subject_relation"][(subject, relation)] / sub_edges
        p_obj = data["object_relation"][(obj, relation)] / obj_edges
        p_rel = relation_edges / stats["edges"]
        triple_images = data["triple_images"][(subject, relation, obj)]
        result.append({
            "split": split, "level": level, "subject": subject,
            "relation": relation, "object": obj,
            "triple_edges": count, "triple_images": triple_images,
            "subject_object_edges": pair_edges, "subject_edges": sub_edges,
            "object_edges": obj_edges, "relation_edges": relation_edges,
            "total_edges": stats["edges"], "total_images": stats["images"],
            "p_relation_given_pair_pct": round(p_pair * 100, 6),
            "p_relation_given_subject_pct": round(p_sub * 100, 6),
            "p_relation_given_object_pct": round(p_obj * 100, 6),
            "p_relation_marginal_pct": round(p_rel * 100, 6),
            "lift_relation_given_pair": round(p_pair / p_rel, 6),
            "lift_relation_given_subject": round(p_sub / p_rel, 6),
            "lift_relation_given_object": round(p_obj / p_rel, 6),
            "triple_image_prevalence_pct": pct(triple_images, stats["images"]),
        })
    return result


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def main() -> None:
    args = parse_args()
    mappings = {
        kind: load_mapping(args.config_dir / f"{kind}_mapping_macro.json", expected)
        for kind, expected in (("object", 14), ("relation", 7))
    }
    rows: list[dict] = []
    diagnostics: list[dict] = []
    summary = {
        "methodology": {
            "dataset": "GQA probing_5k prepared scene graphs",
            "unit_for_probabilities": "directed retained canonical annotation edges",
            "unit_for_support": "distinct images and retained directed annotation edges",
            "triple": "(source object label, canonical directed relation, target object label)",
            "levels": list(LEVELS),
            "original_object": "original endpoint labels; canonical relation names",
            "conditionals": "P(relation|subject,object), P(relation|subject), P(relation|object) among annotated edges",
            "lift": "conditional relation probability divided by relation probability among all annotated edges",
            "unannotated_pairs": "unknown; not counted as negative relation examples",
            "inverse_diagnostic": "left/right inverse edge is counted only when explicitly present; never inferred",
            "rankings": "observed triples filtered by support; high lift does not imply causal bias",
            "sampling": "label-enriched frozen subset; not an unbiased GQA sample",
        },
        "ranking_thresholds": {
            "min_triple_edges": args.min_triple_edges,
            "min_triple_images": args.min_triple_images,
            "min_pair_edges": args.min_pair_edges,
        },
        "splits": {},
    }
    for split in SPLITS:
        path = args.annotations_dir / f"{split}_sceneGraphs.json"
        print(f"Analyzing {split}: {path}")
        stats = analyze_split(path, mappings)
        for level in LEVELS:
            rows.extend(build_rows(split, level, stats))
        for relation in RELATIONS:
            d = stats["directional"][relation]
            diagnostics.append({
                "split": split, "relation": relation,
                "edges": d["edges"], "images": d["images"],
                "matched_inverse_edges": d["matched"],
                "unmatched_inverse_edges": d["edges"] - d["matched"],
                "matched_inverse_fraction_pct": pct(d["matched"], d["edges"]),
            })
        summary["splits"][split] = {
            "images": stats["images"], "objects": stats["objects"],
            "directed_edges": stats["edges"],
            "relation_edges": dict(sorted(stats["relation"].items())),
            "observed_triples": {
                level: len(stats["levels"][level]["triple"])
                for level in LEVELS
            },
            "left_right_inverse_edge_match": {
                relation: {
                    "edges": stats["directional"][relation]["edges"],
                    "matched_inverse_edges": stats["directional"][relation]["matched"],
                }
                for relation in RELATIONS
            },
        }
        print(f"  images={stats['images']:,}, objects={stats['objects']:,}, edges={stats['edges']:,}")

    out = args.output_dir
    write_csv(out / "relation_triples.csv", TRIPLE_FIELDS, rows)
    ranked = [
        row for row in rows
        if row["triple_edges"] >= args.min_triple_edges
        and row["triple_images"] >= args.min_triple_images
        and row["subject_object_edges"] >= args.min_pair_edges
    ]
    ranked.sort(key=lambda r: (
        r["split"], r["level"], -r["lift_relation_given_pair"],
        -r["triple_images"], -r["triple_edges"], r["subject"], r["relation"], r["object"],
    ))
    write_csv(out / "ranked_relation_triples.csv", TRIPLE_FIELDS, ranked)
    write_csv(out / "relation_directional_diagnostics.csv", DIAG_FIELDS, diagnostics)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "relational_bias_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Saved: {out / 'relational_bias_summary.json'}")


if __name__ == "__main__":
    main()
