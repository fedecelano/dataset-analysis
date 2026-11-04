#!/usr/bin/env python3
"""Select disjoint GQA probing subsets with macro- and original-label coverage.

Train and validation are drawn from prepared GQA train; test from prepared
GQA val. The unit of stratification is IMAGE PRESENCE (not annotation count).
Both canonical macro labels and their retained original labels contribute.

Selection is a seeded, heuristic multi-label stratification, not an exact
proportional allocation. Diagnostic JSON reports targets, counts, deviations,
and coverage that could not be achieved. Source graphs are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
KINDS = ("object", "attribute", "relation")
EXPECTED = {"object": 14, "attribute": 5, "relation": 7}
SPLITS = ("train", "val", "test")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def load_rules(config_dir: Path) -> dict:
    rules = {}
    for kind in KINDS:
        path = config_dir / f"{kind}_mapping_macro.json"
        mapping = load_json(path)
        if not isinstance(mapping, dict) or set(mapping) != {"keep", "merge", "remove"}:
            raise ValueError(f"Invalid mapping schema: {path}")
        keep, merge, remove = mapping["keep"], mapping["merge"], mapping["remove"]
        if not isinstance(keep, list) or not isinstance(merge, dict) or not isinstance(remove, list):
            raise ValueError(f"Invalid mapping types: {path}")
        values = [*keep, *merge.keys(), *merge.values(), *remove]
        if any(not isinstance(v, str) or not v for v in values):
            raise ValueError(f"Empty or non-string mapping label: {path}")
        if len(keep) != EXPECTED[kind] or len(set(keep)) != len(keep):
            raise ValueError(f"Expected {EXPECTED[kind]} unique {kind} labels: {path}")
        if len(set(remove)) != len(remove) or (set(keep) & set(remove)) or (set(keep) & set(merge)) or (set(remove) & set(merge)):
            raise ValueError(f"Conflicting or repeated actions: {path}")
        if not set(merge.values()) <= set(keep):
            raise ValueError(f"Merge targets not in KEEP vocabulary: {path}")
        rules[kind] = {"keep": set(keep), "merge": merge, "remove": set(remove), "labels": keep}
    return rules


def canonical(original: str, rule: dict) -> str | None:
    if not isinstance(original, str):
        return None
    if original in rule["keep"]:
        return original
    return rule["merge"].get(original)


def image_labels(image_id: str, graph: dict, rules: dict) -> set[tuple[str, str, str]]:
    """Labels present at least once in an image, with provenance validation."""
    if not isinstance(graph, dict) or not isinstance(graph.get("objects"), dict):
        raise ValueError(f"{image_id}: invalid graph/objects")
    objects = graph["objects"]
    result: set[tuple[str, str, str]] = set()
    for source_id, obj in objects.items():
        if not isinstance(obj, dict):
            raise ValueError(f"{image_id}/{source_id}: invalid object")
        macro = obj.get("name")
        original = obj.get("original_name")
        if macro not in rules["object"]["keep"] or canonical(original, rules["object"]) != macro:
            raise ValueError(f"{image_id}/{source_id}: inconsistent object provenance")
        result.add(("macro", "object", macro))
        result.add(("original", "object", original))

        attrs = obj.get("attributes")
        original_attrs = obj.get("original_attributes")
        if not isinstance(attrs, list) or not isinstance(original_attrs, list):
            raise ValueError(f"{image_id}/{source_id}: missing attribute provenance")
        mapped_attrs = {mapped for raw in original_attrs if (mapped := canonical(raw, rules["attribute"])) is not None}
        if len(attrs) != len(set(attrs)) or set(attrs) != mapped_attrs:
            raise ValueError(f"{image_id}/{source_id}: inconsistent/doubled macro attributes")
        for attr in attrs:
            result.add(("macro", "attribute", attr))
        for raw in original_attrs:
            if canonical(raw, rules["attribute"]) is not None:
                result.add(("original", "attribute", raw))

        rels = obj.get("relations")
        if not isinstance(rels, list):
            raise ValueError(f"{image_id}/{source_id}: invalid relations")
        seen_triples = set()
        for rel in rels:
            if not isinstance(rel, dict):
                raise ValueError(f"{image_id}/{source_id}: malformed relation")
            name, target = rel.get("name"), str(rel.get("object"))
            if name not in rules["relation"]["keep"] or target not in objects:
                raise ValueError(f"{image_id}/{source_id}: invalid macro relation/target")
            key = (name, target)
            if key in seen_triples:
                raise ValueError(f"{image_id}/{source_id}: duplicate canonical relation {key}")
            seen_triples.add(key)
            originals = rel.get("original_names")
            if not isinstance(originals, list) or not originals or any(canonical(raw, rules["relation"]) != name for raw in originals):
                raise ValueError(f"{image_id}/{source_id}: invalid relation provenance")
            if rel.get("original_name") != originals[0]:
                raise ValueError(f"{image_id}/{source_id}: inconsistent original_name")
            result.add(("macro", "relation", name))
            for raw in originals:
                result.add(("original", "relation", raw))
    return result


def build_index(graphs: dict, rules: dict) -> tuple[dict[str, set], dict[tuple, list[str]]]:
    by_image = {}
    inverted = defaultdict(list)
    for image_id in sorted(graphs):
        labels = image_labels(image_id, graphs[image_id], rules)
        by_image[image_id] = labels
        for label in sorted(labels):
            inverted[label].append(image_id)
    return by_image, dict(inverted)


def support_counts(ids: list[str], by_image: dict) -> Counter:
    counts = Counter()
    for image_id in ids:
        counts.update(by_image[image_id])
    return counts


def compute_targets(pool_ids: list[str], pool_counts: Counter, size: int,
                    macro_min: dict[str, int], original_min: dict[str, int], rules: dict) -> dict:
    """Desired image-level counts, combining proportions and coverage floors.

    Floors are *objectives*, not promises. For labels with < floor source
    support, targets are capped to available support and flagged in reports.
    """
    targets = {}
    keys = set(pool_counts)
    keys.update(("macro", kind, label) for kind in KINDS for label in rules[kind]["labels"])
    for label in sorted(keys):
        level, kind, _ = label
        support = pool_counts[label]
        if not support:
            targets[label] = 0
            continue
        expected = round(size * support / len(pool_ids))
        floor = macro_min[kind] if level == "macro" else original_min[kind]
        targets[label] = min(support, size, max(expected, floor))
    return targets


def select_subset(pool_ids: list[str], by_image: dict, inverted: dict, size: int,
                  macro_min: dict, original_min: dict, rules: dict,
                  rng: random.Random, candidate_count: int) -> tuple[list[str], dict]:
    if size > len(pool_ids):
        raise ValueError(f"Requested {size} images from a pool of {len(pool_ids)}")
    pool_counts = support_counts(pool_ids, by_image)
    targets = compute_targets(pool_ids, pool_counts, size, macro_min, original_min, rules)
    remaining = set(pool_ids)
    selected = []
    counts = Counter()
    stratified_picks = 0

    # Prioritize undercovered rare labels while crediting co-occurring labels.
    while len(selected) < size:
        deficits = {key: target - counts[key] for key, target in targets.items()
                    if target > counts[key]}
        if not deficits:
            break
        # Relative shortfall is balanced against scarcity in the current pool.
        priorities = sorted(deficits, key=lambda key: (
            pool_counts[key] / max(1, targets[key]),
            -deficits[key] / max(1, targets[key]), key))
        target_label = next((key for key in priorities
                             if any(image_id in remaining for image_id in inverted.get(key, ()))), None)
        if target_label is None:
            break
        candidates = [i for i in inverted[target_label] if i in remaining]
        if len(candidates) > candidate_count:
            candidates = rng.sample(candidates, candidate_count)

        def score(image_id: str) -> float:
            gain = 0.0
            for key in by_image[image_id]:
                if key in deficits:
                    # Large relative deficit and rare labels have more value.
                    gain += (deficits[key] / max(1, targets[key])) / (
                        max(1, pool_counts[key]) ** 0.25)
            return gain

        # Stable ordering + seeded tie breaking.
        scores = [(score(i), rng.random(), i) for i in sorted(candidates)]
        chosen = max(scores, key=lambda row: (row[0], row[1]))[2]
        selected.append(chosen)
        remaining.remove(chosen)
        counts.update(by_image[chosen])
        stratified_picks += 1

    # No more target-coverage gains are available, or all goals were met.
    if len(selected) < size:
        extra = rng.sample(sorted(remaining), size - len(selected))
        selected.extend(extra)
        counts.update(support_counts(extra, by_image))

    rng.shuffle(selected)
    per_label = []
    for (level, kind, label), target in sorted(targets.items()):
        source = pool_counts[(level, kind, label)]
        observed = counts[(level, kind, label)]
        per_label.append({
            "level": level, "kind": kind, "label": label,
            "pool_images": source, "target_images": target,
            "selected_images": observed, "shortfall": max(0, target - observed),
            "pool_prevalence_pct": round(100 * source / len(pool_ids), 4),
            "selected_prevalence_pct": round(100 * observed / size, 4),
        })
    return selected, {
        "images": size,
        "candidate_pool_images": len(pool_ids),
        "stratification_stage_images": stratified_picks,
        "labels": per_label,
        "targets_unmet": sum(row["shortfall"] > 0 for row in per_label),
        "unmet_macro_labels": [row for row in per_label if row["level"] == "macro" and row["shortfall"]],
        "unmet_original_labels": [row for row in per_label if row["level"] == "original" and row["shortfall"]],
        "labels_below_requested_minimum_in_pool": [
            row for row in per_label if row["pool_images"] < (
                macro_min[row["kind"]] if row["level"] == "macro" else original_min[row["kind"]])
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=ROOT / "data/gqa_probing/annotations")
    p.add_argument("--config-dir", type=Path, default=Path(__file__).resolve().parent / "config")
    p.add_argument("--output-dir", type=Path, default=ROOT / "data/gqa_probing/subsets/probing_5k")
    p.add_argument("--train-size", type=int, default=3000)
    p.add_argument("--val-size", type=int, default=1000)
    p.add_argument("--test-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    for kind in KINDS:
        p.add_argument(f"--{kind}-min-train", type=int, default=100)
        p.add_argument(f"--{kind}-min-eval", type=int, default=40)
        p.add_argument(f"--original-{kind}-min-train", type=int, default=5)
        p.add_argument(f"--original-{kind}-min-eval", type=int, default=2)
    p.add_argument("--candidate-count", type=int, default=200)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if min(args.train_size, args.val_size, args.test_size, args.candidate_count) < 1:
        p.error("Split sizes and candidate count must be positive")
    if any(getattr(args, flag) < 0 for kind in KINDS for flag in (
        f"{kind}_min_train", f"{kind}_min_eval",
        f"original_{kind}_min_train", f"original_{kind}_min_eval")):
        p.error("Minimum coverage targets must be nonnegative")

    outputs = [args.output_dir / f"{name}_ids.json" for name in SPLITS]
    outputs += [args.output_dir / "sampling_stats.json"]
    if not args.overwrite and any(path.exists() for path in outputs):
        p.error("Selection already exists. Use --overwrite only to replace it intentionally.")

    rules = load_rules(args.config_dir)
    train_graphs = load_json(args.input_dir / "train_sceneGraphs.json")
    val_graphs = load_json(args.input_dir / "val_sceneGraphs.json")
    if not isinstance(train_graphs, dict) or not isinstance(val_graphs, dict):
        raise ValueError("Expected dictionaries of image scene graphs")
    if set(train_graphs) & set(val_graphs):
        raise ValueError("Source GQA train/val image-ID overlap")
    if args.train_size + args.val_size > len(train_graphs) or args.test_size > len(val_graphs):
        raise ValueError("Not enough source images for requested split sizes")

    print("Indexing macro and original image-level labels...")
    train_data, train_index = build_index(train_graphs, rules)
    val_data, val_index = build_index(val_graphs, rules)
    del train_graphs, val_graphs

    macro_train = {kind: getattr(args, f"{kind}_min_train") for kind in KINDS}
    macro_eval = {kind: getattr(args, f"{kind}_min_eval") for kind in KINDS}
    orig_train = {kind: getattr(args, f"original_{kind}_min_train") for kind in KINDS}
    orig_eval = {kind: getattr(args, f"original_{kind}_min_eval") for kind in KINDS}

    val_ids, val_stats = select_subset(
        sorted(train_data), train_data, train_index, args.val_size,
        macro_eval, orig_eval, rules, random.Random(args.seed + 1), args.candidate_count)
    remaining = sorted(set(train_data) - set(val_ids))
    train_ids, train_stats = select_subset(
        remaining, train_data, train_index, args.train_size,
        macro_train, orig_train, rules, random.Random(args.seed + 2), args.candidate_count)
    test_ids, test_stats = select_subset(
        sorted(val_data), val_data, val_index, args.test_size,
        macro_eval, orig_eval, rules, random.Random(args.seed + 3), args.candidate_count)

    selections = {"train": train_ids, "val": val_ids, "test": test_ids}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(selections[a]) & set(selections[b]):
            raise ValueError(f"Unexpected overlap: {a} / {b}")

    stats = {
        "selection_method": "seeded greedy multi-label image-presence targets, followed by uniform completion",
        "warning": "Heuristic, label-aware selection; not a representative random sample or guaranteed exact stratification",
        "seed": args.seed,
        "source_splits": {"train": "GQA train", "val": "GQA train", "test": "GQA val"},
        "unit_of_stratification": "number of distinct images containing a label",
        "macro_minimum_targets": {"train": macro_train, "eval": macro_eval},
        "original_minimum_targets": {"train": orig_train, "eval": orig_eval},
        "splits": {"train": train_stats, "val": val_stats, "test": test_stats},
    }
    for name in SPLITS:
        print(f"{name}: {len(selections[name])} images; "
              f"unmet targets: {stats['splits'][name]['targets_unmet']}")
    for name in SPLITS:
        save_json(args.output_dir / f"{name}_ids.json", selections[name])
    save_json(args.output_dir / "sampling_stats.json", stats)
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
