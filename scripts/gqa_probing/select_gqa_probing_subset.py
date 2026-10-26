#!/usr/bin/env python3
"""Select a reproducible 3000/1000/1000 GQA probing subset without changing scene graphs.

The new train and validation come only from the original GQA train split;
the new test comes only from the original GQA validation split. A deterministic
quota-coverage stage favors rare labels; uniform random completion preserves
some scene diversity. Quotas apply to annotated INSTANCES, not images. Results
are a deliberately label-enriched subset, not a representative GQA test sample.
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


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def vocabularies(config_dir):
    result = {}
    for kind in KINDS:
        mapping = load_json(config_dir / f"{kind}_mapping_macro.json")
        labels = mapping["keep"]
        if len(labels) != EXPECTED[kind] or len(set(labels)) != len(labels):
            raise ValueError(f"Unexpected vocabulary for {kind}: {labels}")
        result[kind] = labels
    return result


def image_counts(graph, vocab):
    counts = {kind: Counter() for kind in KINDS}
    objects = graph["objects"]
    for obj in objects.values():
        label = obj["name"]
        if label not in vocab["object"]:
            raise ValueError(f"Unknown object label {label}")
        counts["object"][label] += 1
        # Attribute count refers to distinct objects carrying the macro-label.
        for attr in set(obj.get("attributes", [])):
            if attr not in vocab["attribute"]:
                raise ValueError(f"Unknown attribute label {attr}")
            counts["attribute"][attr] += 1
        for rel in obj.get("relations", []):
            name = rel["name"]
            if name not in vocab["relation"]:
                raise ValueError(f"Unknown relation label {name}")
            counts["relation"][name] += 1
    return counts


def build_index(graphs, vocab):
    indexed = {}
    index = defaultdict(list)
    for image_id, graph in graphs.items():
        counts = image_counts(graph, vocab)
        indexed[image_id] = counts
        for kind in KINDS:
            for label, n in counts[kind].items():
                if n > 0:
                    index[(kind, label)].append(image_id)
    return indexed, index


def count_pool(ids, indexed):
    sums = {kind: Counter() for kind in KINDS}
    for image_id in ids:
        for kind in KINDS:
            sums[kind].update(indexed[image_id][kind])
    return sums


def make_quotas(vocab, minima):
    return {kind: {label: minima[kind] for label in vocab[kind]} for kind in KINDS}


def select(ids, indexed, inverted, size, minimum, vocab, rng, candidate_count=200):
    """Pick quota-covering images first, then uniformly sample remaining images."""
    if size > len(ids):
        raise ValueError(f"Requested {size} from {len(ids)} available images")
    available = set(ids)
    selected = []
    counts = {kind: Counter() for kind in KINDS}
    quotas = make_quotas(vocab, minimum)
    universe = count_pool(available, indexed)
    infeasible = [(k, label, needed, universe[k][label])
                  for k in KINDS for label, needed in quotas[k].items()
                  if universe[k][label] < needed]
    if infeasible:
        raise ValueError(f"Insufficient available examples for quotas: {infeasible}")

    while len(selected) < size:
        deficits = {(kind, label): quota - counts[kind][label]
                    for kind, labels in quotas.items() for label, quota in labels.items()
                    if quota > counts[kind][label]}
        if not deficits:
            break
        # Target rare classes first; available counts are computed on full current pool.
        target = min(deficits, key=lambda k: (universe[k[0]][k[1]] / max(1, quotas[k[0]][k[1]]),
                                             k[0], k[1]))
        candidates = [image_id for image_id in inverted[target] if image_id in available]
        if not candidates:
            break
        if len(candidates) > candidate_count:
            candidates = rng.sample(candidates, candidate_count)
        # Weight every *undercovered* macro-class, discourage single-image dominance.
        def score(image_id):
            c = indexed[image_id]
            gain = 0.0
            for (kind, label), deficit in deficits.items():
                # Count improvements up to remaining quota. Capping prevents
                # a single dense image from dominating the selection.
                amount = min(deficit, c[kind][label], 12)
                if amount:
                    gain += amount / max(1, quotas[kind][label])
            return gain
        best = max(candidates, key=lambda img: (score(img), rng.random()))
        selected.append(best)
        available.remove(best)
        for kind in KINDS:
            counts[kind].update(indexed[best][kind])

    quota_stage_images = len(selected)
    remaining = size - len(selected)
    if remaining:
        # Do not optimize by label after quotas are met: random fill helps
        # preserve diversity and prevents selecting only heavily annotated scenes.
        selected.extend(rng.sample(sorted(available), remaining))
    rng.shuffle(selected)
    totals = count_pool(selected, indexed)
    unmet = {kind: {label: max(0, need - totals[kind][label])
                    for label, need in quotas[kind].items() if totals[kind][label] < need}
             for kind in KINDS}
    return selected, totals, unmet, quota_stage_images


def split_report(name, image_ids, counts, quotas, vocab, quota_stage):
    return {
        "images": len(image_ids),
        "images_selected_to_cover_quotas": quota_stage,
        "minima_per_label": quotas,
        "counts": {kind: {label: counts[kind][label] for label in vocab[kind]}
                   for kind in KINDS},
        "images_with_label": {},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=ROOT / "data/gqa_probing/annotations")
    p.add_argument("--config-dir", type=Path, default=Path(__file__).resolve().parent / "config")
    p.add_argument("--output-dir", type=Path, default=ROOT / "data/gqa_probing/subsets/probing_5k")
    p.add_argument("--train-size", type=int, default=3000)
    p.add_argument("--val-size", type=int, default=1000)
    p.add_argument("--test-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--object-min-train", type=int, default=100)
    p.add_argument("--object-min-eval", type=int, default=40)
    p.add_argument("--attribute-min-train", type=int, default=100)
    p.add_argument("--attribute-min-eval", type=int, default=40)
    p.add_argument("--relation-min-train", type=int, default=100)
    p.add_argument("--relation-min-eval", type=int, default=40)
    p.add_argument("--overwrite", action="store_true", help="Replace an existing selection explicitly")
    args = p.parse_args()
    if any(n <= 0 for n in (args.train_size, args.val_size, args.test_size)):
        p.error("Split sizes must be positive")
    if any(getattr(args, x) < 0 for x in (
        "object_min_train", "object_min_eval", "attribute_min_train", "attribute_min_eval",
        "relation_min_train", "relation_min_eval")):
        p.error("Minimum counts must be nonnegative")
    files = [args.output_dir / f"{split}_ids.json" for split in ("train", "val", "test")]
    files += [args.output_dir / "sampling_stats.json"]
    if not args.overwrite and any(file.exists() for file in files):
        p.error("Output files already exist. Use --overwrite to replace the frozen selection.")

    vocab = vocabularies(args.config_dir)
    train_graphs = load_json(args.input_dir / "train_sceneGraphs.json")
    test_graphs = load_json(args.input_dir / "val_sceneGraphs.json")
    if set(train_graphs) & set(test_graphs):
        raise ValueError("Original GQA train/val overlap: refusing to construct splits")
    if args.train_size + args.val_size > len(train_graphs) or args.test_size > len(test_graphs):
        raise ValueError("Not enough images in source split")
    train_data, train_index = build_index(train_graphs, vocab)
    test_data, test_index = build_index(test_graphs, vocab)
    minima_train = {"object": args.object_min_train,
                    "attribute": args.attribute_min_train, "relation": args.relation_min_train}
    minima_eval = {"object": args.object_min_eval,
                   "attribute": args.attribute_min_eval, "relation": args.relation_min_eval}
    # Select the new validation first, making rare classes available in holdout.
    val_ids, val_counts, val_unmet, val_stage = select(
        sorted(train_data), train_data, train_index, args.val_size, minima_eval,
        vocab, random.Random(args.seed + 1))
    rem = sorted(set(train_data) - set(val_ids))
    train_ids, train_counts, train_unmet, train_stage = select(
        rem, train_data, train_index, args.train_size, minima_train,
        vocab, random.Random(args.seed + 2))
    test_ids, test_counts, test_unmet, test_stage = select(
        sorted(test_data), test_data, test_index, args.test_size, minima_eval,
        vocab, random.Random(args.seed + 3))

    selected = {"train": train_ids, "val": val_ids, "test": test_ids}
    assert not (set(train_ids) & set(val_ids))
    assert not (set(train_ids) & set(test_ids))
    assert not (set(val_ids) & set(test_ids))
    stats = {
        "selection_method": "greedy per-label minimum-coverage then seeded uniform random completion",
        "seed": args.seed,
        "source_splits": {"train": "GQA train", "val": "GQA train", "test": "GQA val"},
        "counts_meaning": {"object": "object instances", "attribute": "distinct objects with an annotated macro-attribute", "relation": "directed relation annotations, including reciprocal left/right"},
        "warning": "This is a label-enriched sample, not an unbiased estimate of the full GQA distribution. Freeze test before model development.",
        "splits": {},
    }
    for name, ids, counts, unmet, stage in (
        ("train", train_ids, train_counts, train_unmet, train_stage),
        ("val", val_ids, val_counts, val_unmet, val_stage),
        ("test", test_ids, test_counts, test_unmet, test_stage),
    ):
        source = train_data if name != "test" else test_data
        report = split_report(name, ids, counts,
                              minima_train if name == "train" else minima_eval, vocab, stage)
        report["images_with_label"] = {
            kind: {label: sum(bool(source[i][kind][label]) for i in ids) for label in vocab[kind]}
            for kind in KINDS
        }
        report["unmet_quotas"] = {kind: vals for kind, vals in unmet.items() if vals}
        stats["splits"][name] = report
        print(f"{name}: {len(ids):,} images; quota-stage: {stage}; unmet: {report['unmet_quotas'] or 'none'}")
    for name, ids in selected.items():
        save_json(args.output_dir / f"{name}_ids.json", ids)
    save_json(args.output_dir / "sampling_stats.json", stats)
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
