#!/usr/bin/env python3
"""Analyze GQA probing macro labels and optionally compare the frozen 5k subset.

Default mode preserves the four historical train/val frequency CSV outputs.
With --compare-subset, compare subset train/val with prepared GQA train, and
subset test with prepared GQA val. Image-level prevalence, occurrence-based
TVD/Jensen-Shannon distance, label coverage and class-triple coverage are
reported separately. No source annotations or selections are modified.

Run from repository root:
    python scripts/gqa_probing/analyze_gqa_probing.py
    python scripts/gqa_probing/analyze_gqa_probing.py --compare-subset
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CLASSES = {"object": 14, "attribute": 5, "relation": 7}
KINDS = tuple(EXPECTED_CLASSES)
SPLITS = ("train", "val")
SUBSET_SPLITS = ("train", "val", "test")
SOURCE_FOR_SUBSET = {"train": "train", "val": "train", "test": "val"}
EXPECTED_SUBSET_SIZES = {"train": 3000, "val": 1000, "test": 1000}


def load_rules(config_dir: Path) -> dict:
    """Validate mapping configuration and retain canonical label order."""
    all_rules = {}
    for kind, expected_count in EXPECTED_CLASSES.items():
        path = config_dir / f"{kind}_mapping_macro.json"
        with path.open(encoding="utf-8") as handle:
            mapping = json.load(handle)
        if not isinstance(mapping, dict) or set(mapping) != {"keep", "merge", "remove"}:
            raise ValueError(f"{path}: expected keep, merge and remove keys")
        keep, merge, remove = (mapping[key] for key in ("keep", "merge", "remove"))
        if not isinstance(keep, list) or not isinstance(merge, dict) or not isinstance(remove, list):
            raise ValueError(f"{path}: invalid mapping types")
        labels = [*keep, *merge.keys(), *merge.values(), *remove]
        if not all(isinstance(label, str) and bool(label) for label in labels):
            raise ValueError(f"{path}: labels must be nonempty strings")
        if len(keep) != expected_count or len(set(keep)) != expected_count:
            raise ValueError(f"{path}: expected {expected_count} distinct canonical labels")
        if len(remove) != len(set(remove)):
            raise ValueError(f"{path}: duplicate REMOVE labels")
        keep_set, merge_set, remove_set = set(keep), set(merge), set(remove)
        if keep_set & merge_set or keep_set & remove_set or merge_set & remove_set:
            raise ValueError(f"{path}: conflicting mapping actions")
        if not set(merge.values()) <= keep_set:
            raise ValueError(f"{path}: MERGE target outside canonical vocabulary")
        all_rules[kind] = {"labels": keep, "keep": keep_set, "merge": merge}
    return all_rules


def resolves_to(original: str, canonical: str, rules: dict) -> bool:
    """Check whether an original label canonically resolves to a given label."""
    return (original in rules["keep"] and original == canonical) or rules["merge"].get(original) == canonical


def load_graphs(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        graphs = json.load(handle)
    if not isinstance(graphs, dict):
        raise ValueError(f"{path}: expected image-ID-to-scene-graph dictionary")
    return graphs


def new_stats() -> dict:
    return {
        "images": 0,
        "objects": Counter(), "attributes": Counter(), "attribute_objects": Counter(),
        "relations": Counter(), "triples": Counter(),
        "original": {kind: Counter() for kind in KINDS},
        "presence": {"macro": {kind: Counter() for kind in KINDS},
                     "original": {kind: Counter() for kind in KINDS},
                     "triples": Counter()},
    }


def analyze_graphs(graphs: dict, rules: dict, location: str) -> dict:
    """Validate prepared scene graphs and collect occurrence/image-presence counts.

    Original attribute counts include all source attributes on RETAINED objects,
    even if an attribute does not map to a macro category. Original relation
    counts reflect distinct preserved original_names, not discarded relations or
    multiple duplicate original edges collapsed during preprocessing.
    """
    stats = new_stats()
    stats["images"] = len(graphs)
    for image_id, graph in graphs.items():
        if not isinstance(graph, dict) or not isinstance(graph.get("objects"), dict):
            raise ValueError(f"{location}: invalid objects at image {image_id!r}")
        objects = graph["objects"]
        if not objects:
            raise ValueError(f"{location}: empty retained graph {image_id!r}")
        image_seen = {"macro": {kind: set() for kind in KINDS},
                      "original": {kind: set() for kind in KINDS},
                      "triples": set()}
        for oid, obj in objects.items():
            if not isinstance(obj, dict) or obj.get("name") not in rules["object"]["keep"]:
                raise ValueError(f"{location}: invalid object at {image_id}/{oid}")
        for source_id, obj in objects.items():
            where = f"{location}: {image_id}/{source_id}"
            label = obj["name"]
            original_name = obj.get("original_name")
            if not isinstance(original_name, str) or not resolves_to(original_name, label, rules["object"]):
                raise ValueError(f"{where}: missing or inconsistent original object label")
            stats["objects"][label] += 1
            stats["original"]["object"][original_name] += 1
            image_seen["macro"]["object"].add(label)
            image_seen["original"]["object"].add(original_name)

            attrs = obj.get("attributes")
            originals = obj.get("original_attributes")
            if not isinstance(attrs, list) or not isinstance(originals, list):
                raise ValueError(f"{where}: invalid attributes or original_attributes")
            if any(not isinstance(a, str) or a not in rules["attribute"]["keep"] for a in attrs):
                raise ValueError(f"{where}: unknown canonical attribute")
            if len(attrs) != len(set(attrs)):
                raise ValueError(f"{where}: duplicate canonical attributes")
            if any(not isinstance(a, str) for a in originals):
                raise ValueError(f"{where}: invalid original attribute")
            expected_attrs = {
                a if a in rules["attribute"]["keep"] else rules["attribute"]["merge"].get(a)
                for a in originals
            } - {None}
            if set(attrs) != expected_attrs:
                raise ValueError(f"{where}: canonical attributes do not match original_attributes")
            for attr in attrs:
                stats["attributes"][attr] += 1
                stats["attribute_objects"][attr] += 1
                image_seen["macro"]["attribute"].add(attr)
            for original in originals:
                stats["original"]["attribute"][original] += 1
                image_seen["original"]["attribute"].add(original)

            rels = obj.get("relations")
            if not isinstance(rels, list):
                raise ValueError(f"{where}: invalid relations")
            seen_edges = set()
            for rel in rels:
                if not isinstance(rel, dict):
                    raise ValueError(f"{where}: invalid relation record")
                relation = rel.get("name")
                if not isinstance(relation, str) or relation not in rules["relation"]["keep"]:
                    raise ValueError(f"{where}: unknown canonical relation {relation!r}")
                target_id = str(rel.get("object"))
                if target_id not in objects:
                    raise ValueError(f"{where}: missing relation target {target_id}")
                edge = (relation, target_id)
                if edge in seen_edges:
                    raise ValueError(f"{where}: duplicate canonical relation to {target_id}")
                seen_edges.add(edge)
                original_names = rel.get("original_names")
                if (not isinstance(original_names, list) or not original_names
                        or len(original_names) != len(set(original_names))
                        or any(not isinstance(n, str) or not resolves_to(n, relation, rules["relation"])
                               for n in original_names)
                        or rel.get("original_name") != original_names[0]):
                    raise ValueError(f"{where}: inconsistent relation provenance")
                triple = (label, relation, objects[target_id]["name"])
                stats["relations"][relation] += 1
                stats["triples"][triple] += 1
                image_seen["macro"]["relation"].add(relation)
                image_seen["triples"].add(triple)
                for name in original_names:
                    stats["original"]["relation"][name] += 1
                    image_seen["original"]["relation"].add(name)

        for level in ("macro", "original"):
            for kind in KINDS:
                stats["presence"][level][kind].update(image_seen[level][kind])
        stats["presence"]["triples"].update(image_seen["triples"])

    if sum(stats["relations"].values()) != sum(stats["triples"].values()):
        raise ValueError(f"{location}: relation/triple count mismatch")
    if stats["attributes"] != stats["attribute_objects"]:
        raise ValueError(f"{location}: attribute occurrence/object count mismatch")
    return stats


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Saved: {path}")


def make_frequency_rows(labels: list[str], results: dict, key: str) -> list[dict]:
    rows = [{"label": label, "train_count": results["train"][key][label],
             "val_count": results["val"][key][label]} for label in labels]
    return sorted(rows, key=lambda r: (-r["train_count"], -r["val_count"], r["label"]))


def write_legacy_tables(results: dict, rules: dict, output_dir: Path) -> None:
    """Preserve historical four output CSV schemas and filenames unchanged."""
    for kind in ("object", "relation"):
        rows = make_frequency_rows(rules[kind]["labels"], results, kind + "s")
        write_csv(output_dir / f"{kind}_macro_frequencies.csv",
                  ["label", "train_count", "val_count"], rows)

    attributes = make_frequency_rows(rules["attribute"]["labels"], results, "attributes")
    totals = {split: sum(results[split]["objects"].values()) for split in SPLITS}
    for row in attributes:
        label = row["label"]
        for split in SPLITS:
            row[f"{split}_objects_with_attribute"] = results[split]["attribute_objects"][label]
            row[f"{split}_object_coverage_pct"] = (
                round(100 * results[split]["attribute_objects"][label] / totals[split], 4)
                if totals[split] else 0.0)
    write_csv(output_dir / "attribute_macro_frequencies.csv",
              ["label", "train_count", "val_count", "train_objects_with_attribute",
               "val_objects_with_attribute", "train_object_coverage_pct", "val_object_coverage_pct"],
              attributes)

    all_triples = set(results["train"]["triples"]) | set(results["val"]["triples"])
    triple_rows = [{"source_class": src, "relation": rel, "target_class": dst,
                    "train_count": results["train"]["triples"][(src, rel, dst)],
                    "val_count": results["val"]["triples"][(src, rel, dst)]}
                   for src, rel, dst in all_triples]
    triple_rows.sort(key=lambda r: (-r["train_count"], -r["val_count"],
                                    r["source_class"], r["relation"], r["target_class"]))
    write_csv(output_dir / "relation_macro_class_triples.csv",
              ["source_class", "relation", "target_class", "train_count", "val_count"], triple_rows)


def distribution_metrics(reference: Counter, subset: Counter) -> dict:
    """TVD and sqrt(JS divergence base 2) for occurrence distributions."""
    labels = set(reference) | set(subset)
    rt = sum(reference.values())
    st = sum(subset.values())
    if rt == 0 or st == 0:
        return {"tvd": None, "js_distance": None,
                "note": "Undefined: at least one occurrence total is zero"}
    tvd = 0.0
    js_divergence = 0.0
    for label in labels:
        p = reference[label] / rt
        q = subset[label] / st
        m = (p + q) / 2
        tvd += abs(p - q) / 2
        if p:
            js_divergence += 0.5 * p * math.log2(p / m)
        if q:
            js_divergence += 0.5 * q * math.log2(q / m)
    return {"tvd": round(tvd, 8),
            "js_distance": round(math.sqrt(max(0.0, js_divergence)), 8),
            "note": "Occurrence-normalized; not image-prevalence distance"}


def label_comparison_rows(level: str, split: str, reference: dict, subset: dict, rules: dict) -> list[dict]:
    rows = []
    for kind in KINDS:
        ref_counts = reference[kind + "s"] if level == "macro" else reference["original"][kind]
        sub_counts = subset[kind + "s"] if level == "macro" else subset["original"][kind]
        ref_presence = reference["presence"][level][kind]
        sub_presence = subset["presence"][level][kind]
        all_labels = set(ref_counts) | set(sub_counts)
        if level == "macro":
            all_labels.update(rules[kind]["labels"])
        for label in sorted(all_labels):
            r_n, s_n = ref_presence[label], sub_presence[label]
            r_pct = 100 * r_n / reference["images"] if reference["images"] else 0.0
            s_pct = 100 * s_n / subset["images"] if subset["images"] else 0.0
            rows.append({"split": split, "kind": kind, "label": label,
                         "reference_images": r_n, "subset_images": s_n,
                         "reference_prevalence_pct": round(r_pct, 6),
                         "subset_prevalence_pct": round(s_pct, 6),
                         "prevalence_diff_pp": round(s_pct-r_pct, 6),
                         "reference_occurrences": ref_counts[label],
                         "subset_occurrences": sub_counts[label],
                         "reference_total_images": reference["images"],
                         "subset_total_images": subset["images"],
                         "status": "missing_in_subset" if r_n and not s_n else
                                   "unseen_in_reference" if s_n and not r_n else "present" if s_n else "absent"})
    return rows


def triple_comparison_rows(split: str, reference: dict, subset: dict) -> list[dict]:
    result = []
    for src, relation, dst in sorted(set(reference["triples"]) | set(subset["triples"])):
        triple = (src, relation, dst)
        rp = reference["presence"]["triples"][triple]
        sp = subset["presence"]["triples"][triple]
        r_pct = 100 * rp/reference["images"] if reference["images"] else 0
        s_pct = 100 * sp/subset["images"] if subset["images"] else 0
        result.append({"split": split, "source_class": src, "relation": relation,
                       "target_class": dst, "reference_occurrences": reference["triples"][triple],
                       "subset_occurrences": subset["triples"][triple],
                       "reference_images": rp, "subset_images": sp,
                       "reference_prevalence_pct": round(r_pct, 6),
                       "subset_prevalence_pct": round(s_pct, 6),
                       "prevalence_diff_pp": round(s_pct-r_pct, 6),
                       "status": "missing_in_subset" if rp and not sp else "present"})
    return result


def check_subset_integrity(subset_dir: Path, source_graphs: dict, subset_graphs: dict,
                           expected_sizes: dict | None) -> dict:
    """Verify IDs and unmodified graph contents before quantitative comparison."""
    ids = {}
    for split in SUBSET_SPLITS:
        ids_path = subset_dir / f"{split}_ids.json"
        with ids_path.open(encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
            raise ValueError(f"{ids_path}: expected nonempty string IDs")
        if len(values) != len(set(values)):
            raise ValueError(f"{ids_path}: duplicate image IDs")
        if expected_sizes is not None and len(values) != expected_sizes[split]:
            raise ValueError(f"{split}: expected {expected_sizes[split]} images, got {len(values)}")
        ids[split] = set(values)
        sub = subset_graphs[split]
        src = source_graphs[SOURCE_FOR_SUBSET[split]]
        if set(sub) != ids[split]:
            raise ValueError(f"{split}: annotation IDs do not match {ids_path.name}")
        if not ids[split] <= set(src):
            raise ValueError(f"{split}: subset IDs missing from source {SOURCE_FOR_SUBSET[split]}")
        # Check that export contains the original prepared scene graphs unchanged.
        for image_id in ids[split]:
            if sub[image_id] != src[image_id]:
                raise ValueError(f"{split}: scene graph differs from source for image {image_id}")
    for i, split in enumerate(SUBSET_SPLITS):
        for other in SUBSET_SPLITS[i+1:]:
            if ids[split] & ids[other]:
                raise ValueError(f"Overlapping image IDs between {split} and {other}")
    if set(source_graphs["train"]) & set(source_graphs["val"]):
        raise ValueError("Source train and validation image IDs overlap")
    return {"status": "passed", "split_sizes": {k: len(v) for k, v in ids.items()},
            "source_for_split": SOURCE_FOR_SUBSET}


def compare_subset(source_graphs: dict, source_stats: dict, rules: dict, subset_dir: Path,
                   output_dir: Path, expected_sizes: dict | None) -> None:
    subset_annotations = subset_dir / "annotations"
    subset_graphs = {split: load_graphs(subset_annotations / f"{split}_sceneGraphs.json")
                     for split in SUBSET_SPLITS}
    integrity = check_subset_integrity(subset_dir, source_graphs, subset_graphs, expected_sizes)
    subset_stats = {split: analyze_graphs(subset_graphs[split], rules, f"subset/{split}")
                    for split in SUBSET_SPLITS}

    macro_rows, original_rows, triple_rows, coverage_rows, metric_rows = [], [], [], [], []
    summary = {"methodology": {
        "reference_mapping": SOURCE_FOR_SUBSET,
        "prevalence": "Percent of images with >=1 annotation of the label; multi-label percentages need not sum to 100",
        "original_attributes": "All original attributes on retained objects, including attributes removed from macro mapping",
        "original_relations": "Distinct original_names recorded on retained canonical edges; discarded relations absent",
        "tvd": "Total variation distance of normalized annotation occurrences, range 0..1",
        "js_distance": "Square root of Jensen-Shannon divergence with base-2 logarithms, range 0..1",
        "warning": "Subset is label-enriched, not an unbiased random sample. No automatic quality thresholds.",
    }, "integrity": integrity, "splits": {}, "cross_split": {}}

    for split in SUBSET_SPLITS:
        reference = source_stats[SOURCE_FOR_SUBSET[split]]
        sampled = subset_stats[split]
        macro_rows.extend(label_comparison_rows("macro", split, reference, sampled, rules))
        original_rows.extend(label_comparison_rows("original", split, reference, sampled, rules))
        triple_rows.extend(triple_comparison_rows(split, reference, sampled))
        coverage = {}
        for level in ("macro", "original"):
            for kind in KINDS:
                ref_counter = reference[kind+"s"] if level == "macro" else reference["original"][kind]
                sub_counter = sampled[kind+"s"] if level == "macro" else sampled["original"][kind]
                ref_labels = {k for k, n in ref_counter.items() if n > 0}
                sub_labels = {k for k, n in sub_counter.items() if n > 0}
                covered = len(ref_labels & sub_labels)
                row = {"split": split, "level": level, "kind": kind,
                       "reference_unique": len(ref_labels), "subset_unique": len(sub_labels),
                       "covered_reference_labels": covered,
                       "missing_reference_labels": len(ref_labels - sub_labels),
                       "coverage_ratio_pct": round(100*covered/len(ref_labels), 6) if ref_labels else None}
                coverage_rows.append(row)
                coverage[f"{level}_{kind}"] = row
                metric_rows.append({"split": split, "level": level, "kind": kind,
                                    **distribution_metrics(ref_counter, sub_counter)})
        rt = set(reference["triples"])
        st = set(sampled["triples"])
        covered = len(rt & st)
        triple_coverage = {"split": split, "level": "macro", "kind": "class_triple",
                           "reference_unique": len(rt), "subset_unique": len(st),
                           "covered_reference_labels": covered,
                           "missing_reference_labels": len(rt-st),
                           "coverage_ratio_pct": round(100*covered/len(rt), 6) if rt else None}
        coverage_rows.append(triple_coverage)
        coverage["macro_class_triple"] = triple_coverage
        metric_rows.append({"split": split, "level": "macro", "kind": "class_triple",
                            **distribution_metrics(reference["triples"], sampled["triples"])})
        summary["splits"][split] = {"images": sampled["images"],
                                     "reference_split": SOURCE_FOR_SUBSET[split],
                                     "reference_images": reference["images"],
                                     "coverage": coverage}

    for split in ("val", "test"):
        novel = {"macro": {}, "original": {}}
        for level in ("macro", "original"):
            for kind in KINDS:
                train_labels = (subset_stats["train"][kind+"s"] if level == "macro"
                                else subset_stats["train"]["original"][kind])
                eval_labels = (subset_stats[split][kind+"s"] if level == "macro"
                               else subset_stats[split]["original"][kind])
                novel[level][kind] = sorted(set(eval_labels) - set(train_labels))
        novel["macro_class_triples"] = [list(v) for v in
                                        sorted(set(subset_stats[split]["triples"]) -
                                               set(subset_stats["train"]["triples"]))]
        summary["cross_split"][split + "_not_in_train"] = novel

    columns = ["split", "kind", "label", "reference_images", "subset_images",
               "reference_prevalence_pct", "subset_prevalence_pct", "prevalence_diff_pp",
               "reference_occurrences", "subset_occurrences", "reference_total_images",
               "subset_total_images", "status"]
    write_csv(output_dir / "macro_label_comparison.csv", columns, macro_rows)
    write_csv(output_dir / "original_label_comparison.csv", columns, original_rows)
    write_csv(output_dir / "relation_triple_comparison.csv",
              ["split", "source_class", "relation", "target_class", "reference_occurrences",
               "subset_occurrences", "reference_images", "subset_images",
               "reference_prevalence_pct", "subset_prevalence_pct", "prevalence_diff_pp", "status"],
              triple_rows)
    write_csv(output_dir / "split_coverage.csv",
              ["split", "level", "kind", "reference_unique", "subset_unique",
               "covered_reference_labels", "missing_reference_labels", "coverage_ratio_pct"],
              coverage_rows)
    write_csv(output_dir / "distribution_metrics.csv",
              ["split", "level", "kind", "tvd", "js_distance", "note"], metric_rows)
    write_json(output_dir / "subset_analysis_summary.json", summary)
    for split in SUBSET_SPLITS:
        print(f"{split}: {subset_stats[split]['images']:,} images; reference "
              f"{SOURCE_FOR_SUBSET[split]} ({source_stats[SOURCE_FOR_SUBSET[split]]['images']:,} images)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "data/gqa_probing/annotations")
    parser.add_argument("--config-dir", type=Path, default=Path(__file__).resolve().parent / "config")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/gqa_probing/tables")
    parser.add_argument("--compare-subset", action="store_true", help="Compare frozen 5k subset with source splits")
    parser.add_argument("--subset-dir", type=Path, default=REPO_ROOT / "data/gqa_probing/subsets/probing_5k")
    parser.add_argument("--allow-custom-sizes", action="store_true",
                        help="Allow any positive split size instead of requiring 3000/1000/1000")
    args = parser.parse_args()
    rules = load_rules(args.config_dir)
    source_graphs = {split: load_graphs(args.input_dir / f"{split}_sceneGraphs.json") for split in SPLITS}
    results = {split: analyze_graphs(source_graphs[split], rules, f"source/{split}") for split in SPLITS}
    write_legacy_tables(results, rules, args.output_dir)
    for split in SPLITS:
        result = results[split]
        print(f"{split}: images={result['images']:,}; objects={sum(result['objects'].values()):,}; "
              f"attributes={sum(result['attributes'].values()):,}; "
              f"relations={sum(result['relations'].values()):,}")
    if args.compare_subset:
        compare_subset(source_graphs, results, rules, args.subset_dir,
                       args.output_dir / "subset_comparison",
                       None if args.allow_custom_sizes else EXPECTED_SUBSET_SIZES)


if __name__ == "__main__":
    main()
