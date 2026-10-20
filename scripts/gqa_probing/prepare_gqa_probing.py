#!/usr/bin/env python3
"""Prepare a filtered GQA scene-graph dataset for layer-wise probing.

Inputs are GQA-style scene graph JSON files whose root maps image IDs to
records containing an 'objects' dictionary. Object records have 'name',
'attributes', 'relations'; relations use 'name' and 'object' (target object ID).

Unknown labels default to REMOVE for all three vocabularies. The script never
modifies source annotations or image files. It preserves all unmodified fields.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

VOCABS = ('object', 'attribute', 'relation')


def load_rules(config_dir: Path):
    rules = {}
    for kind in VOCABS:
        path = config_dir / f'{kind}_mapping.json'
        with path.open(encoding='utf-8') as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or set(data) != {'keep', 'merge', 'remove'}:
            raise ValueError(f'Invalid mapping format: {path}')
        keep = set(data['keep'])
        merge = data['merge']
        remove = set(data['remove'])
        if len(keep | remove | set(merge)) != len(keep) + len(remove) + len(merge):
            raise ValueError(f'Conflicting actions in {path}')
        if not set(merge.values()) <= keep:
            raise ValueError(f'MERGE targets not in KEEP vocabulary: {path}: {set(merge.values()) - keep}')
        rules[kind] = {'keep': keep, 'merge': merge, 'remove': remove}
    return rules


def resolve(label, rules, counters, kind):
    if not isinstance(label, str):
        counters[f'{kind}_invalid_label'] += 1
        return None
    if label in rules['keep']:
        counters[f'{kind}_keep'] += 1
        return label
    if label in rules['merge']:
        counters[f'{kind}_merge'] += 1
        return rules['merge'][label]
    if label in rules['remove']:
        counters[f'{kind}_remove_explicit'] += 1
        return None
    counters[f'{kind}_remove_unlisted'] += 1
    return None


def preprocess_graph(image_graph, rules, counts):
    if not isinstance(image_graph, dict) or not isinstance(image_graph.get('objects'), dict):
        raise ValueError('Expected scene graph with an objects dictionary')
    objects = image_graph['objects']
    counts['objects_before'] += len(objects)
    kept = {}
    for oid, source in objects.items():
        if not isinstance(source, dict):
            raise ValueError(f'Object {oid} must be a dictionary')
        new_name = resolve(source.get('name'), rules['object'], counts, 'object')
        if new_name is None:
            continue
        obj = deepcopy(source)
        obj['name'] = new_name
        attrs = source.get('attributes', [])
        if not isinstance(attrs, list):
            raise ValueError(f'Object {oid}: attributes must be a list')
        counts['attributes_before'] += len(attrs)
        obj['attributes'] = []
        for attr in attrs:
            replacement = resolve(attr, rules['attribute'], counts, 'attribute')
            if replacement is not None:
                obj['attributes'].append(replacement)
        counts['attributes_after'] += len(obj['attributes'])
        kept[str(oid)] = obj
    counts['objects_after'] += len(kept)
    if not kept:
        return None
    for oid, obj in kept.items():
        relations = obj.get('relations', [])
        if not isinstance(relations, list):
            raise ValueError(f'Object {oid}: relations must be a list')
        source_relations = relations
        obj['relations'] = []
        for rel in source_relations:
            counts['relations_before_on_retained_objects'] += 1
            if not isinstance(rel, dict):
                raise ValueError(f'Object {oid}: relation must be a dictionary')
            new_name = resolve(rel.get('name'), rules['relation'], counts, 'relation')
            if new_name is None:
                continue
            target = rel.get('object')
            if str(target) not in kept:
                counts['relations_removed_missing_target'] += 1
                continue
            new_rel = deepcopy(rel)
            new_rel['name'] = new_name
            obj['relations'].append(new_rel)
            counts['relations_after'] += 1
    result = dict(image_graph)
    result['objects'] = kept
    return result



def validate_filtered_graphs(graphs, rules):
    """Fail if any label or relation endpoint violates the selected vocabularies."""
    errors = []
    counts = Counter()
    for image_id, graph in graphs.items():
        objects = graph['objects']
        if not objects:
            errors.append(f'{image_id}: retained image with no objects')
        for oid, obj in objects.items():
            counts['objects_checked'] += 1
            if obj.get('name') not in rules['object']['keep']:
                errors.append(f'{image_id}/{oid}: invalid object label {obj.get("name")!r}')
            for attr in obj.get('attributes', []):
                counts['attributes_checked'] += 1
                if attr not in rules['attribute']['keep']:
                    errors.append(f'{image_id}/{oid}: invalid attribute {attr!r}')
            for rel in obj.get('relations', []):
                counts['relations_checked'] += 1
                if rel.get('name') not in rules['relation']['keep']:
                    errors.append(f'{image_id}/{oid}: invalid relation {rel.get("name")!r}')
                if str(rel.get('object')) not in objects:
                    errors.append(f'{image_id}/{oid}: dangling relation target {rel.get("object")!r}')
            if len(errors) >= 30:
                raise ValueError('Filtered graph validation failed (first 30 errors):\n' + '\n'.join(errors))
    if errors:
        raise ValueError('Filtered graph validation failed:\n' + '\n'.join(errors))
    return dict(counts)


def process_file(source: Path, dest: Path, rules):
    with source.open(encoding='utf-8') as f:
        graphs = json.load(f)
    if not isinstance(graphs, dict):
        raise ValueError(f'{source}: expected image-ID-to-scene-graph dictionary')
    counters = Counter()
    output = {}
    for image_id, scene in graphs.items():
        counters['images_before'] += 1
        filtered = preprocess_graph(scene, rules, counters)
        if filtered is None:
            counters['images_removed_without_selected_objects'] += 1
        else:
            output[image_id] = filtered
            counters['images_after'] += 1
    validation = validate_filtered_graphs(output, rules)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('w', encoding='utf-8') as fh:
        json.dump(output, fh, ensure_ascii=False, separators=(',', ':'))
    ids_path = dest.parent.parent / 'image_ids' / f'{dest.stem.replace("_sceneGraphs", "")}.txt'
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.write_text(''.join(f'{key}\n' for key in output), encoding='utf-8')
    result_stats = dict(sorted(counters.items()))
    result_stats['validation'] = {'status': 'passed', **validation}
    return result_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True, help='Folder containing train_sceneGraphs.json etc.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config-dir', type=Path, default=Path(__file__).resolve().parent / 'config')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'], help='Input scene graph filename stems, e.g. train val')
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if input_dir == output_dir or input_dir in output_dir.parents:
        raise ValueError('Output directory must not be the input directory or a subdirectory of it')
    rules = load_rules(args.config_dir)
    summary = {}
    for split in args.splits:
        if not split.replace('_', '').isalnum():
            raise ValueError(f'Invalid split name: {split}')
        source = input_dir / f'{split}_sceneGraphs.json'
        if not source.is_file():
            raise FileNotFoundError(source)
        dest = output_dir / 'annotations' / source.name
        summary[split] = process_file(source, dest, rules)
        print(f'{split}: {summary[split].get("images_after", 0)} images retained')
    (output_dir / 'preprocessing_stats.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Output: {output_dir}')


if __name__ == '__main__':
    main()
