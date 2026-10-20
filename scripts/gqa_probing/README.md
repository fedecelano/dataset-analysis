# GQA probing preparation

This directory provides a non-destructive preprocessing pipeline driven by the three user-provided CSV class-selection tables. It processes GQA `*_sceneGraphs.json` files and never touches originals.

## Run

```bash
python prepare_gqa_probing.py \
  --input-dir /path/to/GQA/sceneGraphs \
  --output-dir /path/to/GQA_probing \
  --splits train val
```

The input format is the standard GQA scene-graph shape: one JSON object indexed by image ID; each scene includes an `objects` dictionary indexed by object ID, with `name`, `attributes` (list of strings), and `relations` (list of `{name, object}` records).

The output contains `annotations/{split}_sceneGraphs.json`, `image_ids/{split}.txt`, and `preprocessing_stats.json`. Images are *not* copied or deleted; use the retained ID lists for image loading.

## Rules

- `KEEP`: preserve the label.
- `MERGE`: substitute the CSV-specified final label.
- `REMOVE`: delete the corresponding object, attribute or relation.
- Unlisted labels: remove by default.
- When an object is removed, all relations targeting that object are removed.
- Images with no surviving objects are excluded.
- Original object IDs, coordinates, image metadata, and unaffected object fields are kept.
- **No automatic deduplication** of attributes or relations after MERGE (avoids silently changing dataset multiplicity).

## Important

The relation count in the statistics is counted only for relations initially attached to objects that survived object filtering. The `train_source_classes` etc. columns from the relation CSV are observational metadata, not filtering criteria.

The script has been tested on a minimal synthetic scene graph. Check the actual source JSON shape before running it on the complete GQA dataset.

## Post-preprocessing validation

The script checks every retained object, attribute, and relation label against
the corresponding 100-class KEEP vocabulary (after MERGE). It checks every
relation target belongs to a retained object, and that retained images contain
at least one object. Any violation raises ValueError before writing the output.
The per-split statistics include `validation: {status: passed, ...}`.

Verified on the complete original `val_sceneGraphs.json` (10,696 images):
10,431 retained images, 105,287 objects, 49,056 attributes, and 244,347
relations. This validates annotation consistency, not the semantic accuracy
of ground-truth labels or image files.

The original `train` and `val` splits are preserved. A future separate script
will split cleaned `train` into new train/val and use cleaned original `val`
as test. This script performs no resplitting.
