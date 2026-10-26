# GQA Probing Dataset Preparation

This directory contains the preprocessing pipeline used to construct a filtered GQA scene-graph dataset for subsequent layer-wise visual probing experiments. **It prepares annotations and image-ID lists; it does not extract visual features, train probes, or modify the original images.**

The pipeline applies manually defined, canonical vocabularies of **100 object classes, 100 attributes, and 100 relations**. It first selects objects by semantic label, then validates/clips their bounding boxes, filters attributes, and removes relations that are unsupported by the retained objects.

## 1. Directory structure

Run commands from the repository root. The relevant files are:

```text
scripts/gqa_probing/
├── prepare_gqa_probing.py
├── README.md
└── config/
    ├── object_mapping.json
    ├── attribute_mapping.json
    └── relation_mapping.json

data/
├── gqa/
│   ├── annotations/
│   │   ├── train_sceneGraphs.json
│   │   └── val_sceneGraphs.json
│   └── images/
└── gqa_cleaned/
    ├── annotations/
    │   ├── train_sceneGraphs.json
    │   └── val_sceneGraphs.json
    ├── image_ids/
    │   ├── train.txt
    │   └── val.txt
    └── preprocessing_stats.json
```

The cleaned dataset shares image files with `data/gqa/images/`; it does not contain a separate copy of those images.

## 2. Vocabulary selection and mapping configuration

The vocabularies were selected through an exploratory, frequency-informed and manually reviewed procedure:

1. Candidate **object classes** were chosen using training-set frequencies. Semantically similar labels, including some singular/plural forms, were consolidated; unsuitable labels were discarded. Additional candidates were reviewed iteratively until 100 canonical object labels were selected.
2. Candidate **attributes** were analyzed on instances belonging to the selected object vocabulary. Similar labels were merged and unsuitable ones excluded until 100 canonical attributes were selected.
3. Candidate **relations** were analyzed between instances whose source **and** target object classes belonged to the selected object vocabulary. The same manual review produced 100 canonical relation labels.

The three mapping JSON files record the resulting decisions, not the complete sequence or rationale of every manual review step. The exploratory scripts used during vocabulary design are **not required** to execute the current preprocessing pipeline.

Each mapping has the following schema:

```json
{
  "keep": ["canonical_label"],
  "merge": {"alternative_label": "canonical_label"},
  "remove": ["excluded_label"]
}
```

- **`keep`**: preserve the label unchanged; these are the canonical output labels.
- **`merge`**: replace a source label with its canonical target, which must appear in `keep`.
- **`remove`**: explicitly discard a label.
- **Unlisted labels**: also discarded by default; the vocabulary is closed.

For example, the supplied object mapping merges `man` and `woman` into `person`; the attribute mapping merges `tiny` into `small`; the relation mapping merges `beside` into `next to`.

At startup the script validates the three mappings, including the requirement of **exactly 100 distinct `keep` labels per vocabulary**, non-conflicting actions, and `merge` targets included in `keep`. The mappings should be version-controlled alongside the preprocessing script because changes to them can change dataset composition.

## 3. Preprocessing sequence

For each input image scene graph, `prepare_gqa_probing.py` performs the following steps in order.

### 3.1 Semantic object filtering

Each original object label is resolved against `object_mapping.json`. Objects with `keep` or `merge` labels are selected; explicitly removed, unlisted, or invalid labels are discarded. This selection happens **before** bounding-box validation, so geometric checks are applied only to semantically selected objects.

### 3.2 Bounding-box validation and clipping

Bounding boxes use GQA's `(x, y, w, h)` convention. Image dimensions are read from the scene graph's `width` and `height` fields and must be finite and positive.

| Status | Criteria | Action |
|---|---|---|
| `valid` | Positive width/height; box fully inside the image | Keep unchanged |
| `clippable` | Positive width/height; box crosses an image boundary but overlaps the image with positive area | Clip to image boundaries |
| `invalid` | Non-positive/non-finite dimensions or coordinates, or no positive-area intersection | Remove object |

Negative `x` or `y` values are **not** automatically invalid. For example, for a 100-pixel-wide image, a box with `x = -10` and `w = 30` can be clipped to `x = 0` and `w = 20` if its vertical extent also intersects the image.

Clipping intersects the rectangle with the image bounds; no arbitrary overflow threshold is used. The resulting coordinates and dimensions are written only to the **output annotations**, not to the original JSON or image files.

### 3.3 Attribute filtering

Attributes are filtered on objects that survived both semantic and geometric selection, using `attribute_mapping.json`. Removed and unlisted attributes are discarded; retained and merged attributes are written to the output objects. Objects can remain in the dataset with an empty attribute list.

**No post-merge attribute deduplication is performed.** For example, `little` and `small` can both become `small` in one object's list; counts therefore refer to annotation occurrences, not necessarily unique object-attribute pairs.

### 3.4 Relation filtering

For each retained source object, a relation is kept only if its name resolves to a canonical relation and its target object ID still exists in the **retained object dictionary**. This prevents dangling references when semantic or geometric filtering removes objects. An object can be retained without any relations.

### 3.5 Empty graphs and output validation

An image's scene graph is discarded if it has no retained objects. Before writing a split, the script verifies that:

- each retained image contains at least one object;
- object, attribute, and relation labels belong to the respective canonical `keep` vocabularies;
- each output bounding box is entirely in-frame with strictly positive dimensions;
- every relation target refers to an existing retained object.

Validation checks structural and geometric consistency; it **does not** establish annotation semantics, class balance, or the existence/readability of the corresponding image files.

## 4. Running the preprocessing

The preprocessing script uses only the Python standard library and requires **Python 3.10+** (due to type annotations and `str.removesuffix`). No external Python packages are required for this particular script.

From the repository root, first produce a separate test dataset so that the existing cleaned output remains available for comparison:

```powershell
python scripts/gqa_probing/prepare_gqa_probing.py --input-dir data/gqa/annotations --output-dir data/gqa_cleaned_test
```

After inspecting the test output, the intended cleaned directory can be generated with:

```powershell
python scripts/gqa_probing/prepare_gqa_probing.py --input-dir data/gqa/annotations --output-dir data/gqa_cleaned
```

**Caution:** Writing to `data/gqa_cleaned` replaces the generated files for the selected splits. Back up an existing cleaned dataset first if it must be retained.

Optional arguments:

| Argument | Default | Meaning |
|---|---|---|
| `--input-dir` | Required | Directory containing `<split>_sceneGraphs.json` |
| `--output-dir` | Required | Destination root for generated files |
| `--config-dir` | `scripts/gqa_probing/config/` | Directory containing the three mapping JSON files |
| `--splits` | `train val` | Space-separated split names to process |

Example with explicit configuration and splits:

```powershell
python scripts/gqa_probing/prepare_gqa_probing.py --input-dir data/gqa/annotations --output-dir data/gqa_cleaned_test --config-dir scripts/gqa_probing/config --splits train val
```

Input and output directories must be disjoint. The script loads one input split and constructs its filtered output in memory, so sufficient RAM and disk space are necessary for large scene-graph JSON files.

## 5. Generated outputs

For each requested split, the script creates:

| Output | Content |
|---|---|
| `annotations/<split>_sceneGraphs.json` | Filtered scene graphs, including clipped bounding boxes |
| `image_ids/<split>.txt` | One retained image ID per line |
| `preprocessing_stats.json` | Per-split filtering counters and validation results |

The original annotations in `data/gqa/annotations/` and all files in `data/gqa/images/` remain unchanged. The generated ID lists define which existing images belong to each cleaned split.

### Understanding `preprocessing_stats.json`

Selected counters include:

| Statistic | Interpretation |
|---|---|
| `images_before`, `images_after` | Scene graphs loaded and retained |
| `images_removed_without_selected_objects` | Images dropped because no usable object remained |
| `objects_before`, `objects_after` | Total original and retained object instances |
| `objects_selected_semantic`, `objects_removed_semantic` | Result of object-vocabulary filtering |
| `bbox_valid`, `bbox_clippable`, `bbox_invalid` | Geometry categories for semantically selected objects |
| `objects_removed_invalid_bbox` | Selected objects removed due to invalid geometry |
| `attributes_before`, `relations_before` | Annotation occurrences across **all original objects** |
| `attributes_on_retained_objects_before`, `relations_on_retained_objects_before` | Annotation occurrences on objects surviving object and box filtering, before attribute/relation filtering |
| `attributes_after`, `relations_after` | Output annotation occurrences |
| `relations_removed_missing_target` | Canonically selected relations removed because the target object was not retained |
| `object_keep`, `object_merge`, `object_remove_explicit`, `object_remove_unlisted` | Object-label action counts; analogous counters exist for attributes and relations |
| `validation` | Validation status and counts of checked output annotations |

**Counter scopes matter.** Attribute and relation action counters are evaluated only for objects that survive the preceding filters. `relations_removed_missing_target` counts relations that passed relation-label filtering but failed the target-reference check. Missing counters can indicate a zero count because counters are serialized from recorded events.

### Reference run

In the revised preprocessing run examined during dataset preparation, the generated cleaned scene graphs contained:

| Metric | Train | Validation | Total |
|---|---:|---:|---:|
| Retained images | 73,347 | 10,430 | 83,777 |
| Retained objects | 743,909 | 105,229 | 849,138 |
| Retained attribute occurrences | 342,783 | 49,026 | 391,809 |
| Retained relation occurrences | 1,739,075 | 244,281 | 1,983,356 |
| Clipped boxes | 6,787 | 926 | 7,713 |
| Invalid boxes removed | 105 | 58 | 163 |

These are **observed counts for that run**, not fixed expectations enforced by the code. Regenerating the dataset from different source annotations or mapping versions can produce different values.

## 6. Recommended validation workflow

After generating the cleaned dataset, run the read-only checks from `scripts/gqa_analysis/` (see its README):

```powershell
python scripts/gqa_analysis/check_gqa_images.py --dataset cleaned
python scripts/gqa_analysis/analyze_gqa_bboxes.py --dataset cleaned
python scripts/gqa_analysis/inspect_gqa.py --dataset cleaned
python scripts/gqa_analysis/plot_frequencies.py --dataset cleaned
```

Expected quality checks include: no scene graph without a corresponding source image, no overlap between train and validation IDs, consistency between cleaned scene graphs and `image_ids/*.txt`, and **all output bounding boxes classified as `valid`**.

The original GQA images also include files outside the annotated train/validation scene-graph subsets. Those files are **not automatically corrupted or unnecessary**, and this pipeline does not delete them.

## 7. Scope and limitations

- This preprocessing produces a **filtered annotation dataset**, not a probe-ready feature tensor or an object-crop export.
- The fixed 100/100/100 label inventories do not guarantee sufficient examples of every label in each split. Inspect post-filtering frequencies before probe training.
- Manual vocabulary design affects the experiment definition and should be described in the thesis; the mappings preserve final actions, not detailed subjective selection reasons.
- No automatic semantic verification, class balancing, label deduplication, or visual-feature extraction is performed.
- The preprocessing JSON preserves other object/image fields where possible, while updating labels, filtered relations/attributes, and clipped box geometry.
