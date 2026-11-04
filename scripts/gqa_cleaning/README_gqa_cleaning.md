# GQA Dataset Cleaning

This directory contains the scripts and mapping configurations used to build a filtered GQA scene-graph dataset for downstream probing experiments. The cleaning pipeline uses canonical vocabularies of **100 object classes, 100 attributes, and 100 relations**. It filters annotations, clips recoverable bounding boxes, preserves original labels for retained annotations, and exports image-ID lists and preprocessing statistics.

**Source scene graphs and images are never modified.** The scripts write files to the specified cleaned-dataset directory, potentially overwriting earlier generated outputs. Copying the corresponding images is a separate, optional operation; neither script extracts visual features or trains probing models.

## 1. Directory Structure

```text
project_root/
├── data/
│   ├── gqa/
│   │   ├── annotations/
│   │   │   ├── train_sceneGraphs.json
│   │   │   └── val_sceneGraphs.json
│   │   └── images/
│   └── gqa_cleaned/
│       ├── annotations/
│       │   ├── train_sceneGraphs.json
│       │   └── val_sceneGraphs.json
│       ├── image_ids/
│       │   ├── train.txt
│       │   └── val.txt
│       ├── preprocessing_stats.json
│       ├── images/                   # Optional: created by image extraction
│       └── image_copy_report.json    # Created by image extraction
├── scripts/
│   └── gqa_cleaning/
│       ├── prepare_gqa_cleaning.py
│       ├── extract_cleaned_gqa_images.py
│       ├── README_gqa_cleaning.md
│       └── config/
│           ├── object_mapping.json
│           ├── attribute_mapping.json
│           └── relation_mapping.json
└── outputs/
    └── gqa_analysis/
        ├── original/
        └── cleaned/
```

The cleaned annotations can reference images directly from `data/gqa/images/`. The optional `data/gqa_cleaned/images/` directory is not required for the cleaning procedure or for the current `gqa_analysis` scripts.

## 2. Requirements

Both scripts use only the **Python standard library** and require **Python 3.10 or later**. No third-party packages are needed to generate the cleaned annotations or copy the referenced images.

Run the commands below from the project root. Paths passed to `--input-dir`, `--output-dir`, `--images-dir`, and `--cleaned-dir` are interpreted relative to the current working directory when relative paths are provided. The default mapping directory is resolved relative to `prepare_gqa_cleaning.py`.

## 3. Vocabulary Mappings

The three configuration files define the allowed output labels and the transformations applied to incoming labels:

- `object_mapping.json`: object classes.
- `attribute_mapping.json`: object attributes.
- `relation_mapping.json`: directed relation labels.

Each JSON follows this structure (illustrative labels):

```json
{
  "keep": ["canonical_label"],
  "merge": {"alternative_label": "canonical_label"},
  "remove": ["excluded_label"]
}
```

| Action | Behavior |
|---|---|
| `keep` | Preserve the label as a canonical output label. |
| `merge` | Replace a source label with the specified canonical label in `keep`. |
| `remove` | Explicitly discard the label. |
| Unlisted label | Discard the label by default (closed vocabulary). |

For example, the supplied mappings include `man → person` and `woman → person` for objects; `little → small` and `tiny → small` for attributes; and `beside → next to` for relations.

At startup, `prepare_gqa_cleaning.py` checks that each mapping has exactly **100 distinct `keep` labels**, that the action sets do not conflict, and that every `merge` target is in `keep`. Changing a mapping changes the preprocessing specification and may change the resulting dataset. Keep the mapping files under version control.

**The 100 labels define the permitted vocabulary, not a guarantee that all 100 occur in every cleaned split.** The script does not balance label frequencies or assess the semantic suitability of mapping decisions.

## 4. Annotation Cleaning: `prepare_gqa_cleaning.py`

For every requested scene-graph split, the script applies the following steps in order.

### 4.1 Object Filtering

Object names are resolved through `object_mapping.json` before checking bounding boxes. Objects whose labels are removed, unlisted, or invalid are discarded; only semantically selected objects proceed to geometric validation.

### 4.2 Bounding-Box Validation and Clipping

Boxes use GQA's `(x, y, w, h)` representation. Image dimensions and accepted box dimensions must be finite and positive. Calculated box endpoints are also checked for non-finite values.

| Status | Condition | Action |
|---|---|---|
| `valid` | Positive dimensions; the entire box lies within the image. | Keep the coordinates unchanged. |
| `clippable` | Positive dimensions; the box extends outside the image but has positive-area overlap. | Replace its coordinates with the intersection with the image. |
| `invalid` | Invalid/non-finite geometry, non-positive dimensions, or no positive-area overlap. | Remove the object. |

Negative `x` or `y` coordinates do not automatically make a box invalid. There is **no percentage-based overflow threshold**. The original scene graphs are not rewritten.

### 4.3 Attribute Filtering and Deduplication

Attributes are processed only for objects retained after semantic and geometric filtering. Each attribute is kept, merged, or removed according to `attribute_mapping.json`.

**Canonical attributes are deduplicated within each object**, preserving the order of their first occurrence. For example:

```text
Original:  ["small", "little", "tiny", "red"]
Canonical: ["small", "red"]
```

All three original `small`/`little`/`tiny` occurrences are still retained in the provenance field `original_attributes`; only one canonical `small` is present in `attributes`. An object may have an empty canonical attribute list.

This changes the meaning of post-cleaning attribute frequencies: **each canonical attribute is counted at most once per retained object**, rather than once per original annotation occurrence.

### 4.4 Relation Filtering

For each retained source object, the relation name is resolved using `relation_mapping.json`. A relation is retained only if the canonical name is allowed **and** its target object ID still exists in the retained object dictionary. Relations with removed targets are discarded, preventing dangling references.

Relation instances are **not deduplicated**. The original name of each retained relation is preserved. An object may have no retained relations.

### 4.5 Empty Graphs and Validation

Images without retained objects are removed from the cleaned annotation split. Before writing the result, the script validates canonical labels, retained original-label fields, unique canonical attributes per object, in-frame positive bounding boxes, and relation targets. It also checks consistency identities among preprocessing counters.

These checks establish specified structural and geometric properties; they do not validate semantic correctness, class balance, or the availability/readability of image files.

## 5. Original-Label Preservation

The cleaned JSON retains the ordinary GQA fields used by downstream scripts, while adding provenance fields **only for retained annotations**:

| Annotation | Canonical field | Preserved source field |
|---|---|---|
| Object label | `name` | `original_name` |
| Object attributes | `attributes` | `original_attributes` |
| Retained relation label | `name` | `original_name` |

Illustrative object fragment (other fields such as coordinates are omitted):

```json
{
  "name": "person",
  "original_name": "man",
  "attributes": ["small", "red"],
  "original_attributes": ["small", "little", "red"],
  "relations": [
    {
      "name": "next to",
      "original_name": "beside",
      "object": "42"
    }
  ]
}
```

The usual `name`, `attributes`, and `relations` fields continue to expose the **cleaned/canonical values**. The additional fields make it possible to inspect the original labels and attribute occurrences of surviving objects and relations without changing how `gqa_analysis` reads annotations.

**Scope of preservation:** annotations removed by filtering are not copied into the cleaned graphs; they remain available in the original source JSON. The pipeline does not generate a separate per-annotation removal log. The cleaning script rejects source records that already contain its reserved provenance fields rather than silently overwriting them. Original bounding-box coordinates of clipped objects are not separately preserved in the cleaned records.

## 6. Running the Cleaning Pipeline

First generate a **separate test output** to avoid overwriting the existing cleaned dataset:

```powershell
python scripts/gqa_cleaning/prepare_gqa_cleaning.py --input-dir data/gqa/annotations --output-dir data/gqa_cleaned_test
```

After validating the test output, generate the intended cleaned dataset:

```powershell
python scripts/gqa_cleaning/prepare_gqa_cleaning.py --input-dir data/gqa/annotations --output-dir data/gqa_cleaned
```

**Caution:** existing generated files for the requested splits can be overwritten. Back up any output that must be retained. The script rejects overlapping input and output directories but does not make the entire multi-split run atomic.

| Argument | Default | Description |
|---|---|---|
| `--input-dir` | Required | Directory containing `<split>_sceneGraphs.json`. |
| `--output-dir` | Required | Destination root for cleaned files. |
| `--config-dir` | `scripts/gqa_cleaning/config/` | Directory containing the three mapping JSON files. |
| `--splits` | `train val` | Space-separated split names to process. |

Example with explicit configuration and splits:

```powershell
python scripts/gqa_cleaning/prepare_gqa_cleaning.py --input-dir data/gqa/annotations --output-dir data/gqa_cleaned_test --config-dir scripts/gqa_cleaning/config --splits train val
```

One source split and its filtered output are held in memory during processing; large GQA JSON files therefore require sufficient RAM and disk space.

### 6.1 Generated Files

| Output | Content |
|---|---|
| `annotations/<split>_sceneGraphs.json` | Cleaned scene graphs, canonical labels, retained original labels, and clipped boxes. |
| `image_ids/<split>.txt` | One image ID per retained scene graph. |
| `preprocessing_stats.json` | Per-split filtering counters and validation results. |

### 6.2 Understanding `preprocessing_stats.json`

| Counter | Meaning |
|---|---|
| `images_before`, `images_after` | Input and retained scene-graph counts. |
| `images_removed_without_selected_objects` | Images discarded because no usable object survived. |
| `objects_before`, `objects_after` | Original and retained object instances. |
| `objects_selected_semantic`, `objects_removed_semantic` | Results of object-label selection. |
| `bbox_valid`, `bbox_clippable`, `bbox_invalid` | Geometric classifications of semantically selected objects. |
| `objects_removed_invalid_bbox` | Selected objects discarded due to invalid boxes. |
| `attributes_before`, `relations_before` | Occurrences on all original objects. |
| `attributes_on_retained_objects_before`, `relations_on_retained_objects_before` | Occurrences on surviving objects, before label filtering. |
| `attributes_after`, `relations_after` | Canonical occurrences written to output. |
| `attributes_removed_duplicate_after_mapping` | Accepted attribute occurrences omitted because their canonical label was already present on that object. |
| `relations_removed_missing_target` | Accepted relations discarded because the target object was not retained. |
| `<kind>_keep`, `<kind>_merge`, `<kind>_remove_explicit`, `<kind>_remove_unlisted`, `<kind>_invalid_label` | Label-resolution counters for objects, attributes, or relations. |
| `validation` | Output validation result and checked annotation counts. |

**Counter scopes matter.** Attribute and relation mapping counters are computed only for retained objects. Accepted attribute occurrences may be deduplicated afterward; therefore `attribute_keep + attribute_merge` need not equal `attributes_after`. Missing keys in the counter JSON represent operations that did not occur, not necessarily omitted checks.

## 7. Optional Image Extraction: `extract_cleaned_gqa_images.py`

The image extraction script copies source images referenced by the cleaned `image_ids/<split>.txt` lists into `<cleaned-dir>/images/`. This is **optional**: cleaned annotations can continue to reference `data/gqa/images/` directly. The script does not edit source images or annotations.

Start with a dry run:

```powershell
python scripts/gqa_cleaning/extract_cleaned_gqa_images.py --images-dir data/gqa/images --cleaned-dir data/gqa_cleaned --dry-run
```

Copy images:

```powershell
python scripts/gqa_cleaning/extract_cleaned_gqa_images.py --images-dir data/gqa/images --cleaned-dir data/gqa_cleaned
```

Overwrite copied files **with matching filenames** when needed:

```powershell
python scripts/gqa_cleaning/extract_cleaned_gqa_images.py --images-dir data/gqa/images --cleaned-dir data/gqa_cleaned --overwrite
```

| Argument | Default | Description |
|---|---|---|
| `--images-dir` | Required | Directory containing the original images. |
| `--cleaned-dir` | Required | Cleaned dataset root containing `image_ids/`. |
| `--splits` | `train val` | Space-separated ID lists used to select files. |
| `--dry-run` | Off | Inspect available/missing images without copying files or writing a report. |
| `--overwrite` | Off | Replace an existing copied image only when its filename matches the selected source. |

Supported file extensions are `.jpg`, `.jpeg`, and `.png`, matched case-insensitively. The scan is not recursive. If several source images share the same stem, the selection priority is `.jpg`, then `.jpeg`, then `.png`, with a deterministic filename tie-break.

Each distinct image ID is processed once across the requested splits. Repeated IDs within lists, multiple source files for an ID, missing source images, and conflicting filenames already in the destination are reported. A destination conflict (e.g., `123.png` already exists when `123.jpg` is selected) is **not** resolved or overwritten automatically. Existing copies with matching names are skipped unless `--overwrite` is supplied.

On a non-dry run, the script writes `image_copy_report.json` under the cleaned directory. It reports counts and lists of identified anomalies; images missing from the source are not created, and the corresponding scene graphs/ID lists are not changed. The script does not validate image decodability or content equality.

## 8. Recommended Validation Workflow

After cleaning, use the read-only scripts in `scripts/gqa_analysis/` to inspect the generated dataset. The following commands target **`data/gqa_cleaned`**, not `data/gqa_cleaned_test`:

```powershell
python scripts/gqa_analysis/check_gqa_images.py --dataset cleaned
python scripts/gqa_analysis/analyze_gqa_bboxes.py --dataset cleaned
python scripts/gqa_analysis/inspect_gqa.py --dataset cleaned
python scripts/gqa_analysis/plot_frequencies.py --dataset cleaned
python scripts/gqa_analysis/visualize_gqa_scenegraphs.py --dataset cleaned
```

Before replacing the production cleaned dataset, inspect the test output independently or adapt the analysis input paths for the test directory. Do not assume that `--dataset cleaned` automatically points to the test directory.

Check that the generated lists match the cleaned scene-graph IDs, that train and validation do not overlap unexpectedly, that source images are available, and that **all output boxes are classified as `valid`**. Confirm that canonical attributes are unique within each retained object and that added provenance fields correspond to source labels. Compare `preprocessing_stats.json` with the previous run: the new attribute-deduplication behavior can decrease `attributes_after` even with unchanged mapping files.

The original image collection may contain files outside the analyzed scene-graph splits; these are not automatically erroneous and should not be deleted solely for that reason.

## 9. Interpretation and Reproducibility

- **Closed vocabularies:** 100 canonical labels are permitted for each annotation type, but the procedure does not guarantee their presence or sufficient frequency in every split.
- **Manual mapping decisions:** label selection and merging define the experimental dataset. The mapping files record final actions, not the full rationale for each decision.
- **Original-label provenance:** available for retained annotations; removed records are accessible in the source dataset rather than copied into the cleaned output.
- **Attribute frequencies:** the cleaned data contain at most one occurrence of a given canonical attribute per object. This should be considered when comparing original and cleaned frequency tables.
- **Relations:** dangling targets are excluded, but repeated canonical relation instances are not automatically deduplicated.
- **Geometric validity:** clipping ensures positive-area in-frame output boxes under the implemented checks, but does not guarantee correct localization or annotation semantics.
- **Image availability:** annotation cleaning does not check whether a referenced image exists; optional image extraction reports missing IDs without removing annotations.
- **Output management:** rerunning scripts may overwrite generated files. Keep a copy of previous results when comparing preprocessing revisions.
- **Experiment records:** store the mapping versions, script revisions, commands, source dataset version, and `preprocessing_stats.json` used in thesis experiments.

The pipeline prepares a **filtered annotation dataset**, not object crops, probe-ready features, balanced train/validation subsets, or trained probing models.
