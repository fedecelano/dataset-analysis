# GQA Dataset Analysis and Preparation

This repository contains scripts to **inspect, validate, and preprocess GQA scene-graph annotations** for later experiments on visual representations and layer-wise linear probing. It focuses on dataset preparation rather than model training.

The workflow has two components:

1. **GQA analysis:** descriptive statistics, qualitative visualization, bounding-box analysis, image/annotation integrity checks, and label-frequency plots.
2. **GQA preprocessing:** filtering source annotations into canonical vocabularies of 100 object classes, 100 attributes, and 100 relations, including bounding-box clipping and relation-integrity checks.

The original GQA annotations and image files are treated as source data and are **not modified** by these scripts.

## 1. Repository layout

```text
dataset-analysis/
├── .vscode/                         # Local editor configuration, if present
├── data/
│   ├── gqa/
│   │   ├── annotations/              # Original train/val scene graph JSON files
│   │   └── images/                   # Original images, shared by both datasets
│   └── gqa_cleaned/
│       ├── annotations/              # Filtered train/val scene graph JSON files
│       ├── image_ids/                # train.txt and val.txt
│       └── preprocessing_stats.json  # Generated preprocessing report
├── outputs/
│   └── gqa_analysis/
│       ├── original/
│       │   ├── tables/
│       │   ├── plots/
│       │   └── qualitative_examples/
│       └── cleaned/
│           ├── tables/
│           ├── plots/
│           └── qualitative_examples/
├── scripts/
│   ├── gqa_analysis/
│   │   ├── inspect_gqa.py
│   │   ├── analyze_gqa_bboxes.py
│   │   ├── check_gqa_images.py
│   │   ├── plot_frequencies.py
│   │   ├── visualize_gqa_scenegraphs.py
│   │   └── README.md
│   └── gqa_probing/
│       ├── prepare_gqa_probing.py
│       ├── README.md
│       └── config/
│           ├── object_mapping.json
│           ├── attribute_mapping.json
│           └── relation_mapping.json
└── README.md
```

The script filename `visualize_gqa_scenegraphs.py` follows the current local file listing. If your actual file is named `visualize_gqa_scene_graphs.py`, use that spelling in the commands below.

The `preprocessing_stats.json` file is generated at the root of `data/gqa_cleaned/` and may not appear in a directory-only `tree` listing.

## 2. Data inputs and outputs

### Source GQA data

Place the original scene-graph files in:

```text
data/gqa/annotations/train_sceneGraphs.json
data/gqa/annotations/val_sceneGraphs.json
```

Place image files in `data/gqa/images/`. The scripts expect GQA scene graphs keyed by image ID, with object records containing object names, attributes, relations, and bounding-box coordinates (`x`, `y`, `w`, `h`), plus per-image `width` and `height`.

**The image directory is shared** by the original and cleaned datasets. The cleaned dataset stores only filtered annotations, image-ID lists, and preprocessing statistics; it does not duplicate the image files.

Images with no matching train/validation scene graph are not automatically an error: the image collection also includes images outside those two annotated subsets. This repository does not delete them.

### Generated analysis files

Analysis output is grouped by the dataset version:

- `outputs/gqa_analysis/original/` for the source scene graphs.
- `outputs/gqa_analysis/cleaned/` for the filtered scene graphs.

Each has `tables/`, `plots/`, and `qualitative_examples/` directories. The analysis scripts only read annotations/images and write derived reports and figures.

## 3. Requirements

A **Python 3.10+** environment is a suitable baseline for the supplied scripts. External dependencies used by the analysis tools are:

```powershell
python -m pip install numpy pandas matplotlib Pillow
```

`prepare_gqa_probing.py` itself uses only Python's standard library. This repository currently does not include a pinned requirements file or lockfile; package versions should be recorded separately for strict environment reproducibility.

Run the examples below from the **repository root**. On Windows PowerShell, activate the desired Python environment first if applicable.

## 4. Workflow

### Step 1 — Analyze the original dataset

```powershell
python scripts/gqa_analysis/inspect_gqa.py --dataset original
python scripts/gqa_analysis/analyze_gqa_bboxes.py --dataset original
python scripts/gqa_analysis/check_gqa_images.py --dataset original
python scripts/gqa_analysis/visualize_gqa_scenegraphs.py --dataset original
python scripts/gqa_analysis/plot_frequencies.py --dataset original
```

`inspect_gqa.py` produces frequency CSV tables that are consumed by `plot_frequencies.py`; therefore run `inspect_gqa.py` first. The other analysis scripts can run independently when their input files exist.

| Script | Purpose | Main outputs |
|---|---|---|
| `inspect_gqa.py` | Counts of images, objects, attributes, relations; vocabulary differences; per-label frequencies | Six summary/frequency CSV files in `tables/` |
| `analyze_gqa_bboxes.py` | Bounding-box validity and size distributions | `bbox_quality_summary.csv`, `bbox_size_distribution.csv`, `bbox_size_distribution.png` |
| `check_gqa_images.py` | Image-ID coverage, missing images, split overlap, duplicate IDs; cleaned ID-list consistency | `image_integrity_summary.csv`, `image_integrity_issues.csv` |
| `visualize_gqa_scenegraphs.py` | Qualitative train/validation examples with object boxes, attributes, and relations | An overview and eight detailed PNG figures in `qualitative_examples/` |
| `plot_frequencies.py` | Independent train/validation rank-frequency curves and top-N relative-frequency comparisons | `frequency_distributions_combined.png`, `top_labels_relative_combined.png` |

Both frequency plots represent object, attribute, and relation statistics. For the top-N plots, the displayed labels are selected by **training** frequency and the percentage values are calculated separately in each split. Rank-frequency curves, by contrast, are **independently ranked** for train and validation.

For an optional number of top labels other than 30:

```powershell
python scripts/gqa_analysis/plot_frequencies.py --dataset original --top-n 20
```

### Step 2 — Prepare a cleaned scene-graph dataset

The preprocessing uses the three reviewed mapping files in `scripts/gqa_probing/config/`. The object/attribute/relation vocabularies have 100 canonical labels each. The mapping rules are `keep`, `merge`, and `remove`; unlisted labels are also dropped.

The transformation order is:

1. Resolve object labels against the object vocabulary.
2. Validate boxes on the selected objects; clip partly out-of-frame boxes and remove unusable objects.
3. Filter/merge attributes on the surviving objects.
4. Filter/merge relations and discard edges pointing to removed target objects.
5. Drop scene graphs without retained objects and validate output labels, geometry, and relation targets.
6. Export filtered scene graphs, image-ID lists, and preprocessing statistics.

**Recommended first run:** write to a test destination, avoiding accidental replacement of an existing cleaned dataset.

```powershell
python scripts/gqa_probing/prepare_gqa_probing.py --input-dir data/gqa/annotations --output-dir data/gqa_cleaned_test
```

When the output has been reviewed, generate the intended cleaned directory:

```powershell
python scripts/gqa_probing/prepare_gqa_probing.py --input-dir data/gqa/annotations --output-dir data/gqa_cleaned
```

This second command can overwrite generated files in `data/gqa_cleaned/`; preserve any prior version you need before running it. The source JSON files and images remain unchanged.

For full details on label selection, geometry rules, statistics, CLI arguments, and validation, see **[`scripts/gqa_probing/README.md`](scripts/gqa_probing/README.md)**.

### Step 3 — Analyze and verify the cleaned dataset

```powershell
python scripts/gqa_analysis/check_gqa_images.py --dataset cleaned
python scripts/gqa_analysis/analyze_gqa_bboxes.py --dataset cleaned
python scripts/gqa_analysis/inspect_gqa.py --dataset cleaned
python scripts/gqa_analysis/plot_frequencies.py --dataset cleaned
python scripts/gqa_analysis/visualize_gqa_scenegraphs.py --dataset cleaned
```

Recommended checks include:

- Every cleaned scene graph has a corresponding file in the shared image directory.
- Train and validation image IDs do not overlap, and cleaned `image_ids/train.txt` / `val.txt` match the respective scene-graph keys.
- Every retained bounding box is `valid` (no remaining `clippable` or `invalid` boxes).
- All retained object, attribute, and relation labels belong to their canonical vocabularies.
- Class frequency and coverage are inspected **after** preprocessing, not assumed from the selected label lists alone.

Refer to **[`scripts/gqa_analysis/README.md`](scripts/gqa_analysis/README.md)** for per-script methodology, complete output tables, and interpretation notes.

## 5. Dataset preparation summary

For the revised preprocessing run reviewed during development:

| Metric | Train | Validation | Total |
|---|---:|---:|---:|
| Images retained | 73,347 | 10,430 | 83,777 |
| Objects retained | 743,909 | 105,229 | 849,138 |
| Attributes retained | 342,783 | 49,026 | 391,809 |
| Relations retained | 1,739,075 | 244,281 | 1,983,356 |
| Bounding boxes clipped | 6,787 | 926 | 7,713 |
| Invalid bounding boxes removed | 105 | 58 | 163 |

These are reference statistics from one processed dataset, not constants or assertions made by the scripts. Consult `data/gqa_cleaned/preprocessing_stats.json` for statistics from the currently generated output.

## 6. Design choices and limitations

- **No physical image removal:** the cleaned dataset uses image-ID lists to select its subset from the original image directory. Unreferenced images are retained.
- **Semantic filtering before geometry:** only objects selected by `object_mapping.json` are checked for clipping/removal.
- **Positive-area overlap required:** boxes with non-positive sizes, non-finite coordinates, or no positive intersection with the image are excluded. Partially visible boxes can be recovered by clipping, including those with negative `x`/`y`.
- **Relations remain referentially valid:** edges to removed objects are discarded.
- **Manual vocabulary selection:** the final mappings preserve canonicalization and exclusion decisions; they do not completely document the judgment involved in selecting the labels.
- **No feature extraction or probing training:** this repository currently prepares the annotated data for subsequent work on visual encoders, not the probes themselves.
- **No guarantee of class balance:** keeping a vocabulary label does not guarantee adequate train or validation examples for that label.

## 7. Documentation

- [`scripts/gqa_analysis/README.md`](scripts/gqa_analysis/README.md): individual analysis scripts, figures, output CSVs, and methodological limitations.
- [`scripts/gqa_probing/README.md`](scripts/gqa_probing/README.md): vocabulary mapping, preprocessing order, bounding-box clipping, statistics, validation, and command-line interface.
