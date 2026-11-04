
"""Analyze bounding-box validity and relative size in GQA scene graphs.

Analyze the original or cleaned GQA train and validation splits without
modifying the source annotations.

Bounding boxes are classified as:
- valid: positive dimensions, finite geometry, entirely inside the image;
- clippable: positive dimensions, finite geometry, partially outside the
  image but with a strictly positive intersection area;
- invalid: non-positive dimensions, non-finite geometry, invalid image
  dimensions, or no positive-area intersection with the image.

The size distribution uses the clipped area of valid and clippable
bounding boxes only.

Outputs are saved to:
    outputs/gqa_analysis/<dataset>/tables/
    outputs/gqa_analysis/<dataset>/plots/

Usage:
    python scripts/gqa_analysis/analyze_gqa_bboxes.py --dataset original
    python scripts/gqa_analysis/analyze_gqa_bboxes.py --dataset cleaned
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIRS = {
    "original": PROJECT_ROOT / "data" / "gqa" / "annotations",
    "cleaned": PROJECT_ROOT / "data" / "gqa_cleaned" / "annotations",
}

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "gqa_analysis"

BBOX_COLUMNS = (
    "split",
    "image_width",
    "image_height",
    "x",
    "y",
    "w",
    "h",
)

GEOMETRY_COLUMNS = (
    "image_width",
    "image_height",
    "x",
    "y",
    "w",
    "h",
)

BBOX_STATUSES = ("valid", "clippable", "invalid")

SIZE_BINS = [
    0.0,
    0.001,
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    float("inf"),
]

SIZE_LABELS = [
    "<0.1%",
    "0.1-1%",
    "1-5%",
    "5-10%",
    "10-25%",
    "25-50%",
    ">50%",
]


def parse_args() -> argparse.Namespace:
    """Parse the dataset version to analyze."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_DIRS),
        default="original",
        help="Dataset version to analyze (default: original).",
    )

    return parser.parse_args()


def load_json(path: Path) -> dict:
    """Load and validate a GQA scene-graph JSON file."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a dictionary of scene graphs in: {path}"
        )

    return data


def extract_bbox_data(data: dict, split: str) -> pd.DataFrame:
    """Extract one bounding-box record per annotated object."""
    rows = []

    for image_data in data.values():
        image_width = image_data["width"]
        image_height = image_data["height"]

        for obj in image_data["objects"].values():
            rows.append({
                "split": split,
                "image_width": image_width,
                "image_height": image_height,
                "x": obj["x"],
                "y": obj["y"],
                "w": obj["w"],
                "h": obj["h"],
            })

    return pd.DataFrame(rows, columns=BBOX_COLUMNS)


def add_bbox_status(df: pd.DataFrame) -> pd.DataFrame:
    """Classify bounding boxes according to their image intersection.

    All geometric values and calculated box endpoints must be finite.
    A partially external box is clippable only if its intersection
    with the image has strictly positive width and height.
    """
    df = df.copy()

    if df.empty:
        df["bbox_status"] = pd.Series(dtype="object")
        return df

    for column in GEOMETRY_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    finite_geometry = np.isfinite(
        df[list(GEOMETRY_COLUMNS)].to_numpy(dtype=float)
    ).all(axis=1)

    positive_image = (
        (df["image_width"] > 0)
        & (df["image_height"] > 0)
    )

    positive_box = (
        (df["w"] > 0)
        & (df["h"] > 0)
    )

    # Check calculated endpoints as well as the original coordinates.
    with np.errstate(over="ignore", invalid="ignore"):
        box_right = df["x"] + df["w"]
        box_bottom = df["y"] + df["h"]

    finite_endpoints = (
        np.isfinite(box_right)
        & np.isfinite(box_bottom)
    )

    x1 = df["x"].clip(lower=0)
    y1 = df["y"].clip(lower=0)

    x2 = np.minimum(box_right, df["image_width"])
    y2 = np.minimum(box_bottom, df["image_height"])

    intersects = (
        (x2 > x1)
        & (y2 > y1)
    )

    usable = (
        finite_geometry
        & finite_endpoints
        & positive_image
        & positive_box
        & intersects
    )

    inside = (
        (df["x"] >= 0)
        & (df["y"] >= 0)
        & (box_right <= df["image_width"])
        & (box_bottom <= df["image_height"])
    )

    df["bbox_status"] = "invalid"
    df.loc[usable & ~inside, "bbox_status"] = "clippable"
    df.loc[usable & inside, "bbox_status"] = "valid"

    return df


def add_clipped_area(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate clipped bounding-box area relative to image area.

    Only valid and clippable boxes receive a relative area.
    Invalid boxes retain NaN.
    """
    df = df.copy()
    df["clipped_relative_area"] = np.nan

    usable = df["bbox_status"].isin(("valid", "clippable"))

    if not usable.any():
        return df

    boxes = df.loc[usable]

    x1 = boxes["x"].clip(lower=0)
    y1 = boxes["y"].clip(lower=0)

    x2 = np.minimum(
        boxes["x"] + boxes["w"],
        boxes["image_width"],
    )
    y2 = np.minimum(
        boxes["y"] + boxes["h"],
        boxes["image_height"],
    )

    clipped_width = (x2 - x1).clip(lower=0)
    clipped_height = (y2 - y1).clip(lower=0)

    # Normalize dimensions before multiplication to avoid overflow
    # in intermediate area calculations.
    relative_area = (
        (clipped_width / boxes["image_width"])
        * (clipped_height / boxes["image_height"])
    )

    if (
        not np.isfinite(relative_area.to_numpy(dtype=float)).all()
        or (relative_area <= 0).any()
        or (relative_area > 1).any()
    ):
        raise ValueError(
            "Invalid relative area calculated for usable bounding boxes."
        )

    df.loc[usable, "clipped_relative_area"] = relative_area

    return df


def create_quality_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    """Count bounding-box statuses and percentages within each split."""
    rows = []

    for split, df in (
        ("train", train_df),
        ("validation", val_df),
    ):
        counts = df["bbox_status"].value_counts()
        total = len(df)

        for status in BBOX_STATUSES:
            count = int(counts.get(status, 0))

            rows.append({
                "split": split,
                "status": status,
                "count": count,
                "percentage": (
                    round(100 * count / total, 3)
                    if total
                    else 0.0
                ),
            })

    return pd.DataFrame(
        rows,
        columns=["split", "status", "count", "percentage"],
    )


def create_size_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    """Bin relative areas of usable boxes and calculate percentages.

    Percentages are relative to valid and clippable boxes only.
    Verify that every usable box belongs to exactly one size bucket.
    """
    rows = []

    for split, df in (
        ("train", train_df),
        ("validation", val_df),
    ):
        usable = df.loc[
            df["bbox_status"].isin(("valid", "clippable"))
        ]

        bins = pd.cut(
            usable["clipped_relative_area"],
            bins=SIZE_BINS,
            labels=SIZE_LABELS,
            right=False,
        )

        if bins.isna().any():
            raise ValueError(
                f"{split}: some usable bounding boxes could not "
                "be assigned to a size bucket."
            )

        counts = bins.value_counts(sort=False)
        total = len(usable)

        if int(counts.sum()) != total:
            raise ValueError(
                f"{split}: size distribution does not account "
                "for all usable bounding boxes."
            )

        for label in SIZE_LABELS:
            count = int(counts.get(label, 0))

            rows.append({
                "split": split,
                "size_bucket": label,
                "count": count,
                "percentage": (
                    round(100 * count / total, 3)
                    if total
                    else 0.0
                ),
            })

    return pd.DataFrame(
        rows,
        columns=["split", "size_bucket", "count", "percentage"],
    )


def plot_size_distribution(
    size_distribution: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot bounding-box relative-size distributions by split."""
    train = size_distribution[
        size_distribution["split"] == "train"
    ]
    validation = size_distribution[
        size_distribution["split"] == "validation"
    ]

    x = np.arange(len(SIZE_LABELS))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        ax.bar(
            x - width / 2,
            train["percentage"],
            width=width,
            label="Train",
        )

        ax.bar(
            x + width / 2,
            validation["percentage"],
            width=width,
            label="Validation",
        )

        ax.set_xticks(x, SIZE_LABELS, rotation=30)
        ax.set_xlabel("Bounding box area relative to image")
        ax.set_ylabel("Percentage of usable bounding boxes")
        ax.set_title("GQA Bounding Box Size Distribution")
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

    finally:
        plt.close(fig)


def main() -> None:
    """Analyze bounding boxes and export summary tables and figures."""
    args = parse_args()

    annotation_dir = DATASET_DIRS[args.dataset]
    output_dir = OUTPUT_ROOT / args.dataset

    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"

    print(f"Loading {args.dataset} GQA scene graphs...")

    train_data = load_json(
        annotation_dir / "train_sceneGraphs.json"
    )
    val_data = load_json(
        annotation_dir / "val_sceneGraphs.json"
    )

    print("Extracting and classifying bounding boxes...")

    train_df = add_clipped_area(
        add_bbox_status(
            extract_bbox_data(train_data, "train")
        )
    )

    val_df = add_clipped_area(
        add_bbox_status(
            extract_bbox_data(val_data, "validation")
        )
    )

    quality_summary = create_quality_summary(train_df, val_df)
    size_distribution = create_size_distribution(train_df, val_df)

    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    quality_path = tables_dir / "bbox_quality_summary.csv"
    size_path = tables_dir / "bbox_size_distribution.csv"
    plot_path = plots_dir / "bbox_size_distribution.png"

    quality_summary.to_csv(quality_path, index=False)
    size_distribution.to_csv(size_path, index=False)

    plot_size_distribution(size_distribution, plot_path)

    for path in (quality_path, size_path, plot_path):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
