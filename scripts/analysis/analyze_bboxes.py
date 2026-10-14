import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

TRAIN_PATH = Path("data/gqa/annotations/train_sceneGraphs.json")
VAL_PATH = Path("data/gqa/annotations/val_sceneGraphs.json")

TABLES_DIR = Path("outputs/gqa_analysis/tables")
PLOTS_DIR = Path("outputs/gqa_analysis/plots")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Bounding boxes exceeding an image boundary by at most 5% of the
# corresponding image dimension are considered recoverable by clipping.
MAX_RELATIVE_OVERFLOW = 0.05

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


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_json(path):
    """Load a GQA scene graph JSON file."""

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Bounding-box extraction
# ---------------------------------------------------------------------

def extract_bbox_data(data, split):
    """
    Extract bounding-box information for every object in a GQA split.
    """

    rows = []

    for image_data in data.values():
        image_width = image_data["width"]
        image_height = image_data["height"]

        for obj in image_data["objects"].values():
            x = obj["x"]
            y = obj["y"]
            w = obj["w"]
            h = obj["h"]

            overflow_right = max(
                0,
                x + w - image_width,
            )

            overflow_bottom = max(
                0,
                y + h - image_height,
            )

            rows.append({
                "split": split,
                "image_width": image_width,
                "image_height": image_height,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "relative_overflow_right": (
                    overflow_right / image_width
                ),
                "relative_overflow_bottom": (
                    overflow_bottom / image_height
                ),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Bounding-box quality
# ---------------------------------------------------------------------

def add_bbox_status(df):
    """
    Classify each bounding box as:

    - valid: positive dimensions and fully inside the image
    - clippable: positive dimensions and small boundary overflow
    - severe_overflow: positive dimensions but excessive overflow
    - invalid: non-positive width or height
    """

    df = df.copy()

    valid_size = (
        (df["w"] > 0)
        & (df["h"] > 0)
    )

    inside_image = (
        (df["x"] >= 0)
        & (df["y"] >= 0)
        & (df["x"] + df["w"] <= df["image_width"])
        & (df["y"] + df["h"] <= df["image_height"])
    )

    max_relative_overflow = df[
        [
            "relative_overflow_right",
            "relative_overflow_bottom",
        ]
    ].max(axis=1)

    df["bbox_status"] = "invalid"

    df.loc[
        valid_size & inside_image,
        "bbox_status",
    ] = "valid"

    df.loc[
        valid_size
        & ~inside_image
        & (max_relative_overflow <= MAX_RELATIVE_OVERFLOW),
        "bbox_status",
    ] = "clippable"

    df.loc[
        valid_size
        & ~inside_image
        & (max_relative_overflow > MAX_RELATIVE_OVERFLOW),
        "bbox_status",
    ] = "severe_overflow"

    return df


# ---------------------------------------------------------------------
# Bounding-box clipping and area
# ---------------------------------------------------------------------

def add_clipped_area(df):
    """
    Clip valid and recoverable bounding boxes to image boundaries and
    compute their relative area.
    """

    df = df.copy()

    x1 = df["x"].clip(lower=0)
    y1 = df["y"].clip(lower=0)

    x2 = np.minimum(
        df["x"] + df["w"],
        df["image_width"],
    )

    y2 = np.minimum(
        df["y"] + df["h"],
        df["image_height"],
    )

    clipped_width = (x2 - x1).clip(lower=0)
    clipped_height = (y2 - y1).clip(lower=0)

    df["clipped_relative_area"] = (
        clipped_width
        * clipped_height
        / (
            df["image_width"]
            * df["image_height"]
        )
    )

    return df


# ---------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------

def create_quality_summary(train_df, val_df):
    """Create bounding-box quality statistics for both splits."""

    rows = []

    for split_name, df in [
        ("train", train_df),
        ("validation", val_df),
    ]:
        counts = df["bbox_status"].value_counts()

        for status in [
            "valid",
            "clippable",
            "severe_overflow",
            "invalid",
        ]:
            count = counts.get(status, 0)

            rows.append({
                "split": split_name,
                "status": status,
                "count": count,
                "percentage": round(
                    count / len(df) * 100,
                    3,
                ),
            })

    return pd.DataFrame(rows)


def create_size_distribution(train_df, val_df):
    """
    Compute bounding-box size distributions using clipped relative area.
    """

    rows = []

    for split_name, df in [
        ("train", train_df),
        ("validation", val_df),
    ]:
        usable = df[
            df["bbox_status"].isin(
                ["valid", "clippable"]
            )
        ].copy()

        usable["size_bucket"] = pd.cut(
            usable["clipped_relative_area"],
            bins=SIZE_BINS,
            labels=SIZE_LABELS,
            right=False,
        )

        counts = (
            usable["size_bucket"]
            .value_counts(sort=False)
        )

        percentages = (
            usable["size_bucket"]
            .value_counts(
                sort=False,
                normalize=True,
            )
            * 100
        )

        for label in SIZE_LABELS:
            rows.append({
                "split": split_name,
                "size_bucket": label,
                "count": counts.get(label, 0),
                "percentage": round(
                    percentages.get(label, 0),
                    3,
                ),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def plot_size_distribution(size_distribution):
    """Plot train and validation bounding-box size distributions."""

    train = size_distribution[
        size_distribution["split"] == "train"
    ]

    validation = size_distribution[
        size_distribution["split"] == "validation"
    ]

    x = np.arange(len(SIZE_LABELS))
    width = 0.4

    plt.figure(figsize=(10, 6))

    plt.bar(
        x - width / 2,
        train["percentage"],
        width=width,
        label="Train",
    )

    plt.bar(
        x + width / 2,
        validation["percentage"],
        width=width,
        label="Validation",
    )

    plt.xticks(
        x,
        SIZE_LABELS,
        rotation=30,
    )

    plt.xlabel(
        "Bounding box area relative to image"
    )

    plt.ylabel(
        "Percentage of bounding boxes"
    )

    plt.title(
        "GQA Bounding Box Size Distribution"
    )

    plt.legend()
    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "bbox_size_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading GQA scene graphs...")

    train_data = load_json(TRAIN_PATH)
    val_data = load_json(VAL_PATH)

    print("Extracting bounding boxes...")

    train_df = extract_bbox_data(
        train_data,
        "train",
    )

    val_df = extract_bbox_data(
        val_data,
        "validation",
    )

    train_df = add_bbox_status(train_df)
    val_df = add_bbox_status(val_df)

    train_df = add_clipped_area(train_df)
    val_df = add_clipped_area(val_df)

    quality_summary = create_quality_summary(
        train_df,
        val_df,
    )

    size_distribution = create_size_distribution(
        train_df,
        val_df,
    )

    quality_path = (
        TABLES_DIR
        / "bbox_quality_summary.csv"
    )

    size_path = (
        TABLES_DIR
        / "bbox_size_distribution.csv"
    )

    quality_summary.to_csv(
        quality_path,
        index=False,
    )

    size_distribution.to_csv(
        size_path,
        index=False,
    )

    plot_size_distribution(
        size_distribution
    )

    print(f"Saved: {quality_path}")
    print(f"Saved: {size_path}")


if __name__ == "__main__":
    main()