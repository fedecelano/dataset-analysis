#!/usr/bin/env python3
"""Plot label-frequency distributions from inspect_gqa.py CSV outputs.

For original and cleaned datasets, generate the existing two figures comparing
train and validation. For probing_5k, compare train, validation, and test and
add two analogous figures for GQA source labels retained in the prepared data.

Counts are annotation occurrences, not image-level prevalence. Original-label
frequencies reflect the provenance fields exported by inspect_gqa.py, not all
annotations in the full source GQA dataset.

Examples (run from the repository root):
    python scripts/gqa_analysis/plot_frequencies.py --dataset original
    python scripts/gqa_analysis/plot_frequencies.py --dataset cleaned --top-n 30
    python scripts/gqa_analysis/plot_frequencies.py --dataset probing_5k
    python scripts/gqa_analysis/plot_frequencies.py --dataset probing_5k --input-dir path/to/tables --output-dir path/to/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = PROJECT_ROOT / "outputs" / "gqa_analysis"
DATASET_SPLITS = {
    "original": ("train", "val"),
    "cleaned": ("train", "val"),
    "probing_5k": ("train", "val", "test"),
}
SPLIT_LABELS = {"train": "Train", "val": "Validation", "test": "Test"}

FREQUENCY_TABLES = (
    ("class_frequencies.csv", "Object Classes"),
    ("attribute_frequencies.csv", "Attributes"),
    ("relation_frequencies.csv", "Relations"),
)
ORIGINAL_FREQUENCY_TABLES = (
    ("original_class_frequencies.csv", "Original Object Labels"),
    ("original_attribute_frequencies.csv", "Original Attribute Labels"),
    ("original_relation_frequencies.csv", "Original Relation Labels"),
)


def parse_args() -> argparse.Namespace:
    """Parse dataset selection, table/output overrides, and plotting options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=tuple(DATASET_SPLITS), default="original",
        help="Dataset to plot (default: original).",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=None,
        help="Override the input tables directory (normally outputs/gqa_analysis/<dataset>/tables).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override the output plots directory (normally outputs/gqa_analysis/<dataset>/plots).",
    )
    parser.add_argument(
        "--top-n", type=int, default=30,
        help="Number of most frequent training labels for bar plots (default: 30).",
    )
    args = parser.parse_args()
    if args.top_n < 1:
        parser.error("--top-n must be a positive integer")
    return args


def load_frequency_tables(
    input_dir: Path,
    splits: tuple[str, ...] = ("train", "val"),
    definitions: tuple[tuple[str, str], ...] = FREQUENCY_TABLES,
) -> list[tuple[str, pd.DataFrame]]:
    """Read frequency CSVs and validate label uniqueness and integer counts."""
    required_columns = {"label", *(f"{split}_count" for split in splits)}
    tables = []
    for filename, title in definitions:
        path = input_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing input file: {path}\n"
                "Run inspect_gqa.py with the same --dataset and matching output directory first."
            )
        try:
            df = pd.read_csv(path, dtype={"label": "string"})
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"Empty CSV file: {path}") from exc
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"{path}: missing columns {sorted(missing_columns)}")
        if df["label"].isna().any() or (df["label"].str.len() == 0).any():
            raise ValueError(f"{path}: missing/empty label values")
        if df["label"].duplicated().any():
            raise ValueError(f"{path}: duplicate labels found")
        for split in splits:
            column = f"{split}_count"
            values = pd.to_numeric(df[column], errors="coerce")
            if (
                values.isna().any()
                or not np.isfinite(values.to_numpy(dtype=float)).all()
                or (values < 0).any()
                or (values % 1 != 0).any()
            ):
                raise ValueError(f"{path}: {column} must contain nonnegative integer counts")
            try:
                df[column] = values.astype("int64")
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}: {column} cannot be represented as int64") from exc
        tables.append((title, df))
    return tables


def plot_frequency_distributions(
    tables: list[tuple[str, pd.DataFrame]],
    output_dir: Path,
    splits: tuple[str, ...] = ("train", "val"),
    *,
    filename: str = "frequency_distributions_combined.png",
    title_prefix: str = "GQA Label",
) -> Path:
    """Plot independent rank-frequency curves, excluding zero values on log axis."""
    fig, axes = plt.subplots(1, len(tables), figsize=(8 * len(tables), 8), squeeze=False)
    try:
        for ax, (title, df) in zip(axes[0], tables):
            has_occurrences = False
            for split in splits:
                column = f"{split}_count"
                frequencies = np.sort(
                    df.loc[df[column] > 0, column].to_numpy(dtype=np.int64)
                )[::-1]
                if frequencies.size:
                    has_occurrences = True
                    ranks = np.arange(1, frequencies.size + 1)
                    ax.plot(ranks, frequencies, label=SPLIT_LABELS[split])
                else:
                    ax.plot([], [], label=f"{SPLIT_LABELS[split]} (no occurrences)")
            ax.set_yscale("log")
            ax.set_xlabel("Label rank (independently sorted per split)")
            ax.set_ylabel("Number of occurrences")
            ax.set_title(title)
            ax.legend()
            ax.grid(axis="y", alpha=0.25)
            if not has_occurrences:
                ax.text(0.5, 0.5, "No annotation occurrences", ha="center", va="center", transform=ax.transAxes)
        fig.suptitle(f"{title_prefix} Rank-Frequency Distributions", fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)
    return output_path


def plot_top_labels_relative(
    tables: list[tuple[str, pd.DataFrame]],
    output_dir: Path,
    top_n: int = 30,
    splits: tuple[str, ...] = ("train", "val"),
    *,
    filename: str = "top_labels_relative_combined.png",
    title_prefix: str = "GQA",
) -> Path:
    """Compare top-N training labels by annotation-relative percentage.

    Denominator for each split is ALL annotations of the plotted label type,
    not the top-N total or the number of objects/images. A zero denominator
    gives a zero-length bar with an explicit notice (percentage undefined).
    """
    fig, axes = plt.subplots(1, len(tables), figsize=(8 * len(tables), 10), squeeze=False)
    try:
        for ax, (title, df) in zip(axes[0], tables):
            df = df.copy()
            totals = {}
            for split in splits:
                column = f"{split}_count"
                total = int(df[column].sum())
                totals[split] = total
                df[f"{split}_percentage"] = (
                    df[column] / total * 100 if total > 0 else 0.0
                )
            selected = (
                df.loc[df["train_count"] > 0]
                .sort_values(["train_count", "label"], ascending=[False, True])
                .head(top_n)
                .iloc[::-1]
            )
            y = np.arange(len(selected))
            bar_height = 0.8 / len(splits)
            for index, split in enumerate(splits):
                offset = (index - (len(splits) - 1) / 2) * bar_height
                ax.barh(
                    y + offset,
                    selected[f"{split}_percentage"],
                    height=bar_height,
                    label=SPLIT_LABELS[split],
                )
            ax.set_yticks(y)
            ax.set_yticklabels(selected["label"].astype(str))
            ax.set_xlabel("Percentage of annotations within split (%)")
            ax.set_title(title)
            ax.legend()
            if selected.empty:
                ax.text(0.5, 0.5, "No labels in training split", ha="center", va="center", transform=ax.transAxes)
            else:
                empty_splits = [split for split in splits if totals[split] == 0]
                if empty_splits:
                    labels = ", ".join(SPLIT_LABELS[s] for s in empty_splits)
                    ax.text(
                        0.02, 0.02,
                        f"No annotations in: {labels}\nCorresponding percentages are undefined.",
                        transform=ax.transAxes, fontsize=9, va="bottom",
                    )
        fig.suptitle(f"{title_prefix} Top-{top_n} Training Labels: Relative Frequencies", fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    splits = DATASET_SPLITS[args.dataset]
    input_dir = args.input_dir if args.input_dir is not None else ANALYSIS_ROOT / args.dataset / "tables"
    output_dir = args.output_dir if args.output_dir is not None else ANALYSIS_ROOT / args.dataset / "plots"

    tables = load_frequency_tables(input_dir, splits)
    original_tables = (
        load_frequency_tables(input_dir, splits, ORIGINAL_FREQUENCY_TABLES)
        if args.dataset == "probing_5k" else None
    )
    # Validate all inputs before producing any files.
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {args.dataset}")
    path = plot_frequency_distributions(tables, output_dir, splits)
    print(f"Saved: {path}")
    path = plot_top_labels_relative(tables, output_dir, args.top_n, splits)
    print(f"Saved: {path}")
    if original_tables is not None:
        path = plot_frequency_distributions(
            original_tables, output_dir, splits,
            filename="original_frequency_distributions_combined.png",
            title_prefix="GQA Original Label",
        )
        print(f"Saved: {path}")
        path = plot_top_labels_relative(
            original_tables, output_dir, args.top_n, splits,
            filename="original_top_labels_relative_combined.png",
            title_prefix="GQA Original Labels",
        )
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
