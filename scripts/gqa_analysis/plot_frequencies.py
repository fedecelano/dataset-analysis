import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

INPUT_DIR = Path("outputs/gqa_analysis/tables")
OUTPUT_DIR = Path("outputs/gqa_analysis/plots")


# ---------------------------------------------------------------------
# Plot configuration
# ---------------------------------------------------------------------

DATASETS = [
    ("class_frequencies.csv", "Object Classes"),
    ("attribute_frequencies.csv", "Attributes"),
    ("relation_frequencies.csv", "Relations"),
]


# ---------------------------------------------------------------------
# Rank-frequency distributions
# ---------------------------------------------------------------------

def plot_frequency_distributions():
    """
    Plot train and validation rank-frequency distributions for
    object classes, attributes, and relations.

    Labels are sorted by train frequency and the y-axis uses
    logarithmic scale to highlight the long-tail distribution.
    """

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(24, 8),
    )

    for ax, (csv_name, title) in zip(axes, DATASETS):
        df = pd.read_csv(INPUT_DIR / csv_name)

        df = df.sort_values(
            "train_count",
            ascending=False,
        ).reset_index(drop=True)

        ax.plot(
            range(len(df)),
            df["train_count"],
            label="Train",
        )

        ax.plot(
            range(len(df)),
            df["val_count"],
            label="Validation",
        )

        ax.set_yscale("log")

        ax.set_xlabel("Labels sorted by train frequency")
        ax.set_ylabel("Frequency")
        ax.set_title(title)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.suptitle(
        "GQA Label Frequency Distributions",
        fontsize=16,
        y=0.98,
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.90]
    )

    output_path = (
        OUTPUT_DIR
        / "frequency_distributions_combined.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------
# Most frequent labels
# ---------------------------------------------------------------------

def plot_top_labels_relative(top_n=30):
    """
    Plot the most frequent labels using relative frequencies.

    Relative frequencies are used because train and validation
    contain different numbers of annotations.
    """

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(24, 10),
    )

    for ax, (csv_name, title) in zip(axes, DATASETS):
        df = pd.read_csv(INPUT_DIR / csv_name)

        df["train_percentage"] = (
            df["train_count"]
            / df["train_count"].sum()
            * 100
        )

        df["val_percentage"] = (
            df["val_count"]
            / df["val_count"].sum()
            * 100
        )

        df = (
            df.sort_values(
                "train_percentage",
                ascending=False,
            )
            .head(top_n)
            .iloc[::-1]
        )

        y = np.arange(len(df))
        height = 0.4

        ax.barh(
            y - height / 2,
            df["train_percentage"],
            height=height,
            label="Train",
        )

        ax.barh(
            y + height / 2,
            df["val_percentage"],
            height=height,
            label="Validation",
        )

        ax.set_yticks(y)
        ax.set_yticklabels(df["label"])

        ax.set_xlabel(
            "Percentage of occurrences within split"
        )

        ax.set_title(title)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.suptitle(
        "GQA Label Frequency Distributions",
        fontsize=16,
        y=0.98,
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.90]
    )

    output_path = (
        OUTPUT_DIR
        / "top_labels_relative_combined.png"
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
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_frequency_distributions()
    plot_top_labels_relative()


if __name__ == "__main__":
    main()