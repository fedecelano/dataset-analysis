#!/usr/bin/env python3
"""Visualize GQA probing_5k bias analyses without recomputing associations.

Run from the repository root:
    python scripts/gqa_bias_analysis/plot_gqa_bias.py
    python scripts/gqa_bias_analysis/plot_gqa_bias.py --top-n 15 --format png
    python scripts/gqa_bias_analysis/plot_gqa_bias.py --input-dir outputs/gqa_bias_analysis/probing_5k/tables

Inputs: CSV tables produced by the four analyze_* scripts. The complete tables
are preferred; if they are absent, the ranked tables are used as a *filtered*
view. Missing filtered entries are displayed as N/A, never as zero.

Statistical units are inherited from the upstream analyses:
object-attribute and attribute-attribute: retained object instances;
object co-occurrence: distinct images; relational triples: annotated directed edges.
Plots show annotation associations, not causal or model biases.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("plot_gqa_bias")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLES = PROJECT_ROOT / "outputs" / "gqa_bias_analysis" / "probing_5k" / "tables"
DEFAULT_PLOTS = PROJECT_ROOT / "outputs" / "gqa_bias_analysis" / "probing_5k" / "plots"
SPLITS = ("train", "val", "test")
SPLIT_NAMES = {"train": "Train", "val": "Validation", "test": "Test"}

# The first candidate is always the complete table; ranked files are fallbacks.
ANALYSES = {
    "object_attribute": {
        "dir": "object_attribute", "files": ("object_attribute_associations.csv", "ranked_object_attribute_associations.csv"),
        "levels": {
            "macro_object_macro_attribute": ("object_label", "attribute_label", "lift", "pair_object_instances", "Lift"),
            "macro_object_original_attribute": ("object_label", "attribute_label", "lift", "pair_object_instances", "Lift"),
            "original_object_original_attribute": ("object_label", "attribute_label", "lift", "pair_object_instances", "Lift"),
        },
        "heatmap": "macro_object_macro_attribute",
    },
    "object_cooccurrence": {
        "dir": "object_cooccurrence", "files": ("object_cooccurrence_pairs.csv", "ranked_object_cooccurrence.csv"),
        "levels": {
            "macro_object": ("object_a", "object_b", "lift", "pair_images", "Lift"),
            "original_object": ("object_a", "object_b", "lift", "pair_images", "Lift"),
        },
        "heatmap": "macro_object",
    },
    "relational": {
        "dir": "relational", "files": ("relation_triples.csv", "ranked_relation_triples.csv"),
        "levels": {
            "macro_object": ("subject", "object", "lift_relation_given_pair", "triple_edges", "Lift"),
            "original_object": ("subject", "object", "lift_relation_given_pair", "triple_edges", "Lift"),
        },
        "heatmap": None,
    },
    "attribute_cooccurrence": {
        "dir": "attribute_cooccurrence", "files": ("attribute_cooccurrence_pairs.csv", "ranked_attribute_cooccurrence.csv"),
        "levels": {
            "macro_attribute": ("attribute_a", "attribute_b", "lift", "pair_objects", "Lift"),
            "original_attribute": ("attribute_a", "attribute_b", "lift", "pair_objects", "Lift"),
        },
        "heatmap": "macro_attribute",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_TABLES, help="Parent tables directory containing the four analysis subfolders")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_PLOTS, help="Plot output directory")
    p.add_argument("--top-n", type=int, default=15, help="Maximum ranked associations per split and level (default: 15)")
    p.add_argument("--format", choices=("png", "pdf", "svg"), default="png", help="Output image format")
    p.add_argument("--dpi", type=int, default=180, help="Raster resolution (default: 180)")
    p.add_argument("--analysis", choices=("all", *ANALYSES), default="all", help="Plot one analysis or all available")
    args = p.parse_args()
    if args.top_n < 1 or args.dpi < 1:
        p.error("--top-n and --dpi must be positive")
    return args


def load_table(folder: Path, filenames: tuple[str, str], columns: set[str]) -> tuple[pd.DataFrame, str] | None:
    for idx, filename in enumerate(filenames):
        path = folder / filename
        if not path.is_file():
            continue
        try:
            df = pd.read_csv(path, dtype={"split": "string", "level": "string"}, low_memory=False)
        except (pd.errors.EmptyDataError, UnicodeError) as exc:
            raise ValueError(f"Cannot read {path}: {exc}") from exc
        # The complete object co-occurrence table stores unordered pairs as
        # object_a/object_b. The ranked fallback stores directional views of
        # those pairs as anchor_object/context_object. Normalize the latter to
        # the same schema so both files are accepted without recomputation.
        if folder.name == "object_cooccurrence" and {"anchor_object", "context_object"}.issubset(df.columns) and not {"object_a", "object_b"}.issubset(df.columns):
            df = df.rename(columns={"anchor_object": "object_a", "context_object": "object_b"})
            # Both directions share the same lift/support. Keep one unordered
            # pair for plotting and avoid duplicate bars in ranked fallback.
            a = df["object_a"].astype(str)
            b = df["object_b"].astype(str)
            df["object_a"] = np.where(a <= b, a, b)
            df["object_b"] = np.where(a <= b, b, a)
            df = df.drop_duplicates(subset=["split", "level", "object_a", "object_b"])
        missing = columns - set(df.columns)
        if missing:
            LOGGER.warning("Skipping incompatible %s: missing columns %s", path, sorted(missing))
            continue
        if idx == 1:
            LOGGER.warning("%s: complete table unavailable; using RANKED/FILTERED data from %s. Blank heatmap cells mean unavailable, not zero.", folder.name, path.name)
        else:
            LOGGER.info("%s: using complete table %s", folder.name, path.name)
        return df, "ranked" if idx else "complete"
    LOGGER.warning("%s: no compatible input table found; skipping", folder)
    return None


def ensure_numeric(df: pd.DataFrame, column: str) -> None:
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df.loc[~np.isfinite(df[column]), column] = np.nan


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def save_fig(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved %s", path)


def get_label(row: pd.Series, left: str, right: str, analysis: str) -> str:
    if analysis == "relational":
        return f"{row[left]}  —{row['relation']}→  {row[right]}"
    return f"{row[left]}  /  {row[right]}"


def ranked_bars(df: pd.DataFrame, *, analysis: str, level: str, spec: tuple[str, str, str, str, str],
                split: str, output_dir: Path, top_n: int, extension: str, dpi: int, source: str) -> bool:
    left, right, metric, support, metric_label = spec
    d = df[(df["split"] == split) & (df["level"] == level)].copy()
    d = d.dropna(subset=[metric, support, left, right])
    d = d[(d[support] > 0) & (d[metric] >= 0)]
    if d.empty:
        LOGGER.warning("No chart data for %s/%s/%s", analysis, level, split)
        return False
    # To avoid unstable, very high lift estimates dominating the ranked plots,
    # rank by support first, then display selected rows sorted by metric.
    selected = d.nlargest(top_n, support).sort_values(metric, ascending=True).copy()
    selected["display_label"] = selected.apply(lambda row: get_label(row, left, right, analysis), axis=1)
    fig, ax = plt.subplots(figsize=(11, max(4.4, 0.53 * len(selected) + 1.6)))
    yy = np.arange(len(selected))
    ax.barh(yy, selected[metric].to_numpy(float), color="#4477AA")
    ax.set_yticks(yy, selected["display_label"].tolist(), fontsize=9)
    ax.set_xlabel(metric_label)
    ax.set_title(f"{analysis.replace('_', ' ').title()} — {level.replace('_', ' ')} — {SPLIT_NAMES[split]}\n"
                 f"Top {len(selected)} by {support}; support shown at bar ends")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    values = selected[metric].to_numpy(float)
    max_value = max(float(values.max()), 1e-8)
    ax.set_xlim(0, max_value * 1.23)
    for y, v, count in zip(yy, values, selected[support]):
        ax.text(float(v) + max_value * 0.012, y, f"n={int(count):,}", va="center", fontsize=8)
    fig.text(0.01, 0.01, f"Source: {source} CSV. {support} is annotation support; results are descriptive.", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    save_fig(fig, output_dir / analysis / f"{slug(level)}_{split}_top{top_n}.{extension}", dpi)
    return True


def heatmap(df: pd.DataFrame, *, analysis: str, level: str, spec: tuple[str, str, str, str, str],
            split: str, output_dir: Path, extension: str, dpi: int, source: str) -> bool:
    left, right, metric, _, metric_label = spec
    d = df[(df["split"] == split) & (df["level"] == level)].copy()
    d = d.dropna(subset=[left, right, metric])
    if d.empty:
        return False
    labels = sorted(set(d[left]).union(d[right])) if analysis in {"object_cooccurrence", "attribute_cooccurrence"} else None
    if labels is not None:
        matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
        for _, row in d.iterrows():
            a, b, value = str(row[left]), str(row[right]), float(row[metric])
            matrix.loc[a, b] = value
            matrix.loc[b, a] = value
        # Same-label pair is not part of these non-identical-label analyses.
        # Do not mutate matrix.values: recent pandas may expose a read-only view.
    else:
        matrix = d.pivot_table(index=left, columns=right, values=metric, aggfunc="first")
        matrix = matrix.reindex(index=sorted(matrix.index), columns=sorted(matrix.columns))
    if matrix.empty:
        return False
    array = matrix.to_numpy(dtype=float, copy=True)
    if labels is not None:
        np.fill_diagonal(array, np.nan)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return False
    vmax = max(float(np.quantile(finite, 0.98)), 0.001)
    if metric_label == "Lift":
        vmax = max(vmax, 1.0)
    fig_w = max(8, min(17, len(matrix.columns) * 0.65 + 3))
    fig_h = max(6, min(17, len(matrix.index) * 0.43 + 2.8))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    masked = np.ma.masked_invalid(array)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e5e5e5")
    im = ax.imshow(masked, interpolation="nearest", aspect="auto", cmap=cmap, norm=Normalize(vmin=0, vmax=vmax))
    ax.set_xticks(np.arange(len(matrix.columns)), [str(s) for s in matrix.columns], rotation=50, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(matrix.index)), [str(s) for s in matrix.index], fontsize=8)
    ax.set_xlabel(right.replace("_", " "))
    ax.set_ylabel(left.replace("_", " "))
    ax.set_title(f"{analysis.replace('_', ' ').title()} — {SPLIT_NAMES[split]} — {metric_label}")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=f"{metric_label} (98th percentile colour cap)")
    fig.text(0.01, 0.01, "Grey = no value in source table (or same-class excluded). Missing ≠ zero. "
             + ("Ranked/filtered input." if source == "ranked" else "Complete input."), fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_fig(fig, output_dir / analysis / f"{slug(level)}_{split}_heatmap.{extension}", dpi)
    return True


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    selected = ANALYSES if args.analysis == "all" else {args.analysis: ANALYSES[args.analysis]}
    written = 0
    for name, config in selected.items():
        specs = config["levels"]
        common = {"split", "level"}
        for spec in specs.values():
            common.update(spec[:4])
        if name == "relational":
            common.add("relation")
        found = load_table(args.input_dir / config["dir"], config["files"], common)
        if found is None:
            continue
        df, source = found
        for spec in specs.values():
            for col in (spec[2], spec[3]):
                ensure_numeric(df, col)
        for level, spec in specs.items():
            for split in SPLITS:
                if ranked_bars(df, analysis=name, level=level, spec=spec, split=split,
                               output_dir=args.output_dir, top_n=args.top_n, extension=args.format,
                               dpi=args.dpi, source=source):
                    written += 1
                if config["heatmap"] == level and heatmap(df, analysis=name, level=level, spec=spec,
                                                             split=split, output_dir=args.output_dir,
                                                             extension=args.format, dpi=args.dpi, source=source):
                    written += 1
    if not written:
        LOGGER.error("No plots created. Ensure the bias analysis CSV files exist under %s", args.input_dir)
        return 1
    LOGGER.info("Created %d plots under %s", written, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
