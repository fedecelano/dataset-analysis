#!/usr/bin/env python3
"""Slide-friendly bias association plots for frozen GQA probing_5k.

Reads existing bias CSVs without changing annotations or recomputing metrics.
Supports object-attribute, object-object, and directed object-relation-object.

Usage (from repository root):
    python scripts/gqa_bias_analysis/plot_gqa_bias_slides.py
    python scripts/gqa_bias_analysis/plot_gqa_bias_slides.py --level all --top-n 8 --format pdf
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

LEVELS = (
    "macro_object_macro_attribute",
    "macro_object_original_attribute",
    "original_object_original_attribute",
)
SPLITS = ("train", "val", "test")
REQUIRED = {
    "split", "level", "object_label", "attribute_label", "object_instances",
    "pair_object_instances", "lift", "p_attribute_given_object_pct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=Path("outputs/gqa_bias_analysis/probing_5k/tables"),
                        help="Root of bias tables; legacy object_attribute folder accepted for that analysis")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/gqa_bias_analysis/probing_5k/plots_slides"),
                        help="Root of slide plots, with per-analysis subfolders")
    parser.add_argument("--analysis", choices=("all", "object_attribute", "object_cooccurrence", "relational"), default="all")
    parser.add_argument("--level", choices=("macro", "all", *LEVELS), default="macro")
    parser.add_argument("--split", choices=("all", *SPLITS), default="all")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-pair-objects", type=int, default=20)
    parser.add_argument("--min-object-instances", type=int, default=50)
    parser.add_argument("--label-points", type=int, default=4)
    parser.add_argument("--min-pair-images", type=int, default=20)
    parser.add_argument("--min-anchor-images", type=int, default=50)
    parser.add_argument("--min-triple-edges", type=int, default=20)
    parser.add_argument("--min-triple-images", type=int, default=10)
    parser.add_argument("--min-rel-pair-edges", type=int, default=50)
    parser.add_argument("--per-relation-top-n", type=int, default=4)
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_records(folder: Path) -> tuple[list[dict], Path, bool]:
    candidates = (
        (folder / "object_attribute_associations.csv", False),
        (folder / "ranked_object_attribute_associations.csv", True),
    )
    issues: list[str] = []
    for path, ranked in candidates:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                missing = REQUIRED - set(reader.fieldnames or [])
                if missing:
                    issues.append(f"{path.name}: missing {sorted(missing)}")
                    continue
                rows = []
                for r in reader:
                    try:
                        support = int(r["pair_object_instances"])
                        anchor = int(r["object_instances"])
                        lift = float(r["lift"])
                        prob = float(r["p_attribute_given_object_pct"])
                    except (ValueError, TypeError):
                        continue  # Undefined/zero-denominator rows are not ranked.
                    if not (math.isfinite(lift) and math.isfinite(prob)):
                        continue
                    rows.append({
                        "split": r["split"], "level": r["level"],
                        "object": r["object_label"], "attribute": r["attribute_label"],
                        "label": f'{r["object_label"]}  →  {r["attribute_label"]}',
                        "support": support, "anchor": anchor,
                        "lift": lift, "prob": prob,
                    })
            return rows, path, ranked
        except (OSError, UnicodeError) as exc:
            issues.append(f"{path.name}: {exc}")
    raise FileNotFoundError("No compatible CSV found in " + str(folder) + "\n" + "\n".join(issues))


def eligible(rows: list[dict], split: str, level: str, args: argparse.Namespace) -> list[dict]:
    result = [r for r in rows if r["split"] == split and r["level"] == level
              and r["support"] >= args.min_pair_objects
              and r["anchor"] >= args.min_object_instances]
    # Stable tie-breaking: lift desc, pair support desc, label asc.
    return sorted(result, key=lambda r: (-r["lift"], -r["support"], r["label"]))


def apply_theme():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.edgecolor": "#BBC4D0",
        "axes.labelcolor": "#253244", "xtick.color": "#586677",
        "ytick.color": "#253244", "text.color": "#182332",
    })


def title_text(level: str) -> str:
    return {
        LEVELS[0]: "Object–attribute types",
        LEVELS[1]: "Object types–attribute values",
        LEVELS[2]: "Object values–attribute values",
    }[level]


def save(fig, path: Path, args: argparse.Namespace):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"SAVED {path}")


def bar_chart(rows: list[dict], level: str, split: str, path: Path, args: argparse.Namespace):
    top = rows[:args.top_n]
    if not top:
        print(f"SKIP empty bar plot: {level}/{split}")
        return
    # Selection is strict descending; invert_yaxis renders largest at the top.
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    fig.subplots_adjust(left=0.34 if level != LEVELS[0] else 0.29, right=0.88,
                        top=0.79, bottom=0.16)
    y = np.arange(len(top))
    values = np.array([r["lift"] for r in top])
    bars = ax.barh(y, values, color="#2478B4", height=0.7)
    bars[0].set_color("#125283")
    ax.set_yticks(y, [r["label"].replace("_", " ") for r in top], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1.15, values.max() * 1.31))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax.grid(axis="x", color="#DDE5EB", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.set_xlabel("Lift  ·  observed association / independence baseline", fontsize=12, labelpad=10)
    for rect, r in zip(bars, top):
        ax.text(rect.get_width() + values.max() * 0.025,
                rect.get_y() + rect.get_height()/2,
                f'{r["lift"]:.2f}×   n={r["support"]:,}',
                va="center", ha="left", fontsize=11, color="#2B3F53")
    fig.text(0.055, 0.94, "TOP ASSOCIATIONS BY LIFT", fontsize=13,
             color="#3275A5", weight="bold")
    fig.text(0.055, 0.875, f"{title_text(level)} · {split.upper()}",
             fontsize=24, weight="bold")
    fig.text(0.055, 0.825,
             "Descending lift; n = objects carrying both labels. Minimum support applied.",
             fontsize=11, color="#657386")
    fig.text(0.055, 0.055,
             "Association in annotated GQA probing_5k; lift is not a model-bias measurement.",
             fontsize=10, color="#687789")
    save(fig, path, args)


def scatter_chart(rows: list[dict], level: str, split: str, path: Path, args: argparse.Namespace):
    if not rows:
        print(f"SKIP empty scatter: {level}/{split}")
        return
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    fig.subplots_adjust(left=0.10, right=0.92, top=0.79, bottom=0.17)
    xs = np.array([r["support"] for r in rows], dtype=float)
    ys = np.array([r["lift"] for r in rows], dtype=float)
    ax.scatter(xs, ys, s=52, color="#2478B4", alpha=0.58,
               edgecolors="white", linewidths=0.6)
    ax.axhline(1, linestyle="--", color="#8A99A8", linewidth=1.3,
               label="Lift = 1 (independence)")
    ax.set_xscale("log" if xs.max()/max(1, xs.min()) > 30 else "linear")
    ax.set_xlabel("Support  ·  number of objects carrying both labels", fontsize=12, labelpad=10)
    ax.set_ylabel("Lift", fontsize=12, labelpad=10)
    ax.grid(color="#DFE6ED", alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0, top=max(1.3, ys.max()*1.17))
    # Only highlight top points by lift, and keep annotation offset inside figure.
    for idx, r in enumerate(rows[:args.label_points]):
        ax.scatter([r["support"]], [r["lift"]], s=120,
                   color="#E58346", edgecolor="white", linewidth=1.1, zorder=4)
        offset = (12, -19 if idx % 2 else 12)
        ax.annotate(r["label"].replace("_", " "),
                    (r["support"], r["lift"]), xytext=offset,
                    textcoords="offset points", fontsize=10,
                    color="#8B3E1D", weight="bold",
                    bbox={"boxstyle":"round,pad=0.2", "facecolor":"white", "edgecolor":"none", "alpha":0.8})
    ax.legend(loc="upper right", frameon=False, fontsize=11)
    fig.text(0.055, 0.94, "STRENGTH VS EVIDENCE", fontsize=13,
             color="#3275A5", weight="bold")
    fig.text(0.055, 0.875, f"{title_text(level)} · {split.upper()}",
             fontsize=24, weight="bold")
    fig.text(0.055, 0.825,
             "Each dot is an observed object–attribute association; highlighted = highest lift.",
             fontsize=11, color="#657386")
    fig.text(0.055, 0.055,
             "Support is counted in object instances; values are computed from dataset annotations.",
             fontsize=10, color="#687789")
    save(fig, path, args)


def run_object_attribute(args: argparse.Namespace) -> int:
    folder = args.input_dir / "object_attribute" if (args.input_dir / "object_attribute").is_dir() else args.input_dir
    records, source, ranked_only = load_records(folder)
    print(f"SOURCE {source} ({'filtered/ranked' if ranked_only else 'complete'})")
    if ranked_only:
        print("NOTE: input is already filtered; excluded associations cannot be recovered.")
    apply_theme()
    levels = LEVELS if args.level == "all" else (LEVELS[0],) if args.level == "macro" else (args.level,)
    splits = SPLITS if args.split == "all" else (args.split,)
    generated = 0
    for level in levels:
        for split in splits:
            rows = eligible(records, split, level, args)
            if not rows:
                print(f"SKIP no eligible associations: {level}/{split}")
                continue
            stem = f"{level}_{split}"
            bar_chart(rows, level, split, args.output_dir / "object_attribute" / f"{stem}_top{args.top_n}_lift.{args.format}", args)
            scatter_chart(rows, level, split, args.output_dir / "object_attribute" / f"{stem}_lift_vs_support.{args.format}", args)
            generated += 2
    print(f"DONE {generated} figures")
    return 0




# ---- Object-object and directed relation plots ----

def _load_first_compatible(folder: Path, files: tuple[tuple[str, bool], ...], required: set[str]):
    problems = []
    for filename, ranked in files:
        path = folder / filename
        if not path.exists():
            continue
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                missing = required - set(reader.fieldnames or ())
                if missing:
                    problems.append(f"{filename}: missing {sorted(missing)}")
                    continue
                rows = list(reader)
            print(f"SOURCE {path} ({'ranked/filtered' if ranked else 'complete'})")
            if ranked:
                print("NOTE: ranked source already excludes low-support combinations.")
            return rows
        except (OSError, UnicodeError) as exc:
            problems.append(f"{filename}: {exc}")
    raise FileNotFoundError(f"No compatible CSV in {folder}; {'; '.join(problems)}")


def _numeric(row: dict, field: str):
    try:
        x = float(row.get(field, ""))
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _ordered(rows: list[dict]):
    return sorted(rows, key=lambda r: (-r['lift'], -r['support'], r['label']))


def load_object_pairs(folder: Path) -> list[dict]:
    folder = folder / 'object_cooccurrence' if (folder / 'object_cooccurrence').is_dir() else folder
    # Complete file has unordered object_a/object_b; ranked file has both directions.
    complete = folder / 'object_cooccurrence_pairs.csv'
    ranked = folder / 'ranked_object_cooccurrence.csv'
    candidates = (
        ('object_cooccurrence_pairs.csv', False),
        ('ranked_object_cooccurrence.csv', True),
    )
    problems = []
    for name, filtered in candidates:
        path = folder / name
        if not path.exists():
            continue
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            if {'object_a', 'object_b'} <= fields:
                a, b = 'object_a', 'object_b'
            elif {'anchor_object', 'context_object'} <= fields:
                a, b = 'anchor_object', 'context_object'
            else:
                problems.append(f'{name}: no object pair labels')
                continue
            if not {'split', 'level', 'pair_images', 'lift'} <= fields:
                problems.append(f'{name}: missing split/level/pair_images/lift')
                continue
            dedup = {}
            for r in reader:
                lift, support = _numeric(r, 'lift'), _numeric(r, 'pair_images')
                if lift is None or support is None or support <= 0:
                    continue
                oa, ob = (r.get(a) or '').strip(), (r.get(b) or '').strip()
                if not oa or not ob or oa == ob:
                    continue
                first, second = sorted((oa, ob))
                key = r['split'], r['level'], first, second
                # Complete unordered-pair tables can omit anchor/context prevalence columns.
                # Such missing values must NOT be interpreted as zero: doing so would
                # incorrectly filter every pair against --min-anchor-images.
                anchor_fields = (
                    ('anchor_images', 'context_images') if a == 'anchor_object' else
                    ('object_a_images', 'object_b_images')
                )
                marginal_values = [_numeric(r, field) for field in anchor_fields]
                available_marginals = [v for v in marginal_values if v is not None]
                max_marginal = max(available_marginals) if available_marginals else None
                dedup.setdefault(key, {
                    'split': r['split'], 'level': r['level'], 'first': first, 'second': second,
                    'label': f'{first}  ↔  {second}', 'support': int(support), 'lift': lift,
                    'anchor': max_marginal,
                })
            missing_marginals = sum(v['anchor'] is None for v in dedup.values())
            print(f"SOURCE {path} ({'ranked/filtered' if filtered else 'complete'}); {len(dedup)} undirected pairs")
            if missing_marginals:
                print(f'NOTE: marginal object-image supports are unavailable for {missing_marginals} pairs; '
                      '--min-anchor-images is not applied to those pairs. '
                      'The --min-pair-images filter remains active.')
            if filtered:
                print('NOTE: ranked data omit pairs below prior thresholds.')
            return list(dedup.values())
    raise FileNotFoundError(f'No compatible object cooccurrence CSV in {folder}: {problems}')


def load_relation_triples(folder: Path) -> list[dict]:
    folder = folder / 'relational' if (folder / 'relational').is_dir() else folder
    raw = _load_first_compatible(folder, (
        ('relation_triples.csv', False), ('ranked_relation_triples.csv', True)),
        {'split','level','subject','relation','object','triple_edges','triple_images',
         'subject_object_edges','lift_relation_given_pair'})
    rows = []
    for r in raw:
        lift = _numeric(r, 'lift_relation_given_pair')
        edges = _numeric(r, 'triple_edges')
        images = _numeric(r, 'triple_images')
        pair_edges = _numeric(r, 'subject_object_edges')
        if None in (lift, edges, images, pair_edges) or edges <= 0:
            continue
        rows.append({
            'split': r['split'], 'level': r['level'], 'subject': r['subject'],
            'relation': r['relation'], 'object': r['object'],
            'label': f"{r['subject']}  — {r['relation']} →  {r['object']}",
            'lift': lift, 'support': int(edges), 'images': int(images),
            'pair_edges': int(pair_edges)
        })
    return rows


def _new_bar(rows, title, kicker, subtitle, footnote, path, args, support_label='n'):
    top = _ordered(rows)[:args.top_n]
    if not top:
        return False
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    max_label = max(len(r['label']) for r in top)
    left = min(0.51, max(0.29, 0.29 + (max_label - 26) * 0.004))
    fig.subplots_adjust(left=left, right=0.87, top=0.78, bottom=0.16)
    y = np.arange(len(top))
    vals = np.array([r['lift'] for r in top])
    bars = ax.barh(y, vals, color='#2478B4', height=0.68)
    bars[0].set_color('#125283')
    ax.set_yticks(y, [r['label'].replace('_', ' ') for r in top], fontsize=10.5 if max_label > 42 else 12)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1.15, vals.max() * 1.37))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
    ax.grid(axis='x', color='#DDE5EB', linewidth=.8)
    ax.set_axisbelow(True)
    ax.set_xlabel('Lift', labelpad=10)
    for rect, r in zip(bars, top):
        ax.text(rect.get_width() + vals.max()*.025, rect.get_y()+rect.get_height()/2,
                f"{r['lift']:.2f}×  {support_label}={r['support']:,}", va='center', fontsize=10.5)
    fig.text(.055,.94,kicker,fontsize=13,color='#3275A5',weight='bold')
    fig.text(.055,.875,title,fontsize=23,weight='bold')
    fig.text(.055,.825,subtitle,fontsize=11,color='#657386')
    fig.text(.055,.055,footnote,fontsize=9.5,color='#687789')
    save(fig,path,args)
    return True


def _new_scatter(rows, title, kicker, subtitle, footnote, path, args, support_unit):
    rows = _ordered(rows)
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    fig.subplots_adjust(left=.10,right=.92,top=.79,bottom=.17)
    xs = np.array([r['support'] for r in rows], dtype=float)
    ys = np.array([r['lift'] for r in rows], dtype=float)
    ax.scatter(xs,ys,s=51,color='#2478B4',alpha=.54,edgecolors='white',linewidth=.55)
    ax.axhline(1,linestyle='--',color='#8A99A8',linewidth=1.3,label='Lift = 1')
    if xs.min() > 0 and xs.max()/xs.min()>30:
        ax.set_xscale('log')
    ax.set_xlabel(f'Support  ·  {support_unit}',labelpad=10)
    ax.set_ylabel('Lift',labelpad=10)
    ax.grid(color='#DFE6ED',alpha=.8)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0,top=max(1.3,float(ys.max())*1.20))
    for idx,r in enumerate(rows[:args.label_points]):
        ax.scatter([r['support']],[r['lift']],s=110,color='#E58346',edgecolors='white',linewidth=1.1,zorder=4)
        ax.annotate(r['label'].replace('_',' '),(r['support'],r['lift']),xytext=(12, 12 if idx%2==0 else -22),
                    textcoords='offset points',fontsize=9.3,color='#8B3E1D',weight='bold',
                    bbox={'boxstyle':'round,pad=.2','facecolor':'white','edgecolor':'none','alpha':.85})
    ax.legend(loc='upper right',frameon=False)
    fig.text(.055,.94,kicker,fontsize=13,color='#3275A5',weight='bold')
    fig.text(.055,.875,title,fontsize=23,weight='bold')
    fig.text(.055,.825,subtitle,fontsize=11,color='#657386')
    fig.text(.055,.055,footnote,fontsize=9.5,color='#687789')
    save(fig,path,args)
    return True


def _pair_heatmap(rows, level, split, path, args):
    top = _ordered(rows)[:args.top_n]
    labels=[]
    for r in top:
        for obj in (r['first'],r['second']):
            if obj not in labels:
                labels.append(obj)
    if len(labels)<2:
        return False
    # Only retain labels participating in the strongest pairs. Unseen pairs remain blank.
    data=np.full((len(labels),len(labels)),np.nan)
    positions={name:i for i,name in enumerate(labels)}
    for r in rows:
        if r['first'] in positions and r['second'] in positions:
            i,j=positions[r['first']],positions[r['second']]
            data[i,j]=data[j,i]=r['lift']
    fig,ax=plt.subplots(figsize=(13.33,7.5))
    fig.subplots_adjust(left=.17,right=.78,top=.77,bottom=.17)
    masked=np.ma.masked_invalid(data)
    cmap=plt.get_cmap('YlGnBu').copy()
    cmap.set_bad('#EDF1F5')
    im=ax.imshow(masked,cmap=cmap,aspect='equal',vmin=0)
    ax.set_xticks(np.arange(len(labels)),[x.replace('_',' ') for x in labels],rotation=45,ha='right',fontsize=9.2)
    ax.set_yticks(np.arange(len(labels)),[x.replace('_',' ') for x in labels],fontsize=9.2)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if np.isfinite(data[i,j]):
                ax.text(j,i,f'{data[i,j]:.1f}',ha='center',va='center',fontsize=8.2,
                        color='white' if data[i,j]>np.nanmax(data)*.55 else '#182332')
    cb=fig.colorbar(im,ax=ax,fraction=.048,pad=.035)
    cb.set_label('Lift')
    fig.text(.055,.94,'CONTEXTUAL ASSOCIATIONS',fontsize=13,color='#3275A5',weight='bold')
    fig.text(.055,.875,f'Object co-occurrence · {split.upper()}',fontsize=23,weight='bold')
    fig.text(.055,.825,f'Labels from top {args.top_n} pairs by lift · symmetric matrix',fontsize=11,color='#657386')
    fig.text(.055,.055,'Diagonal excluded; blank off-diagonal cells mean unavailable or below the source threshold.',fontsize=9.5,color='#687789')
    save(fig,path,args)
    return True


def _per_relation_panels(rows, split, path, args):
    """Compact cards with pair names ABOVE bars, not overflowing into other panels."""
    relations=('behind','in','in front of','near','on','to the left of','to the right of')
    fig, axes=plt.subplots(2,4,figsize=(18.5,10.4))
    axes=axes.ravel()
    fig.subplots_adjust(left=.055,right=.975,top=.79,bottom=.10,hspace=.47,wspace=.22)
    nonempty=False
    for i,relation in enumerate(relations):
        ax=axes[i]
        top=_ordered([r for r in rows if r['relation']==relation])[:args.per_relation_top_n]
        ax.set_title(relation,weight='bold',fontsize=14,pad=12,loc='left')
        if not top:
            ax.text(.5,.5,'No qualifying triples',transform=ax.transAxes,ha='center',va='center',fontsize=10)
            ax.set_axis_off()
            continue
        nonempty=True
        y=np.arange(len(top))*2
        vals=np.array([r['lift'] for r in top])
        ax.barh(y,vals,color='#2478B4',height=.56)
        ax.invert_yaxis()
        ax.set_ylim(y[-1]+.9,-.95)
        ax.set_yticks([])
        ax.set_xlim(0,max(1.1,vals.max()*1.20))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
        ax.grid(axis='x',color='#DDE5EB',alpha=.8)
        ax.set_axisbelow(True)
        for yi,r in zip(y,top):
            pair=f"{r['subject'].replace('_',' ')} → {r['object'].replace('_',' ')}"
            if len(pair)>31:
                pair=pair[:28]+'…'
            ax.text(0,yi-.46,pair,fontsize=9.4,va='bottom',color='#253244')
            ax.text(r['lift']+vals.max()*.024,yi,f"{r['lift']:.1f}",va='center',fontsize=9.5)
    axes[-1].set_axis_off()
    fig.text(.055,.947,'RELATION-SPECIFIC ASSOCIATIONS',fontsize=13,color='#3275A5',weight='bold')
    fig.text(.055,.890,f'Top object pairs per relation · {split.upper()}',fontsize=23,weight='bold')
    fig.text(.055,.846,'Each panel is independently ranked by lift; support thresholds apply.',fontsize=11,color='#657386')
    fig.text(.055,.045,'Lift = P(relation | annotated directed subject–object edges) / P(relation | all annotated edges).',fontsize=10,color='#687789')
    if nonempty:
        save(fig,path,args)
    else:
        plt.close(fig)
    return nonempty


def run_object_cooccurrence(args):
    raw=load_object_pairs(args.input_dir)
    levels=('macro_object','original_object') if args.level=='all' else ('macro_object',) if args.level=='macro' or args.level not in ('macro_object','original_object') else (args.level,)
    splits=SPLITS if args.split=='all' else (args.split,)
    output=args.output_dir/'object_cooccurrence'
    count=0
    for level in levels:
        for split in splits:
            rows=_ordered([r for r in raw if r['split']==split and r['level']==level
                           and r['support']>=args.min_pair_images
                           and (r['anchor'] is None or r['anchor']>=args.min_anchor_images)])
            if not rows:
                print(f'SKIP object-object {level}/{split}: no qualifying pairs')
                continue
            stem=f'{level}_{split}'
            title=f'Object co-occurrence · {split.upper()}'
            note='Each unordered pair is counted once; support = images containing both classes.'
            count+=_new_bar(rows,title,'TOP OBJECT PAIRS BY LIFT','Descending lift; duplicates from opposite conditional directions removed.',note,
                            output/f'{stem}_top{args.top_n}_lift.{args.format}',args,support_label='images')
            count+=_new_scatter(rows,title,'STRENGTH VS EVIDENCE','Each dot is an unordered object pair.',note,
                                output/f'{stem}_lift_vs_support.{args.format}',args,'distinct images')
            count+=_pair_heatmap(rows,level,split,output/f'{stem}_compact_heatmap.{args.format}',args)
    return count


def run_relational(args):
    raw=load_relation_triples(args.input_dir)
    levels=('macro_object','original_object') if args.level=='all' else ('macro_object',) if args.level=='macro' or args.level not in ('macro_object','original_object') else (args.level,)
    splits=SPLITS if args.split=='all' else (args.split,)
    output=args.output_dir/'relation_triples'
    count=0
    for level in levels:
        for split in splits:
            rows=_ordered([r for r in raw if r['split']==split and r['level']==level
                           and r['support']>=args.min_triple_edges and r['images']>=args.min_triple_images
                           and r['pair_edges']>=args.min_rel_pair_edges])
            if not rows:
                print(f'SKIP relational {level}/{split}: no qualifying triples')
                continue
            stem=f'{level}_{split}'
            title=f'Directed relation triples · {split.upper()}'
            note='Support = annotated directed edges; inverse left/right edges are retained.'
            count+=_new_bar(rows,title,'TOP RELATION TRIPLES BY LIFT','Lift conditions on the directed subject–object class pair.',note,
                            output/f'{stem}_top{args.top_n}_lift.{args.format}',args,support_label='edges')
            count+=_new_scatter(rows,title,'STRENGTH VS EVIDENCE','Each dot is a directed (subject, relation, object) triple.',note,
                                output/f'{stem}_lift_vs_support.{args.format}',args,'directed annotation edges')
            if level=='macro_object':
                count+=_per_relation_panels(rows,split,output/f'{stem}_top_pairs_per_relation.{args.format}',args)
    return count


def main() -> int:
    args=parse_args()
    if args.top_n<1 or args.per_relation_top_n<1 or args.min_pair_objects<1 or args.min_object_instances<1 \
            or args.min_pair_images<1 or args.min_anchor_images<1 or args.min_triple_edges<1 \
            or args.min_triple_images<1 or args.min_rel_pair_edges<1 or args.label_points<0:
        raise SystemExit('Minimum support thresholds and top-N must be positive; --label-points >= 0')
    apply_theme()
    jobs=('object_attribute','object_cooccurrence','relational') if args.analysis=='all' else (args.analysis,)
    count=0
    for job in jobs:
        try:
            if job=='object_attribute':
                count+=run_object_attribute(args)
            elif job=='object_cooccurrence':
                count+=run_object_cooccurrence(args)
            else:
                count+=run_relational(args)
        except FileNotFoundError as exc:
            print(f'SKIP {job}: {exc}')
            if args.analysis!='all':
                return 1
    print(f'DONE {count} figures')
    return 0 if count else 1


if __name__ == '__main__':
    raise SystemExit(main())
