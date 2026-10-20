"""Create eight contrasting GQA scene-graph visualizations and one overview.

For each of train and validation, select one image in each profile:
``simple``, ``attributes``, ``relations``, and ``complex``. The four profiles
are selected comparatively within the split, not by fixed annotation-density
thresholds. A finite annotation budget keeps every annotation legible in the
tables below the image; candidates that exceed it are skipped rather than truncated.

Run from any working directory::

    python scripts/gqa_analysis/visualize_gqa_scene_graphs.py --dataset original
    python scripts/gqa_analysis/visualize_gqa_scene_graphs.py --dataset cleaned

Writes exactly nine PNG files to
``outputs/gqa_analysis/<dataset>/qualitative_examples`` on a fresh run.
"""

import argparse
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_DIRS = {
    "original": PROJECT_ROOT / "data/gqa/annotations",
    "cleaned": PROJECT_ROOT / "data/gqa_cleaned/annotations",
}
IMAGE_DIR = PROJECT_ROOT / "data/gqa/images"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/gqa_analysis"
SPLITS = ("train", "val")
PROFILES = ("simple", "attributes", "relations", "complex")
PROFILE_TITLES = {
    "simple": "Simple scene",
    "attributes": "Attribute-rich",
    "relations": "Relation-rich",
    "complex": "Complex scene",
}
# These limits are about fitting *complete* tables on a page, not about what
# counts as a simple / rich / complex scene (those scores are relative).
MAX_OBJECTS_IN_FIGURE = 14
MAX_ATTRIBUTES_IN_FIGURE = 36
MAX_RELATIONS_IN_FIGURE = 24


@dataclass(frozen=True)
class Candidate:
    image_id: str
    image_path: Path
    data: dict
    objects: int
    attributes: int
    relations: int


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(ANNOTATION_DIRS), default="original")
    return parser.parse_args()


def find_image_path(image_id):
    """Return a corresponding local image, if present."""
    for extension in (".jpg", ".jpeg", ".png"):
        path = IMAGE_DIR / f"{image_id}{extension}"
        if path.is_file():
            return path
    return None


def collect_candidates(scene_graphs):
    """Build eligible, renderable examples, with no annotation truncation."""
    candidates = []
    for image_id in sorted(scene_graphs):
        data = scene_graphs[image_id]
        objects = data["objects"]
        count_objects = len(objects)
        if not 2 <= count_objects <= MAX_OBJECTS_IN_FIGURE:
            continue
        attributes = sum(len(obj["attributes"]) for obj in objects.values())
        relations = sum(len(obj["relations"]) for obj in objects.values())
        if attributes > MAX_ATTRIBUTES_IN_FIGURE or relations > MAX_RELATIONS_IN_FIGURE:
            continue
        image_path = find_image_path(image_id)
        if image_path is None:
            continue
        # All relations should refer to an existing object. Do not silently
        # omit broken edges when displaying an example.
        if any(
            str(rel["object"]) not in objects
            for obj in objects.values()
            for rel in obj["relations"]
        ):
            continue
        candidates.append(Candidate(
            image_id=str(image_id), image_path=image_path, data=data,
            objects=count_objects, attributes=attributes, relations=relations,
        ))
    return candidates


def percentile_ranks(values):
    """Return rank scores in [0, 1], assigning equal scores to ties."""
    unique = sorted(set(values))
    if len(unique) == 1:
        return {unique[0]: 0.5}
    return {v: i / (len(unique) - 1) for i, v in enumerate(unique)}


def select_profiles(candidates):
    """Select four DISTINCT examples spanning relative annotation profiles.

    Profile scores emphasize low total density (simple), attribute density,
    relation density, and high total density (complex), respectively. All
    scores are computed against the *eligible pool of the same split*.
    """
    if len(candidates) < 4:
        raise ValueError(
            f"Only {len(candidates)} eligible images; need at least four. "
            "Check local image files and annotation budgets."
        )
    ranks = {
        key: percentile_ranks([getattr(c, key) for c in candidates])
        for key in ("objects", "attributes", "relations")
    }
    ranked = []
    for c in candidates:
        o = ranks["objects"][c.objects]
        a = ranks["attributes"][c.attributes]
        r = ranks["relations"][c.relations]
        # Require at least one relevant annotation for the two rich profiles.
        scores = {
            "simple": -(o + a + r),
            "attributes": (2 * a + 0.4 * o - 0.7 * r)
                          if c.attributes else float("-inf"),
            "relations": (2 * r + 0.4 * o - 0.7 * a)
                         if c.relations else float("-inf"),
            "complex": o + a + r,
        }
        ranked.append((c, scores))

    # Prioritize each profile's strongest available match and enforce unique IDs.
    # Explicit stable tie-breaking makes selection reproducible with no global RNG.
    remaining = set(PROFILES)
    selected = {}
    used_ids = set()
    while remaining:
        options = []
        for profile in remaining:
            for c, scores in ranked:
                if c.image_id not in used_ids and math.isfinite(scores[profile]):
                    options.append((scores[profile], profile, c.image_id, c))
        if not options:
            raise ValueError("Cannot find distinct candidates for all four profiles.")
        options.sort(key=lambda x: (-x[0], PROFILES.index(x[1]), x[2]))
        _, profile, _, candidate = options[0]
        selected[profile] = candidate
        used_ids.add(candidate.image_id)
        remaining.remove(profile)
    return selected


def wrap(value, width):
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=True)) or "—"


def draw_table(ax, title, headers, rows, widths, font_size=9):
    """Draw a table with row heights proportional to wrapped cell content."""
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=8)
    if not rows:
        rows = [["—"] + [""] * (len(headers) - 1)]
    all_rows = [list(headers)] + [list(row) for row in rows]
    row_units = [max(s.count("\n") + 1 for s in row) + 0.5 for row in all_rows]
    total = sum(row_units)
    table = ax.table(cellText=all_rows, colWidths=widths, cellLoc="left", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    y = 1.0
    for row_index, (cells, units) in enumerate(zip(all_rows, row_units)):
        h = units / total
        y -= h
        for col in range(len(headers)):
            cell = table[(row_index, col)]
            cell.set_y(y)
            cell.set_height(h)
            cell.set_edgecolor("#D6DCE4")
            cell.set_facecolor("#E9EEF5" if row_index == 0 else (
                "#FFFFFF" if row_index % 2 else "#F7F9FC"
            ))
            cell.PAD = 0.035
            cell.get_text().set_va("center")
            if row_index == 0:
                cell.get_text().set_weight("bold")
    return total


def make_detail(candidate, split, profile, destination):
    objects = candidate.data["objects"]
    obj_items = list(objects.items())
    visual_ids = {object_id: i + 1 for i, (object_id, _) in enumerate(obj_items)}
    object_rows = []
    relation_rows = []
    for object_id, obj in obj_items:
        object_rows.append([
            str(visual_ids[object_id]), wrap(obj["name"], 20),
            wrap(", ".join(obj["attributes"]) or "—", 34),
        ])
        for rel in obj["relations"]:
            target_id = str(rel["object"])
            relation_rows.append([
                str(visual_ids[object_id]), wrap(rel["name"], 24),
                str(visual_ids[target_id]),
            ])

    # Each detail sheet flows vertically: annotated image, object table,
    # relation table. Heights are derived from the displayed table content
    # so that sparse scenes do not inherit the height of complex scenes.
    def table_units(rows):
        displayed = rows if rows else [["—", "", ""]]
        return sum(max(cell.count("\n") + 1 for cell in row) + 0.5 for row in displayed) + 1.5

    object_units = table_units(object_rows)
    relation_units = table_units(relation_rows)

    figure_width = 14.5
    image_height = 6.0
    object_height = max(1.1, object_units * 0.30)
    relation_height = max(1.1, relation_units * 0.30)
    top_margin = 0.85
    section_gap = 0.42
    bottom_margin = 0.32
    figure_height = (
        top_margin + image_height + section_gap + object_height
        + section_gap + relation_height + bottom_margin
    )

    fig = plt.figure(figsize=(figure_width, figure_height), facecolor="white")
    # Figure coordinates are calculated in inches, then normalized. This
    # avoids the vertically centered image and large blank areas seen when
    # an image shares a row with tall tables.
    left = 0.06
    right = 0.94
    usable_width = right - left
    current_top = figure_height - top_margin

    current_top -= image_height
    ax_image = fig.add_axes([
        left, current_top / figure_height,
        usable_width, image_height / figure_height,
    ])
    with Image.open(candidate.image_path) as image:
        ax_image.imshow(image.convert("RGB"))
    for object_id, obj in obj_items:
        x, y, w, h = (obj[key] for key in ("x", "y", "w", "h"))
        ax_image.add_patch(Rectangle(
            (x, y), w, h, linewidth=1.5, edgecolor="red", facecolor="none"
        ))
        ax_image.text(
            x, max(0, y - 3), str(visual_ids[object_id]), fontsize=9,
            weight="bold", color="black",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "red", "pad": 1.5},
        )
    ax_image.set_anchor("N")
    ax_image.set_title(
        f"{split.upper()} | {PROFILE_TITLES[profile]}\n"
        f"Image ID: {candidate.image_id} | Objects: {candidate.objects} | "
        f"Attributes: {candidate.attributes} | Relations: {candidate.relations}",
        fontsize=12, pad=12,
    )
    ax_image.axis("off")

    current_top -= section_gap + object_height
    ax_objects = fig.add_axes([
        left, current_top / figure_height,
        usable_width, object_height / figure_height,
    ])
    draw_table(ax_objects, "Objects and attributes", ["ID", "Object", "Attributes"],
               object_rows, [0.09, 0.29, 0.62], font_size=10)

    current_top -= section_gap + relation_height
    ax_relations = fig.add_axes([
        left, current_top / figure_height,
        usable_width, relation_height / figure_height,
    ])
    draw_table(ax_relations, "Relations", ["Source ID", "Relation", "Target ID"],
               relation_rows, [0.18, 0.64, 0.18], font_size=10)

    fig.savefig(destination, dpi=155, facecolor="white")
    plt.close(fig)


def make_overview(selections, destination):
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), layout="constrained")
    for row, split in enumerate(SPLITS):
        for column, profile in enumerate(PROFILES):
            candidate = selections[split][profile]
            ax = axes[row, column]
            with Image.open(candidate.image_path) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(
                f"{split.upper()} — {PROFILE_TITLES[profile]}\n"
                f"ID: {candidate.image_id} | O:{candidate.objects} "
                f"A:{candidate.attributes} R:{candidate.relations}",
                fontsize=11,
            )
            ax.axis("off")
    fig.savefig(destination, dpi=170, facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    annotation_dir = ANNOTATION_DIRS[args.dataset]
    out_dir = OUTPUT_ROOT / args.dataset / "qualitative_examples"
    selections = {}
    for split in SPLITS:
        scene_graph_path = annotation_dir / f"{split}_sceneGraphs.json"
        print(f"Loading {scene_graph_path} ...", flush=True)
        with scene_graph_path.open("r", encoding="utf-8") as file:
            scene_graphs = json.load(file)
        candidates = collect_candidates(scene_graphs)
        print(f"{split}: {len(candidates)} eligible images", flush=True)
        selections[split] = select_profiles(candidates)
        del scene_graphs  # Keep only the eight selected scene graphs.

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for profile in PROFILES:
            candidate = selections[split][profile]
            filename = f"{split}_{profile}.png"
            make_detail(candidate, split, profile, out_dir / filename)
            print(f"Saved {filename}: {candidate.image_id}", flush=True)
    make_overview(selections, out_dir / "overview.png")
    print(f"Saved 9 PNG images in: {out_dir}")


if __name__ == "__main__":
    main()
