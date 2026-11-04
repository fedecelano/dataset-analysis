
"""Visualize eight contrasting GQA scene graphs and generate an overview.

For each train and validation split, select one image for each profile:
simple, attributes, relations, and complex.

Profiles are defined relative to eligible images within each split.
Selection is deterministic and uses a fixed profile priority.

Annotation limits keep the complete object and relation tables
reasonably readable. Annotations are never truncated.

The script does not modify the source images or annotations.

Outputs are saved to:
    outputs/gqa_analysis/<dataset>/qualitative_examples/

Usage:
    python scripts/gqa_analysis/visualize_gqa_scenegraphs.py --dataset original
    python scripts/gqa_analysis/visualize_gqa_scenegraphs.py --dataset cleaned
"""

import argparse
import json
import math
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANNOTATION_DIRS = {
    "original": PROJECT_ROOT / "data" / "gqa" / "annotations",
    "cleaned": PROJECT_ROOT / "data" / "gqa_cleaned" / "annotations",
}

IMAGE_DIR = PROJECT_ROOT / "data" / "gqa" / "images"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "gqa_analysis"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

SPLITS = ("train", "val")
PROFILES = ("simple", "attributes", "relations", "complex")

PROFILE_TITLES = {
    "simple": "Simple scene",
    "attributes": "Attribute-rich",
    "relations": "Relation-rich",
    "complex": "Complex scene",
}

# Limits control the amount of information displayed, not the
# definition of a simple or complex scene.
MIN_OBJECTS_IN_FIGURE = 2
MAX_OBJECTS_IN_FIGURE = 14
MAX_ATTRIBUTES_IN_FIGURE = 36
MAX_RELATIONS_IN_FIGURE = 24


@dataclass(frozen=True)
class Candidate:
    """Metadata describing an eligible image and its annotation density."""

    image_id: str
    image_path: Path
    objects: int
    attributes: int
    relations: int
    data: dict | None = None


def parse_args() -> argparse.Namespace:
    """Parse the dataset version to visualize."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset",
        choices=tuple(ANNOTATION_DIRS),
        default="original",
        help="Dataset version to visualize (default: original).",
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


def index_images(image_dir: Path) -> dict[str, Path]:
    """Index supported image files by their filename stem.

    Extensions are matched case-insensitively. If multiple files
    share an ID, prefer .jpg, then .jpeg, then .png.
    """
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}"
        )

    image_paths = sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: (
            path.stem,
            IMAGE_EXTENSIONS.index(path.suffix.lower()),
            path.name,
        ),
    )

    images_by_id = {}

    for path in image_paths:
        images_by_id.setdefault(path.stem, path)

    return images_by_id


def collect_candidates(
    scene_graphs: dict,
    images_by_id: dict[str, Path],
) -> list[Candidate]:
    """Collect eligible images without retaining all their annotations.

    Candidates must satisfy the annotation limits, have a matching
    image file, and contain only resolvable relation targets.
    """
    candidates = []

    for image_id in sorted(scene_graphs):
        image_data = scene_graphs[image_id]
        objects = image_data["objects"]

        count_objects = len(objects)

        if not (
            MIN_OBJECTS_IN_FIGURE
            <= count_objects
            <= MAX_OBJECTS_IN_FIGURE
        ):
            continue

        count_attributes = sum(
            len(obj["attributes"])
            for obj in objects.values()
        )

        count_relations = sum(
            len(obj["relations"])
            for obj in objects.values()
        )

        if (
            count_attributes > MAX_ATTRIBUTES_IN_FIGURE
            or count_relations > MAX_RELATIONS_IN_FIGURE
        ):
            continue

        image_path = images_by_id.get(str(image_id))

        if image_path is None:
            continue

        object_ids = {str(object_id) for object_id in objects}

        if any(
            str(relation["object"]) not in object_ids
            for obj in objects.values()
            for relation in obj["relations"]
        ):
            continue

        candidates.append(
            Candidate(
                image_id=str(image_id),
                image_path=image_path,
                objects=count_objects,
                attributes=count_attributes,
                relations=count_relations,
            )
        )

    return candidates


def percentile_ranks(values: list[int]) -> dict[int, float]:
    """Assign relative rank scores in [0, 1] with equal scores for ties."""
    unique_values = sorted(set(values))

    if len(unique_values) == 1:
        return {unique_values[0]: 0.5}

    return {
        value: index / (len(unique_values) - 1)
        for index, value in enumerate(unique_values)
    }


def select_profiles(
    candidates: list[Candidate],
) -> dict[str, Candidate]:
    """Select four distinct candidates using profile-specific rankings.

    Each profile is ranked independently. Profiles are assigned
    in fixed order, and already selected image IDs cannot be reused.

    This deterministic greedy strategy does not optimize the
    combined scores of all four profiles globally.
    """
    if len(candidates) < len(PROFILES):
        raise ValueError(
            f"Only {len(candidates)} eligible images; "
            f"at least {len(PROFILES)} are required."
        )

    ranks = {
        key: percentile_ranks(
            [getattr(candidate, key) for candidate in candidates]
        )
        for key in ("objects", "attributes", "relations")
    }

    scores = {}

    for candidate in candidates:
        objects_rank = ranks["objects"][candidate.objects]
        attributes_rank = ranks["attributes"][candidate.attributes]
        relations_rank = ranks["relations"][candidate.relations]

        scores[candidate.image_id] = {
            "simple": -(
                objects_rank + attributes_rank + relations_rank
            ),
            "attributes": (
                2 * attributes_rank
                + 0.4 * objects_rank
                - 0.7 * relations_rank
                if candidate.attributes > 0
                else float("-inf")
            ),
            "relations": (
                2 * relations_rank
                + 0.4 * objects_rank
                - 0.7 * attributes_rank
                if candidate.relations > 0
                else float("-inf")
            ),
            "complex": (
                objects_rank + attributes_rank + relations_rank
            ),
        }

    selected = {}
    used_ids = set()

    for profile in PROFILES:
        available = [
            candidate
            for candidate in candidates
            if candidate.image_id not in used_ids
            and math.isfinite(scores[candidate.image_id][profile])
        ]

        if not available:
            raise ValueError(
                f"No distinct eligible candidate for profile: {profile}"
            )

        available.sort(
            key=lambda candidate: (
                -scores[candidate.image_id][profile],
                candidate.image_id,
            )
        )

        chosen = available[0]

        selected[profile] = chosen
        used_ids.add(chosen.image_id)

    return selected


def image_is_readable(path: Path) -> bool:
    """Check whether Pillow can fully decode an image."""
    try:
        with Image.open(path) as image:
            image.load()
        return True
    except (OSError, ValueError, UnidentifiedImageError):
        return False


def select_readable_profiles(
    candidates: list[Candidate],
) -> dict[str, Candidate]:
    """Select profiles, excluding unreadable images when encountered.

    Only selected candidates are decoded. If an image cannot be
    decoded, it is removed from the pool and selection is repeated.
    """
    remaining = candidates

    while True:
        selected = select_profiles(remaining)

        unreadable_ids = {
            candidate.image_id
            for candidate in selected.values()
            if not image_is_readable(candidate.image_path)
        }

        if not unreadable_ids:
            return selected

        for image_id in sorted(unreadable_ids):
            print(f"Skipping unreadable image: {image_id}")

        remaining = [
            candidate
            for candidate in remaining
            if candidate.image_id not in unreadable_ids
        ]


def wrap(value: object, width: int) -> str:
    """Wrap text for a table cell."""
    return (
        "\n".join(
            textwrap.wrap(
                str(value),
                width=width,
                break_long_words=True,
            )
        )
        or "—"
    )


def draw_table(
    ax,
    title: str,
    headers: list[str],
    rows: list[list[str]],
    widths: list[float],
    font_size: int = 9,
) -> None:
    """Draw a table with row heights based on wrapped text content."""
    ax.axis("off")
    ax.set_title(
        title,
        loc="left",
        fontsize=12,
        weight="bold",
        pad=8,
    )

    if not rows:
        rows = [["—"] + [""] * (len(headers) - 1)]

    all_rows = [list(headers)] + [list(row) for row in rows]

    row_units = [
        max(cell.count("\n") + 1 for cell in row) + 0.5
        for row in all_rows
    ]

    total_units = sum(row_units)

    table = ax.table(
        cellText=all_rows,
        colWidths=widths,
        cellLoc="left",
        bbox=[0, 0, 1, 1],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    y = 1.0

    for row_index, units in enumerate(row_units):
        height = units / total_units
        y -= height

        for column in range(len(headers)):
            cell = table[(row_index, column)]

            cell.set_y(y)
            cell.set_height(height)
            cell.set_edgecolor("#D6DCE4")

            cell.set_facecolor(
                "#E9EEF5"
                if row_index == 0
                else "#FFFFFF"
                if row_index % 2
                else "#F7F9FC"
            )

            cell.PAD = 0.035
            cell.get_text().set_va("center")

            if row_index == 0:
                cell.get_text().set_weight("bold")


def make_detail(
    candidate: Candidate,
    split: str,
    profile: str,
    destination: Path,
) -> None:
    """Create one detailed image with object and relation tables."""
    if candidate.data is None:
        raise ValueError("Selected candidate has no scene-graph data.")

    objects = candidate.data["objects"]
    obj_items = list(objects.items())

    visual_ids = {
        str(object_id): index + 1
        for index, (object_id, _) in enumerate(obj_items)
    }

    object_rows = []
    relation_rows = []

    for object_id, obj in obj_items:
        source_id = visual_ids[str(object_id)]

        object_rows.append([
            str(source_id),
            wrap(obj["name"], 20),
            wrap(", ".join(obj["attributes"]) or "—", 34),
        ])

        for relation in obj["relations"]:
            target_id = str(relation["object"])

            relation_rows.append([
                str(source_id),
                wrap(relation["name"], 24),
                str(visual_ids[target_id]),
            ])

    def table_units(rows: list[list[str]]) -> float:
        """Estimate the vertical space required by a table."""
        displayed = rows if rows else [["—", "", ""]]

        return (
            sum(
                max(cell.count("\n") + 1 for cell in row) + 0.5
                for row in displayed
            )
            + 1.5
        )

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
        top_margin
        + image_height
        + section_gap
        + object_height
        + section_gap
        + relation_height
        + bottom_margin
    )

    fig = plt.figure(
        figsize=(figure_width, figure_height),
        facecolor="white",
    )

    try:
        left = 0.06
        right = 0.94
        usable_width = right - left

        current_top = figure_height - top_margin

        # Annotated image.
        current_top -= image_height

        ax_image = fig.add_axes([
            left,
            current_top / figure_height,
            usable_width,
            image_height / figure_height,
        ])

        with Image.open(candidate.image_path) as image:
            ax_image.imshow(image.convert("RGB"))

        for object_id, obj in obj_items:
            x, y, w, h = (
                obj[key] for key in ("x", "y", "w", "h")
            )

            ax_image.add_patch(
                Rectangle(
                    (x, y),
                    w,
                    h,
                    linewidth=1.5,
                    edgecolor="red",
                    facecolor="none",
                )
            )

            ax_image.text(
                x,
                max(0, y - 3),
                str(visual_ids[str(object_id)]),
                fontsize=9,
                weight="bold",
                color="black",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.9,
                    "edgecolor": "red",
                    "pad": 1.5,
                },
            )

        ax_image.set_anchor("N")
        ax_image.set_title(
            f"{split.upper()} | {PROFILE_TITLES[profile]}\n"
            f"Image ID: {candidate.image_id} | "
            f"Objects: {candidate.objects} | "
            f"Attributes: {candidate.attributes} | "
            f"Relations: {candidate.relations}",
            fontsize=12,
            pad=12,
        )
        ax_image.axis("off")

        # Object and attribute table.
        current_top -= section_gap + object_height

        ax_objects = fig.add_axes([
            left,
            current_top / figure_height,
            usable_width,
            object_height / figure_height,
        ])

        draw_table(
            ax_objects,
            "Objects and attributes",
            ["ID", "Object", "Attributes"],
            object_rows,
            [0.09, 0.29, 0.62],
            font_size=10,
        )

        # Instance-level relation table.
        current_top -= section_gap + relation_height

        ax_relations = fig.add_axes([
            left,
            current_top / figure_height,
            usable_width,
            relation_height / figure_height,
        ])

        draw_table(
            ax_relations,
            "Relations",
            ["Source ID", "Relation", "Target ID"],
            relation_rows,
            [0.18, 0.64, 0.18],
            font_size=10,
        )

        fig.savefig(
            destination,
            dpi=155,
            facecolor="white",
        )

    finally:
        plt.close(fig)


def make_overview(
    selections: dict[str, dict[str, Candidate]],
    destination: Path,
) -> None:
    """Create a 2x4 overview of the selected train and val images."""
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(19, 9),
        layout="constrained",
    )

    try:
        for row, split in enumerate(SPLITS):
            for column, profile in enumerate(PROFILES):
                candidate = selections[split][profile]
                ax = axes[row, column]

                with Image.open(candidate.image_path) as image:
                    ax.imshow(image.convert("RGB"))

                ax.set_title(
                    f"{split.upper()} — {PROFILE_TITLES[profile]}\n"
                    f"ID: {candidate.image_id} | "
                    f"O:{candidate.objects} "
                    f"A:{candidate.attributes} "
                    f"R:{candidate.relations}",
                    fontsize=11,
                )

                ax.axis("off")

        fig.savefig(
            destination,
            dpi=170,
            facecolor="white",
        )

    finally:
        plt.close(fig)


def main() -> None:
    """Select contrasting examples and save nine visualization figures."""
    args = parse_args()

    annotation_dir = ANNOTATION_DIRS[args.dataset]
    output_dir = (
        OUTPUT_ROOT / args.dataset / "qualitative_examples"
    )

    print("Indexing GQA images...")
    images_by_id = index_images(IMAGE_DIR)

    selections = {}

    for split in SPLITS:
        scene_graph_path = (
            annotation_dir / f"{split}_sceneGraphs.json"
        )

        print(f"Loading {scene_graph_path}...", flush=True)
        scene_graphs = load_json(scene_graph_path)

        candidates = collect_candidates(
            scene_graphs,
            images_by_id,
        )

        print(
            f"{split}: {len(candidates)} eligible images",
            flush=True,
        )

        selected = select_readable_profiles(candidates)

        # Retain annotations only for the four selected examples.
        selections[split] = {
            profile: replace(
                candidate,
                data=scene_graphs[candidate.image_id],
            )
            for profile, candidate in selected.items()
        }

        del scene_graphs
        del candidates

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        for profile in PROFILES:
            candidate = selections[split][profile]
            filename = f"{split}_{profile}.png"

            make_detail(
                candidate,
                split,
                profile,
                output_dir / filename,
            )

            print(
                f"Saved {filename}: {candidate.image_id}",
                flush=True,
            )

    make_overview(
        selections,
        output_dir / "overview.png",
    )

    print(f"Saved 9 PNG images in: {output_dir}")


if __name__ == "__main__":
    main()
