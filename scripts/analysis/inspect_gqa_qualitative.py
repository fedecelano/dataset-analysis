import json
import random
import textwrap

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path
from PIL import Image


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCENE_GRAPH_PATH = Path("data/gqa/annotations/train_sceneGraphs.json")
IMAGE_DIR = Path("data/gqa/images")

OUTPUT_DIR = Path("outputs/gqa_analysis/qualitative_examples")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

NUM_EXAMPLES = 8
RANDOM_SEED = 42

# Avoid extremely sparse or extremely dense examples.
MIN_OBJECTS = 4
MAX_OBJECTS = 15

# Maximum number of relations written in the text file.
MAX_RELATIONS_TO_SHOW = 30


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_json(path):
    """Load a JSON file."""

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Image lookup
# ---------------------------------------------------------------------

def find_image_path(image_id):
    """
    Find the image corresponding to a GQA image ID.

    Common image extensions are checked automatically.
    """

    extensions = [
        ".jpg",
        ".jpeg",
        ".png",
    ]

    for extension in extensions:
        path = IMAGE_DIR / f"{image_id}{extension}"

        if path.exists():
            return path

    return None


# ---------------------------------------------------------------------
# Example selection
# ---------------------------------------------------------------------

def count_attributes(image_data):
    """Count all attribute annotations in one image."""

    return sum(
        len(obj["attributes"])
        for obj in image_data["objects"].values()
    )


def count_relations(image_data):
    """Count all relation annotations in one image."""

    return sum(
        len(obj["relations"])
        for obj in image_data["objects"].values()
    )


def select_examples(scene_graphs, num_examples):
    """
    Select qualitative examples with moderate annotation density.

    Images must:
    - exist locally
    - contain a reasonable number of objects
    - contain at least one attribute
    - contain at least one relation
    """

    candidates = []

    for image_id, image_data in scene_graphs.items():
        num_objects = len(image_data["objects"])
        num_attributes = count_attributes(image_data)
        num_relations = count_relations(image_data)

        image_path = find_image_path(image_id)

        if image_path is None:
            continue

        if not (
            MIN_OBJECTS
            <= num_objects
            <= MAX_OBJECTS
        ):
            continue

        if num_attributes == 0:
            continue

        if num_relations == 0:
            continue

        candidates.append(image_id)

    if len(candidates) < num_examples:
        raise ValueError(
            f"Only {len(candidates)} suitable images found, "
            f"but {num_examples} were requested."
        )

    random.seed(RANDOM_SEED)

    return random.sample(
        candidates,
        num_examples,
    )


# ---------------------------------------------------------------------
# Annotation extraction
# ---------------------------------------------------------------------

def get_object_label(obj):
    """
    Build a compact object label containing class and attributes.
    """

    name = obj["name"]
    attributes = obj["attributes"]

    if attributes:
        attribute_text = ", ".join(attributes)

        return f"{name}\n[{attribute_text}]"

    return name


def extract_relation_strings(image_data):
    """
    Convert GQA relations into readable strings.

    Example:
        man -- riding --> bicycle
    """

    objects = image_data["objects"]

    relations = []

    for source_id, source_obj in objects.items():
        source_name = source_obj["name"]

        for relation in source_obj["relations"]:
            target_id = relation["object"]

            if target_id not in objects:
                continue

            target_name = objects[target_id]["name"]

            relations.append(
                f"{source_name} -- {relation['name']} --> {target_name}"
            )

    return relations


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def draw_annotated_image(
    image_id,
    image_data,
    image_path,
    output_path,
):
    """
    Draw bounding boxes and object labels on a GQA image.
    """

    image = Image.open(image_path).convert("RGB")

    fig, ax = plt.subplots(
        figsize=(12, 9)
    )

    ax.imshow(image)

    for object_id, obj in image_data["objects"].items():
        x = obj["x"]
        y = obj["y"]
        w = obj["w"]
        h = obj["h"]

        rectangle = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=1.5,
            edgecolor="red",
            facecolor="none",
        )

        ax.add_patch(rectangle)

        label = get_object_label(obj)

        ax.text(
            x,
            max(0, y - 3),
            label,
            fontsize=7,
            bbox={
                "facecolor": "white",
                "alpha": 0.75,
                "edgecolor": "none",
            },
        )

    ax.set_title(
        f"GQA image {image_id}"
    )

    ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ---------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------

def save_annotation_summary(
    image_id,
    image_data,
    output_path,
):
    """
    Save object, attribute, and relation annotations
    in a human-readable text file.
    """

    objects = image_data["objects"]

    lines = []

    lines.append(
        f"Image ID: {image_id}"
    )

    lines.append(
        f"Number of objects: {len(objects)}"
    )

    lines.append("")

    # -------------------------------------------------------------
    # Objects
    # -------------------------------------------------------------

    lines.append("OBJECTS")
    lines.append("-" * 70)

    for object_id, obj in objects.items():
        lines.append(
            f"{object_id}: {obj['name']}"
        )

    lines.append("")

    # -------------------------------------------------------------
    # Attributes
    # -------------------------------------------------------------

    lines.append("ATTRIBUTES")
    lines.append("-" * 70)

    for object_id, obj in objects.items():
        if not obj["attributes"]:
            continue

        attributes = ", ".join(
            obj["attributes"]
        )

        lines.append(
            f"{object_id} ({obj['name']}): {attributes}"
        )

    lines.append("")

    # -------------------------------------------------------------
    # Relations
    # -------------------------------------------------------------

    lines.append("RELATIONS")
    lines.append("-" * 70)

    relations = extract_relation_strings(
        image_data
    )

    for relation in relations[
        :MAX_RELATIONS_TO_SHOW
    ]:
        lines.append(relation)

    if len(relations) > MAX_RELATIONS_TO_SHOW:
        lines.append(
            f"... {len(relations) - MAX_RELATIONS_TO_SHOW} "
            "additional relations omitted."
        )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Optional overview figure
# ---------------------------------------------------------------------

def create_contact_sheet(example_ids):
    """
    Create a compact figure containing all selected raw images.
    """

    num_columns = 4
    num_rows = (
        len(example_ids)
        + num_columns
        - 1
    ) // num_columns

    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(16, 4 * num_rows),
    )

    axes = axes.flatten()

    for ax, image_id in zip(
        axes,
        example_ids,
    ):
        image_path = find_image_path(
            image_id
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        ax.imshow(image)
        ax.set_title(
            f"Image {image_id}"
        )
        ax.axis("off")

    for ax in axes[len(example_ids):]:
        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "qualitative_examples_overview.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading GQA scene graphs..."
    )

    scene_graphs = load_json(
        SCENE_GRAPH_PATH
    )

    print(
        "Selecting qualitative examples..."
    )

    example_ids = select_examples(
        scene_graphs,
        NUM_EXAMPLES,
    )

    for index, image_id in enumerate(
        example_ids,
        start=1,
    ):
        image_data = scene_graphs[
            image_id
        ]

        image_path = find_image_path(
            image_id
        )

        image_output = (
            OUTPUT_DIR
            / f"example_{index:02d}_{image_id}.png"
        )

        text_output = (
            OUTPUT_DIR
            / f"example_{index:02d}_{image_id}.txt"
        )

        draw_annotated_image(
            image_id=image_id,
            image_data=image_data,
            image_path=image_path,
            output_path=image_output,
        )

        save_annotation_summary(
            image_id=image_id,
            image_data=image_data,
            output_path=text_output,
        )

        print(
            f"Saved example {index}: {image_id}"
        )

    create_contact_sheet(
        example_ids
    )

    print(
        f"\nSaved qualitative examples to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()