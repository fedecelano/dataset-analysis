import json

from pathlib import Path


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

IMAGE_DIR = Path("data/gqa/images")

TRAIN_SCENE_GRAPH_PATH = Path(
    "data/gqa/annotations/train_sceneGraphs.json"
)

VAL_SCENE_GRAPH_PATH = Path(
    "data/gqa/annotations/val_sceneGraphs.json"
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    image_files = (
        list(IMAGE_DIR.glob("*.jpg"))
        + list(IMAGE_DIR.glob("*.jpeg"))
        + list(IMAGE_DIR.glob("*.png"))
    )

    image_ids = {
        path.stem
        for path in image_files
    }

    train_scene_graphs = load_json(
        TRAIN_SCENE_GRAPH_PATH
    )

    val_scene_graphs = load_json(
        VAL_SCENE_GRAPH_PATH
    )

    train_ids = set(train_scene_graphs)
    val_ids = set(val_scene_graphs)

    scene_graph_ids = (
        train_ids
        | val_ids
    )

    print("===== GQA IMAGE CHECK =====")

    print(
        "Image files:",
        len(image_files),
    )

    print(
        "Unique image IDs:",
        len(image_ids),
    )

    print(
        "Train scene graphs:",
        len(train_ids),
    )

    print(
        "Validation scene graphs:",
        len(val_ids),
    )

    print(
        "Train + validation scene graphs:",
        len(scene_graph_ids),
    )

    print(
        "Images with train/validation scene graph:",
        len(image_ids & scene_graph_ids),
    )

    print(
        "Images without train/validation scene graph:",
        len(image_ids - scene_graph_ids),
    )

    print(
        "Scene graphs without image:",
        len(scene_graph_ids - image_ids),
    )


if __name__ == "__main__":
    main()