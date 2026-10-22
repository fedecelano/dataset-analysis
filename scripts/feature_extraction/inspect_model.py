#!/usr/bin/env python3
"""Inspect SteerViT statically (YAML) or dynamically (checkpoint + forward).

Static mode needs only PyYAML and does not load neural networks.
Dynamic mode requires the SteerViT package and its dependencies, and loads a
TRUSTED SteerViT checkpoint. No features or model weights are saved.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


def inspect_config(config_path: Path) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Static mode requires PyYAML: pip install pyyaml") from exc

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Expected a YAML mapping at the document root")
    vision = config.get("vision_encoder")
    if not isinstance(vision, dict):
        raise ValueError("Missing or invalid 'vision_encoder' configuration")

    model_name = vision.get("model_name")
    resolution = vision.get("resolution")
    aggregation = vision.get("feature_aggregation")
    layers = vision.get("cross_attn_layers", [])
    text_name = config.get("text_encoder")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("'vision_encoder.model_name' must be a nonempty string")
    if not isinstance(resolution, int) or isinstance(resolution, bool) or resolution <= 0:
        raise ValueError("'vision_encoder.resolution' must be a positive integer")
    if not isinstance(layers, list) or any(
        not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in layers
    ):
        raise ValueError("'cross_attn_layers' must be a list of nonnegative integers")

    print(f"Configuration: {config_path.resolve()}")
    print("\nVision encoder")
    print(f"  model_name: {model_name}")
    print(f"  resolution: {resolution} x {resolution}")
    print(f"  feature_aggregation: {aggregation}")
    print(f"  use_ffn: {vision.get('use_ffn')}")
    print(f"  cross_attn_layers (zero-based): {layers}")
    print(f"\nText encoder\n  model_name: {text_name}")

    match = re.search(r"patch(\d+)", model_name)
    if match:
        patch_size = int(match.group(1))
        print("\nInferred from model name (not verified against loaded weights)")
        print(f"  patch_size: {patch_size}")
        if resolution % patch_size == 0:
            side = resolution // patch_size
            print(f"  patch_grid: {side} x {side}")
            print(f"  patch_tokens: {side ** 2}")
        else:
            print("  WARNING: resolution is not divisible by inferred patch_size")
    else:
        print("\nPatch size cannot be inferred from model name")
    expected_blocks = 12 if model_name.startswith("vit_base_") else None
    if expected_blocks is not None:
        print(f"\nArchitecture expectation (not verified): {expected_blocks} blocks for ViT-Base")
        out_of_range = [i for i in layers if i >= expected_blocks]
        if out_of_range:
            print(f"  WARNING: these cross-attention indices exceed the expected range: {out_of_range}")
            print("  Confirm the block count using the actual checkpoint/model.")
    print("\nNo checkpoint was loaded; no forward pass was executed.")


def resolve_device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def inspect_dynamic(checkpoint_name: str, steervit_src: Path, device_name: str, prompt: str) -> None:
    if not (steervit_src / "steervit" / "model.py").is_file():
        raise FileNotFoundError(
            f"Cannot find steervit/model.py under {steervit_src}; pass --steervit-src PATH"
        )
    sys.path.insert(0, str(steervit_src.resolve()))
    try:
        import torch
        import torch.nn.functional as F
        from huggingface_hub import hf_hub_download
        from steervit.model import SteerViT
    except ImportError as exc:
        raise RuntimeError(
            "Dynamic mode requires PyTorch and the dependencies of the SteerViT fork. "
            "Install the fork on the cluster (e.g. pip install -e ../SteerViT)."
        ) from exc

    device = resolve_device(torch, device_name)
    path = Path(checkpoint_name).expanduser()
    if not path.is_file():
        if path.name != checkpoint_name or checkpoint_name.startswith("."):
            raise FileNotFoundError(f"Local checkpoint not found: {checkpoint_name}")
        print(f"Downloading/checking checkpoint in HF cache: {checkpoint_name}", flush=True)
        path = Path(hf_hub_download(repo_id="JonaRuthardt/SteerViT", filename=checkpoint_name))

    print(f"Checkpoint: {path}", flush=True)
    print(f"Device: {device}", flush=True)
    print("SECURITY: loading a checkpoint with weights_only=False; only use trusted files.", flush=True)
    # Match the upstream SteerViT loader; its checkpoint stores config + state_dict.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not {"config", "state_dict"} <= checkpoint.keys():
        raise ValueError("Unexpected checkpoint structure: expected config and state_dict")
    config = checkpoint["config"]
    if not isinstance(config, dict):
        raise ValueError("Checkpoint config must be a dictionary")
    model = SteerViT(config)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    print(f"Load state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  Missing keys (first 15): {missing[:15]}")
    if unexpected:
        print(f"  Unexpected keys (first 15): {unexpected[:15]}")
    if missing or unexpected:
        raise RuntimeError("Checkpoint did not exactly match instantiated model; inspection aborted")
    del checkpoint
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    trunk = model.vision_model.trunk
    blocks = trunk.blocks
    n_layers = len(blocks)
    prefix = trunk.num_prefix_tokens
    patch_size = model.patch_size
    resolution = model.image_size[0]
    expected_patches = (resolution // patch_size) ** 2
    steered = [i for i, block in enumerate(blocks) if block.gated_cross_attn is not None]
    configured = config["vision_encoder"].get("cross_attn_layers", [])

    print("\nCheckpoint architecture")
    print(f"  visual backbone: {config['vision_encoder']['model_name']}")
    print(f"  image size: {model.image_size}")
    print(f"  patch size: {patch_size}")
    print(f"  block count: {n_layers}")
    print(f"  prefix tokens: {prefix}")
    print(f"  patch tokens: {expected_patches}")
    print(f"  visual dimension: {model.feature_dim}")
    print(f"  text encoder: {config['text_encoder']}")
    print(f"  text dimension: {model.text_dim}")
    print(f"  configured cross-attention layers: {configured}")
    print(f"  instantiated cross-attention layers: {steered}")
    if any(i >= n_layers for i in configured):
        print("  NOTE: out-of-range cross-attention indices are ignored by backbone.py")

    seen = {}
    handles = []

    def make_hook(idx):
        def hook(_module, _inputs, output):
            if not isinstance(output, tuple) or not torch.is_tensor(output[0]):
                raise TypeError(f"Block {idx}: expected tuple with visual tensor at index 0")
            visual = output[0]
            if idx in seen:
                raise RuntimeError(f"Block {idx}: hook invoked more than once")
            seen[idx] = (tuple(visual.shape), bool(torch.isfinite(visual).all().item()))
        return hook

    for i, block in enumerate(blocks):
        handles.append(block.register_forward_hook(make_hook(i)))

    print("\nVisual forward (synthetic image, texts=None, steering disabled)", flush=True)
    try:
        with torch.inference_mode():
            image = torch.zeros((1, 3, resolution, resolution), device=device)
            final_visual = model(image, texts=None)
            if not torch.isfinite(final_visual).all().item():
                raise ValueError("Final visual output has NaN/Inf")
    finally:
        for handle in handles:
            handle.remove()

    if set(seen) != set(range(n_layers)):
        raise RuntimeError(f"Expected {n_layers} hooks, observed {sorted(seen)}")
    for i in range(n_layers):
        shape, finite = seen[i]
        print(f"  block {i:02d} -> {shape}, finite={finite}")
        if shape != (1, prefix + expected_patches, model.feature_dim) or not finite:
            raise ValueError(f"Unexpected/nonfinite output at block {i}: {shape}")
    final_shape = tuple(final_visual.shape)
    print(f"  after final LayerNorm -> {final_shape}")
    if final_shape != (1, prefix + expected_patches, model.feature_dim):
        raise ValueError(f"Unexpected final visual shape: {final_shape}")
    del image, final_visual

    print("\nText forward (last_hidden_state only; no image conditioning)", flush=True)
    tokenizer = model.tokenizer
    inputs = tokenizer([prompt], padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        token_features = model.text_model(**inputs).last_hidden_state
        normalized = F.normalize(token_features, dim=-1)
        projected = model.connector(normalized)
        if not bool(torch.isfinite(token_features).all().item()) or not bool(torch.isfinite(projected).all().item()):
            raise ValueError("Text or connector output has NaN/Inf")
    print(f"  prompt: {prompt!r}")
    print(f"  tokenized length: {inputs['input_ids'].shape[1]}")
    print(f"  RoBERTa last hidden state: {tuple(token_features.shape)}")
    print(f"  normalized token embeddings: {tuple(normalized.shape)}")
    print(f"  connector token embeddings: {tuple(projected.shape)}")
    if projected.shape[-1] != model.feature_dim:
        raise ValueError("Connector output dimension does not match the visual embedding dimension")
    print("\nInspection passed. No features were saved. Cosine alignment was NOT evaluated.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_parent = Path(__file__).resolve().parents[3]
    parser.add_argument("--mode", choices=["static", "dynamic"], default="static")
    parser.add_argument("--config", type=Path, default=repo_parent / "SteerViT" / "configs" / "default.yaml")
    parser.add_argument("--steervit-src", type=Path, default=repo_parent / "SteerViT" / "src")
    parser.add_argument("--checkpoint", default="steervit_dinov2_base.pth", help="Trusted local checkpoint path or Hugging Face filename")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--prompt", default="This is a photo of a dog.")
    args = parser.parse_args()
    if args.mode == "static":
        if not args.config.is_file():
            parser.error(f"Configuration not found: {args.config}; specify --config PATH")
        inspect_config(args.config)
    else:
        inspect_dynamic(args.checkpoint, args.steervit_src, args.device, args.prompt)


if __name__ == "__main__":
    main()
