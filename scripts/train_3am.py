#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from three_am.models.three_am import ThreeAMConfig, ThreeAMCore
from three_am.training.optim import build_adamw, mark_trainable_3am_modules
from three_am.utils.config import load_yaml


def smoke_train(config_path: str, iterations: int) -> None:
    config = load_yaml(config_path)
    model_config = config["model"]
    model = ThreeAMCore(
        ThreeAMConfig(
            sam_channels=model_config["sam_channels"],
            must3r_channels=tuple(model_config["must3r_channels"]),
            hidden_channels=model_config["hidden_channels"],
            attention_heads=model_config["attention_heads"],
        )
    )
    mark_trainable_3am_modules(model)
    optimizer = build_adamw(model)
    for step in range(iterations):
        sam = torch.randn(1, model.config.sam_channels, 16, 16)
        must3r = [torch.randn(1, channels, 16, 16) for channels in model.config.must3r_channels]
        merged = model.forward_features(sam, must3r)
        loss = merged.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f"step={step + 1} loss={loss.item():.6f}")
    output = Path(config["training"]["checkpoint_out"])
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config}, output)
    print(f"Wrote smoke checkpoint to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the unofficial 3AM reimplementation")
    parser.add_argument("--config", default="configs/full_reproduction.yaml")
    parser.add_argument("--smoke", action="store_true", help="run synthetic 10-step FeatureMerger training")
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.smoke:
        smoke_train(args.config, args.iterations)
        return
    raise NotImplementedError(
        "Full training requires official SAM2 training internals, MUSt3R features, and normalized full datasets. Use --smoke to validate this scaffold."
    )


if __name__ == "__main__":
    main()
