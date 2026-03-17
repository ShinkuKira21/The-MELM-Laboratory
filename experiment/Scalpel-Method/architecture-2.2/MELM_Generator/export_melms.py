#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List


def resolve_path(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Export Architecture-3 MELM descriptor JSON from generator artifacts.")
    parser.add_argument("--config", default="../config.generator.json", help="Architecture-3 generator config")
    parser.add_argument("--output_dir", default=None, help="Override MELM descriptor output dir (default: artifacts/melms)")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    base_dir = os.path.dirname(config_path)

    artifacts_root = resolve_path(base_dir, cfg["export"]["output_dir"])
    experts_dir = os.path.join(artifacts_root, "experts")
    masks_dir = os.path.join(artifacts_root, "masks")
    melms_dir = resolve_path(base_dir, args.output_dir) if args.output_dir else os.path.join(artifacts_root, "melms")
    ensure_dir(melms_dir)

    domains: List[str] = [str(d) for d in cfg.get("domains", [])]

    for domain in domains:
        melm = {
            "schema_version": "arch3.melm_descriptor.v1",
            "category": domain,
            "base_model": cfg["base_model"],
            "artifacts": {
                "expert_checkpoint": os.path.join(experts_dir, f"{domain}_expert.pt"),
                "shared_mask": os.path.join(masks_dir, "shared_mask.pt"),
                "domain_mask": os.path.join(masks_dir, f"{domain}_mask.pt"),
            },
            "notes": "Descriptor generated from Arch-3 generator outputs. This is not Arch-2 linkage.",
        }
        out_path = os.path.join(melms_dir, domain, "melm.json")
        save_json(out_path, melm)
        print(f"Wrote MELM descriptor: {out_path}")


if __name__ == "__main__":
    main()

