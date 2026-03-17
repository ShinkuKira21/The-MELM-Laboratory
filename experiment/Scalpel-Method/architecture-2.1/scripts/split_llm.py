#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow running from repo root or scripts/.
sys.path.append(os.path.dirname(__file__))
from utils_data import filter_by_domain, load_jsonl  # noqa: E402


def resolve_path(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DomainCausalDataset(Dataset):
    def __init__(self, items: List[Dict], tokenizer, max_len: int):
        self.items = items
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.items[idx].get("text", ""))
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


def compute_fisher(
    model,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> Dict[str, torch.Tensor]:
    fisher = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            fisher[name] = torch.zeros_like(p, device="cpu")

    model.train()
    steps = 0
    for batch in dataloader:
        for k in batch:
            batch[k] = batch[k].to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            fisher[name] += (p.grad.detach().float().cpu() ** 2)
        model.zero_grad(set_to_none=True)
        steps += 1
        if steps >= max_batches:
            break

    if steps == 0:
        return fisher

    for name in fisher:
        fisher[name] /= float(steps)
    return fisher


def topk_mask(t: torch.Tensor, ratio: float) -> torch.Tensor:
    if ratio <= 0.0:
        return torch.zeros_like(t, dtype=torch.bool)
    if ratio >= 1.0:
        return torch.ones_like(t, dtype=torch.bool)
    flat = t.flatten()
    k = max(1, int(flat.numel() * ratio))
    if k >= flat.numel():
        return torch.ones_like(t, dtype=torch.bool)
    idx = torch.topk(flat, k).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[idx] = True
    return mask.view_as(t)


def build_masks(
    fisher_by_domain: Dict[str, Dict[str, torch.Tensor]],
    shared_ratio: float,
    expert_ratio: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
    domains = list(fisher_by_domain.keys())
    avg = {}
    for name in fisher_by_domain[domains[0]]:
        avg[name] = sum(fisher_by_domain[d][name] for d in domains) / float(len(domains))

    shared_mask = {}
    expert_masks = {d: {} for d in domains}
    for name, avg_tensor in avg.items():
        shared_mask[name] = topk_mask(avg_tensor, shared_ratio)
        for d in domains:
            rel = fisher_by_domain[d][name] / (avg_tensor + 1e-8)
            expert_mask = topk_mask(rel, expert_ratio)
            expert_masks[d][name] = expert_mask & (~shared_mask[name])
    return shared_mask, expert_masks


def apply_mask(state_dict: Dict[str, torch.Tensor], mask: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for name, tensor in state_dict.items():
        if name in mask:
            m = mask[name].to(dtype=tensor.dtype)
            out[name] = tensor * m
        else:
            out[name] = tensor
    return out


def merge_masks(shared_mask: Dict[str, torch.Tensor], expert_mask: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    merged = {}
    for name in shared_mask:
        merged[name] = shared_mask[name] | expert_mask[name]
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config.example.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    base_dir = os.path.dirname(config_path)
    train_path = resolve_path(base_dir, cfg["data"]["train"])

    domains = cfg["domains"]
    set_seed(cfg["split"].get("seed", 7))

    items = load_jsonl(train_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"]).to(device)

    fisher_by_domain = {}
    for domain in domains:
        domain_items = filter_by_domain(items, domain)
        ds = DomainCausalDataset(domain_items, tokenizer, cfg["router"]["max_length"])
        dl = DataLoader(ds, batch_size=cfg["router"]["batch_size"], shuffle=True)
        print(f"Computing Fisher for {domain} on {len(ds)} samples")
        fisher_by_domain[domain] = compute_fisher(
            model,
            dl,
            device=device,
            max_batches=cfg["split"]["max_batches"],
        )

    shared_mask, expert_masks = build_masks(
        fisher_by_domain,
        shared_ratio=cfg["split"]["shared_ratio"],
        expert_ratio=cfg["split"]["expert_ratio"],
    )

    out_dir = args.output or resolve_path(base_dir, cfg["export"]["output_dir"])
    mask_dir = os.path.join(out_dir, "masks")
    expert_dir = os.path.join(out_dir, "experts")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(expert_dir, exist_ok=True)

    torch.save(shared_mask, os.path.join(mask_dir, "shared_mask.pt"))
    for domain in domains:
        torch.save(expert_masks[domain], os.path.join(mask_dir, f"{domain}_mask.pt"))

    # Export expert checkpoints by applying shared + domain masks.
    model = model.to("cpu")
    base_state = model.state_dict()
    for domain in domains:
        merged = merge_masks(shared_mask, expert_masks[domain])
        expert_state = apply_mask(base_state, merged)
        torch.save(expert_state, os.path.join(expert_dir, f"{domain}_expert.pt"))

    manifest = {
        "base_model": cfg["base_model"],
        "domains": domains,
        "split": cfg["split"],
        "notes": "Masks are unstructured; consider structured head/MLP pruning for deployment.",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Masks saved to {mask_dir}")
    print(f"Experts saved to {expert_dir}")


if __name__ == "__main__":
    main()
