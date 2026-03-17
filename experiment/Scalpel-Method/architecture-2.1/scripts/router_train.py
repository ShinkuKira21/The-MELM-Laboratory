#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# Allow running from repo root or scripts/.
sys.path.append(os.path.dirname(__file__))
from utils_data import build_multihot, load_jsonl, normalize_domains  # noqa: E402


def resolve_path(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


class RouterDataset(Dataset):
    def __init__(self, items: List[Dict], tokenizer, max_len: int, domain_to_idx: Dict[str, int]):
        self.items = items
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.domain_to_idx = domain_to_idx

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.items[idx]
        text = str(ex.get("text", ""))
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        labels = build_multihot(normalize_domains(ex), self.domain_to_idx)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class RouterHead(nn.Module):
    def __init__(self, emb_dim: int, num_domains: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim // 2, num_domains),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(embedder, router, dataloader, device):
    router.eval()
    total_loss = 0.0
    crit = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = embedder(input_ids=input_ids, attention_mask=attention_mask)
            emb = mean_pool(outputs.last_hidden_state, attention_mask)
            logits = router(emb)
            loss = crit(logits, labels)
            total_loss += loss.item()
    if len(dataloader) == 0:
        return 0.0
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config.example.json")
    parser.add_argument("--output", default=None, help="Override output directory")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    base_dir = os.path.dirname(config_path)
    train_path = resolve_path(base_dir, cfg["data"]["train"])
    valid_path = resolve_path(base_dir, cfg["data"]["valid"])

    domains = cfg["domains"]
    domain_to_idx = {d: i for i, d in enumerate(domains)}

    train_items = load_jsonl(train_path)
    valid_items = load_jsonl(valid_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = RouterDataset(train_items, tokenizer, cfg["router"]["max_length"], domain_to_idx)
    valid_ds = RouterDataset(valid_items, tokenizer, cfg["router"]["max_length"], domain_to_idx)

    train_loader = DataLoader(train_ds, batch_size=cfg["router"]["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg["router"]["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = AutoModel.from_pretrained(cfg["base_model"]).to(device)
    embedder.eval()

    router = RouterHead(embedder.config.hidden_size, len(domains)).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=cfg["router"]["lr"])
    criterion = nn.BCEWithLogitsLoss()

    epochs = cfg["router"]["epochs"]
    for epoch in range(epochs):
        router.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.no_grad():
                outputs = embedder(input_ids=input_ids, attention_mask=attention_mask)
                emb = mean_pool(outputs.last_hidden_state, attention_mask)
            logits = router(emb)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_loss = evaluate(embedder, router, valid_loader, device)
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1}/{epochs} - train_loss={avg_loss:.4f} val_loss={val_loss:.4f}")

    out_dir = args.output or resolve_path(base_dir, cfg["export"]["output_dir"])
    router_dir = os.path.join(out_dir, "router")
    os.makedirs(router_dir, exist_ok=True)

    torch.save(
        {
            "state_dict": router.state_dict(),
            "domains": domains,
            "base_model": cfg["base_model"],
        },
        os.path.join(router_dir, "router.pt"),
    )

    manifest = {
        "base_model": cfg["base_model"],
        "domains": domains,
        "router": {
            "top_k": cfg["router"]["top_k"],
            "threshold": cfg["router"]["threshold"],
        },
    }
    with open(os.path.join(router_dir, "router_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Router saved to {router_dir}")


if __name__ == "__main__":
    main()
