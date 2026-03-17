from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel


def resolve_path(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class RouterHead(nn.Module):
    # Must match the architecture used in MELM_Generator/router_train.py
    def __init__(self, emb_dim: int, num_domains: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim // 2, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _safe_torch_load(path: str, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class TrainedRouter:
    base_model: str
    domains: List[str]
    top_k: int
    threshold: float
    embedder_max_length: int
    embedder: Any
    head: RouterHead
    device: torch.device

    @classmethod
    def load(
        cls,
        *,
        checkpoint_path: str,
        manifest_path: Optional[str],
        device: torch.device,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        embedder_max_length: int = 256,
    ) -> "TrainedRouter":
        ckpt = _safe_torch_load(checkpoint_path, map_location="cpu")
        domains = [str(d) for d in (ckpt.get("domains") or [])]
        base_model = str(ckpt.get("base_model") or "")

        man = None
        if manifest_path and os.path.isfile(manifest_path):
            man = _load_json(manifest_path)
            if not base_model:
                base_model = str(man.get("base_model") or "")
            if not domains:
                domains = [str(d) for d in (man.get("domains") or [])]

        if not base_model or not domains:
            raise ValueError("Router checkpoint/manifest missing base_model or domains.")

        # Use config overrides if provided, else fall back to manifest, else hard defaults.
        if top_k is None:
            try:
                top_k = int(((man or {}).get("router") or {}).get("top_k"))
            except Exception:
                top_k = 2
        if threshold is None:
            try:
                threshold = float(((man or {}).get("router") or {}).get("threshold"))
            except Exception:
                threshold = 0.0

        embedder = AutoModel.from_pretrained(base_model).to(device)
        embedder.eval()

        head = RouterHead(embedder.config.hidden_size, len(domains)).to(device)
        head.load_state_dict(ckpt["state_dict"])
        head.eval()

        return cls(
            base_model=base_model,
            domains=domains,
            top_k=int(top_k),
            threshold=float(threshold),
            embedder_max_length=int(embedder_max_length),
            embedder=embedder,
            head=head,
            device=device,
        )

    @torch.no_grad()
    def route(self, *, tokenizer, prompt: str) -> List[Dict[str, Any]]:
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(self.embedder_max_length),
            padding="max_length",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        out = self.embedder(input_ids=input_ids, attention_mask=attention_mask)
        emb = mean_pool(out.last_hidden_state, attention_mask)
        logits = self.head(emb).squeeze(0)
        probs = torch.sigmoid(logits).detach().float().cpu().tolist()

        scored = [{"category": self.domains[i], "score": float(probs[i])} for i in range(len(self.domains))]
        scored.sort(key=lambda x: x["score"], reverse=True)

        selected = [s for s in scored if s["score"] >= float(self.threshold)]
        if not selected:
            selected = scored[: max(1, int(self.top_k))]
        else:
            selected = selected[: max(1, int(self.top_k))]
        return selected

