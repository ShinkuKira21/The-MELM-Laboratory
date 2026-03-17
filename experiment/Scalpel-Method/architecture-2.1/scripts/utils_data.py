import json
from typing import Dict, Iterable, List


def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def normalize_domains(example: Dict) -> List[str]:
    if "domains" in example and example["domains"]:
        return [str(d).strip() for d in example["domains"]]
    if "domain" in example and example["domain"]:
        return [str(example["domain"]).strip()]
    if "labels" in example and example["labels"]:
        return [str(d).strip() for d in example["labels"]]
    return []


def build_multihot(domains: Iterable[str], domain_to_idx: Dict[str, int]) -> List[float]:
    vec = [0.0] * len(domain_to_idx)
    for d in domains:
        if d in domain_to_idx:
            vec[domain_to_idx[d]] = 1.0
    return vec


def filter_by_domain(items: List[Dict], domain: str) -> List[Dict]:
    out = []
    for ex in items:
        ds = normalize_domains(ex)
        if domain in ds:
            out.append(ex)
    return out
