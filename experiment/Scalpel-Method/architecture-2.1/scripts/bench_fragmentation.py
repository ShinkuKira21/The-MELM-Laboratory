#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def resolve_path(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def try_git_commit(cwd: str) -> Optional[str]:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return p.stdout.strip() or None
    except Exception:
        return None


def try_repo_root(start_dir: str) -> Optional[str]:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.isdir(os.path.join(cur, ".git")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def maybe_reset_peak(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass


def torch_vram_stats(device: torch.device) -> Dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"available": False}
    try:
        return {
            "available": True,
            "device": torch.cuda.get_device_name(device),
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    except Exception as e:
        return {"available": True, "error": repr(e)}


def safe_torch_load(path: str, map_location: str) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location=map_location)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def json_dump(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


class JsonlLogger:
    def __init__(self, path: str):
        self._f = open(path, "a", encoding="utf-8")

    def write(self, event: Dict[str, Any]):
        self._f.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


def default_prompts_for(domain: str) -> List[str]:
    presets = {
        "Science": [
            "Explain the scientific method in 5 concise steps.",
            "What is the difference between accuracy and precision in experiments?",
        ],
        "History": [
            "Summarize the main causes of World War I in one paragraph.",
            "What were two major consequences of the Industrial Revolution?",
        ],
        "Literature": [
            "Give a short analysis of the theme of 'power' in a classic novel of your choice.",
            "Explain what a metaphor is, with two examples.",
        ],
        "Physics": [
            "Explain Newton's second law and give a simple example.",
            "What is the difference between velocity and acceleration?",
        ],
        "Chemistry": [
            "What is a covalent bond? Explain briefly.",
            "Explain pH and what it measures.",
        ],
        "Biology": [
            "What is DNA and what does it do?",
            "Explain natural selection in simple terms.",
        ],
        "Programming": [
            "Write a Python function that checks whether a string is a palindrome.",
            "Explain what a hashmap/dictionary is and why it is useful.",
        ],
        "Language": [
            "Explain the difference between 'their', 'there', and 'they're'.",
            "Rewrite this sentence to be more formal: 'I can't make it, sorry.'",
        ],
    }
    if domain in presets:
        return presets[domain]
    return [f"Answer this question clearly and concisely about {domain}: What is it and why does it matter?"]


def load_prompts_map(path: Optional[str], domains: List[str]) -> Dict[str, List[str]]:
    if not path:
        return {d: default_prompts_for(d) for d in domains}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, List[str]] = {}
    for d in domains:
        prompts = data.get(d)
        if isinstance(prompts, list) and prompts and all(isinstance(p, str) for p in prompts):
            out[d] = prompts
        else:
            out[d] = default_prompts_for(d)
    return out


def parse_categories(arg: Optional[str], all_domains: List[str]) -> List[str]:
    if not arg:
        return list(all_domains)
    requested = [p.strip() for p in arg.split(",") if p.strip()]
    return requested or list(all_domains)


def dtype_from_arg(arg: str, device: torch.device) -> torch.dtype:
    arg = (arg or "auto").strip().lower()
    if arg == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    if arg in {"fp16", "float16"}:
        return torch.float16
    if arg in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if arg in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported --dtype '{arg}' (use auto|float16|bfloat16|float32)")


STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "as",
    "at",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "them",
    "his",
    "her",
    "their",
    "our",
    "your",
    "my",
    "me",
    "do",
    "does",
    "did",
    "not",
    "no",
    "yes",
    "can",
    "could",
    "would",
    "should",
    "may",
    "might",
    "will",
    "just",
    "also",
    "so",
    "than",
    "into",
    "about",
    "over",
    "under",
    "between",
    "within",
}


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def word_tokens(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def content_words(text: str) -> List[str]:
    toks = word_tokens(text)
    return [t for t in toks if t not in STOPWORDS and not t.isdigit()]


def jaccard(a: Iterable[str], b: Iterable[str]) -> Optional[float]:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return None
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / float(union) if union else None


def recall(ref: Iterable[str], cand: Iterable[str]) -> Optional[float]:
    sref, scand = set(ref), set(cand)
    if not sref:
        return None
    return len(sref & scand) / float(len(sref))


def precision(ref: Iterable[str], cand: Iterable[str]) -> Optional[float]:
    sref, scand = set(ref), set(cand)
    if not scand:
        return None
    return len(sref & scand) / float(len(scand))


def f1(p: Optional[float], r: Optional[float]) -> Optional[float]:
    if p is None or r is None or (p + r) == 0:
        return None
    return 2 * p * r / (p + r)


def lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    # DP with rolling rows to keep memory small
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = cur[j - 1] if cur[j - 1] >= prev[j] else prev[j]
        prev = cur
    return prev[-1]


def rouge_l_f1(reference: str, candidate: str) -> Optional[float]:
    ref_toks = content_words(reference)
    cand_toks = content_words(candidate)
    if not ref_toks or not cand_toks:
        return None
    l = lcs_len(ref_toks, cand_toks)
    p = l / float(len(cand_toks)) if cand_toks else 0.0
    r = l / float(len(ref_toks)) if ref_toks else 0.0
    return f1(p, r)


def bleu_4(reference: str, candidate: str) -> Optional[float]:
    # Lightweight BLEU-4 on word tokens with add-1 smoothing.
    ref = word_tokens(reference)
    cand = word_tokens(candidate)
    if not ref or not cand:
        return None

    def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        return [tuple(tokens[i : i + n]) for i in range(0, max(0, len(tokens) - n + 1))]

    precisions = []
    for n in range(1, 5):
        ref_ngrams = ngrams(ref, n)
        cand_ngrams = ngrams(cand, n)
        if not cand_ngrams:
            precisions.append(0.0)
            continue
        ref_counts: Dict[Tuple[str, ...], int] = {}
        for g in ref_ngrams:
            ref_counts[g] = ref_counts.get(g, 0) + 1
        match = 0
        cand_counts: Dict[Tuple[str, ...], int] = {}
        for g in cand_ngrams:
            cand_counts[g] = cand_counts.get(g, 0) + 1
        for g, c in cand_counts.items():
            match += min(c, ref_counts.get(g, 0))
        # add-1 smoothing
        precisions.append((match + 1.0) / (len(cand_ngrams) + 1.0))

    # geometric mean
    score = 1.0
    for p in precisions:
        score *= p
    score **= 0.25

    # brevity penalty
    bp = 1.0
    if len(cand) < len(ref):
        bp = float(pow(2.718281828, 1.0 - (len(ref) / float(max(1, len(cand))))))
    return bp * score


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a / (a.norm(p=2) + 1e-12)
    b = b / (b.norm(p=2) + 1e-12)
    return float((a * b).sum().item())


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


@dataclass
class GenParams:
    max_new_tokens: int
    temperature: float
    top_p: float


def generate_text(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    gen: GenParams,
) -> Dict[str, Any]:
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    maybe_reset_peak(device)
    cuda_sync()
    before = torch_vram_stats(device)
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(gen.max_new_tokens),
            do_sample=True,
            temperature=float(gen.temperature),
            top_p=float(gen.top_p),
            pad_token_id=tokenizer.eos_token_id,
        )
    cuda_sync()
    seconds = time.perf_counter() - start
    after = torch_vram_stats(device)

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return {
        "prompt": prompt,
        "response": text,
        "seconds": seconds,
        "tokens": {
            "input": int(input_ids.shape[-1]),
            "output": int(out.shape[-1]),
            "new": int(out.shape[-1] - input_ids.shape[-1]),
        },
        "vram": {"before": before, "after": after},
    }


def load_category_state(
    category: str,
    experts_dir: str,
    masks_dir: str,
    use_masks: bool,
    base_state_cpu: Optional[Dict[str, torch.Tensor]],
    shared_mask_cpu: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    if category == "__base__":
        if base_state_cpu is None:
            raise RuntimeError("Base state not cached; can't restore __base__.")
        return base_state_cpu

    if not use_masks:
        expert_path = os.path.join(experts_dir, f"{category}_expert.pt")
        return safe_torch_load(expert_path, map_location="cpu")

    if base_state_cpu is None or shared_mask_cpu is None:
        raise RuntimeError("use_masks requested but base_state/shared_mask not loaded")

    dom_mask_path = os.path.join(masks_dir, f"{category}_mask.pt")
    dom_mask_cpu = safe_torch_load(dom_mask_path, map_location="cpu")
    merged: Dict[str, torch.Tensor] = {}
    for name in shared_mask_cpu:
        merged[name] = shared_mask_cpu[name] | dom_mask_cpu[name]

    masked_state: Dict[str, torch.Tensor] = {}
    for name, t in base_state_cpu.items():
        m = merged.get(name)
        masked_state[name] = t if m is None else (t * m.to(dtype=t.dtype))
    return masked_state


def metrics_against_full(full_text: str, modular_text: str, embed_sim: Optional[float]) -> Dict[str, Any]:
    ref_words = word_tokens(full_text)
    cand_words = word_tokens(modular_text)
    ref_content = content_words(full_text)
    cand_content = content_words(modular_text)

    cw_recall = recall(ref_content, cand_content)
    cw_prec = precision(ref_content, cand_content)
    cw_f1 = f1(cw_prec, cw_recall)

    rouge = rouge_l_f1(full_text, modular_text)
    bleu = bleu_4(full_text, modular_text)

    # "Fragmentation" proxy: missing content from full model.
    # Higher means more missing/degraded relative to full.
    fragmentation = None if cw_recall is None else float(1.0 - cw_recall)

    return {
        "length_ratio_words": (len(cand_words) / float(len(ref_words))) if ref_words else None,
        "token_overlap_jaccard": jaccard(ref_words, cand_words),
        "content_word": {"precision": cw_prec, "recall": cw_recall, "f1": cw_f1},
        "fragmentation_score": fragmentation,
        "rouge_l_f1": rouge,
        "bleu_4": bleu,
        "semantic_cosine": embed_sim,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Arch-2 modular experts vs full model (fragmentation/data loss).")
    parser.add_argument("--config", default="../config.example.json")
    parser.add_argument("--categories", default=None, help="Comma-separated subset of categories/domains to run")
    parser.add_argument("--prompts", default=None, help="JSON file mapping domain -> [prompts]")
    parser.add_argument("--output_dir", default=None, help="Directory to save results/logs")
    parser.add_argument("--use_masks", action="store_true", help="Build expert weights from masks instead of loading *_expert.pt")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="auto", help="auto|float16|bfloat16|float32")
    parser.add_argument("--embedder_device", default="cpu", help="cpu|cuda|auto (for semantic cosine metric)")
    parser.add_argument("--no_semantic", action="store_true", help="Disable semantic cosine metric")
    parser.add_argument("--interactive", action="store_true", help="Interactive prompt loop (type /help)")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    base_dir = os.path.dirname(config_path)

    export_dir = resolve_path(base_dir, cfg["export"]["output_dir"])
    experts_dir = os.path.join(export_dir, "experts")
    masks_dir = os.path.join(export_dir, "masks")
    shared_mask_path = os.path.join(masks_dir, "shared_mask.pt")

    domains = list(cfg["domains"])
    categories = parse_categories(args.categories, domains)
    prompts_map = load_prompts_map(args.prompts, categories)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath(args.output_dir or os.path.join(export_dir, "benchmarks_fragmentation", ts))
    ensure_dir(out_dir)
    log_path = os.path.join(out_dir, "events.jsonl")
    results_path = os.path.join(out_dir, "results.json")
    logger = JsonlLogger(log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    model_dtype = dtype_from_arg(args.dtype, device)
    gen = GenParams(max_new_tokens=int(args.max_new_tokens), temperature=float(args.temperature), top_p=float(args.top_p))

    repo_root = try_repo_root(base_dir) or base_dir
    meta = {
        "timestamp": ts,
        "config_path": config_path,
        "base_model": cfg["base_model"],
        "export_dir": export_dir,
        "experts_dir": experts_dir,
        "masks_dir": masks_dir,
        "device": str(device),
        "dtype": str(model_dtype).replace("torch.", ""),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "transformers": None,
        "git_commit": try_git_commit(repo_root),
        "args": vars(args),
    }

    try:
        import transformers  # noqa: F401

        meta["transformers"] = transformers.__version__  # type: ignore[attr-defined]
    except Exception:
        pass

    logger.write({"event": "run_start", "meta": meta})

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=model_dtype)
    model.to(device)
    model.eval()
    cuda_sync()
    base_load_sec = time.perf_counter() - load_start
    logger.write({"event": "base_model_loaded", "seconds": base_load_sec, "vram": torch_vram_stats(device)})

    # Cache base state for restoring full-model baseline after swapping expert weights.
    base_state_cpu = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    shared_mask_cpu = safe_torch_load(shared_mask_path, map_location="cpu") if args.use_masks else None

    # Embedder for semantic similarity (optional).
    embedder = None
    embedder_device = torch.device("cpu")
    if not args.no_semantic:
        ed = (args.embedder_device or "cpu").strip().lower()
        if ed == "auto":
            embedder_device = device
        elif ed == "cuda":
            embedder_device = device
        else:
            embedder_device = torch.device("cpu")

        try:
            embedder = AutoModel.from_pretrained(cfg["base_model"])
            embedder.to(embedder_device)
            embedder.eval()
            logger.write({"event": "embedder_loaded", "device": str(embedder_device)})
        except Exception as e:
            embedder = None
            logger.write({"event": "embedder_disabled", "reason": "load_failed", "error": repr(e)})

    def embed_text(text: str) -> Optional[torch.Tensor]:
        if embedder is None:
            return None
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        input_ids = enc["input_ids"].to(embedder_device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(embedder_device)
        with torch.no_grad():
            out = embedder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = mean_pool(out.last_hidden_state, attention_mask).squeeze(0).detach().to("cpu")
        return pooled

    current_loaded = "__base__"

    def load_into_model(category: str) -> Dict[str, Any]:
        nonlocal current_loaded
        maybe_reset_peak(device)
        cuda_sync()
        before = torch_vram_stats(device)
        start = time.perf_counter()
        state = load_category_state(
            category=category,
            experts_dir=experts_dir,
            masks_dir=masks_dir,
            use_masks=bool(args.use_masks),
            base_state_cpu=base_state_cpu,
            shared_mask_cpu=shared_mask_cpu,
        )
        model.load_state_dict(state, strict=False)
        cuda_sync()
        seconds = time.perf_counter() - start
        after = torch_vram_stats(device)
        current_loaded = category
        return {"seconds": seconds, "vram": {"before": before, "after": after}}

    def run_one(prompt: str, category: str) -> Dict[str, Any]:
        # Full-model baseline (base weights).
        if current_loaded != "__base__":
            load_into_model("__base__")
        full = generate_text(model, tokenizer, device, prompt, gen)

        # Modular category model.
        if category != "__base__":
            load_into_model(category)
        modular = generate_text(model, tokenizer, device, prompt, gen)

        # Metrics
        sim = None
        if embedder is not None:
            ef = embed_text(full["response"])
            em = embed_text(modular["response"])
            if ef is not None and em is not None:
                sim = cosine_sim(ef, em)

        metrics = metrics_against_full(full["response"], modular["response"], sim)
        return {"prompt": prompt, "full": full, "modular": modular, "metrics": metrics}

    results: Dict[str, Any] = {"meta": meta, "categories": {}}

    if args.interactive:
        # Load a default category to start, if provided.
        if categories:
            try:
                load_into_model(categories[0])
            except Exception:
                load_into_model("__base__")

        print("Interactive mode. Commands: /help, /cat <Name>, /base, /loaded, /exit")
        while True:
            try:
                prompt = input("prompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not prompt:
                continue
            if prompt.startswith("/"):
                parts = prompt.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""
                if cmd in {"/exit", "/quit"}:
                    break
                if cmd == "/help":
                    print("  /cat <Category>  - load a category expert (e.g., Science)")
                    print("  /base            - restore full/base model weights")
                    print("  /loaded          - print currently loaded model")
                    print("  /exit            - quit")
                    continue
                if cmd == "/loaded":
                    print(f"loaded={current_loaded}")
                    continue
                if cmd == "/base":
                    info = load_into_model("__base__")
                    print(f"loaded=__base__ load_seconds={info['seconds']:.3f}")
                    continue
                if cmd == "/cat":
                    if not arg:
                        print("usage: /cat <Category>")
                        continue
                    try:
                        info = load_into_model(arg)
                        print(f"loaded={arg} load_seconds={info['seconds']:.3f}")
                    except Exception as e:
                        print(f"Failed to load category '{arg}': {e}")
                    continue
                print("Unknown command. Type /help")
                continue

            # Answer with current loaded model; also show which expert is active.
            logger.write({"event": "interactive_prompt", "loaded": current_loaded})
            out = generate_text(model, tokenizer, device, prompt, gen)
            print(f"[loaded={current_loaded}]")
            print(out["response"])
        logger.write({"event": "interactive_done"})

        # Save a minimal interactive transcript placeholder.
        json_dump(results_path, {"meta": meta, "interactive": True})
        logger.close()
        print(f"Wrote logs to:    {log_path}")
        print(f"Wrote results to: {results_path}")
        return

    for category in categories:
        logger.write({"event": "category_start", "category": category})
        try:
            load_info = load_into_model(category)
        except FileNotFoundError as e:
            logger.write({"event": "category_skip", "category": category, "reason": "missing_artifact", "error": repr(e)})
            continue

        cat_obj: Dict[str, Any] = {
            "load": load_info,
            "runs": [],
        }
        for prompt in prompts_map.get(category, default_prompts_for(category)):
            logger.write({"event": "prompt_start", "category": category})
            run = run_one(prompt, category)
            logger.write({"event": "prompt_done", "category": category, "seconds_full": run["full"]["seconds"], "seconds_mod": run["modular"]["seconds"]})
            cat_obj["runs"].append(run)

        results["categories"][category] = cat_obj
        logger.write({"event": "category_done", "category": category})

    logger.write({"event": "run_done"})
    logger.close()
    json_dump(results_path, results)
    print(f"Wrote results to: {results_path}")
    print(f"Wrote logs to:    {log_path}")


if __name__ == "__main__":
    main()

