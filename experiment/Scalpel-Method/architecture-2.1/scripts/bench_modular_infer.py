#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def maybe_reset_peak(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass


def safe_torch_load(path: str, map_location: str) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location=map_location)


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
    if not requested:
        return list(all_domains)
    return requested


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


def main():
    parser = argparse.ArgumentParser(description="Benchmark Arch-2 modular expert loading + inference.")
    parser.add_argument("--config", default="../config.example.json")
    parser.add_argument("--categories", default=None, help="Comma-separated subset of categories/domains to run")
    parser.add_argument("--prompts", default=None, help="JSON file mapping domain -> [prompts]")
    parser.add_argument("--output_dir", default=None, help="Directory to save results/logs")
    parser.add_argument("--use_masks", action="store_true", help="Build expert weights from masks instead of loading *_expert.pt")
    parser.add_argument("--include_base", action="store_true", help="Also run the unsplit base model (category '__base__')")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="auto", help="auto|float16|bfloat16|float32")
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
    if args.include_base:
        categories = ["__base__"] + categories

    prompts_map = load_prompts_map(args.prompts, [c for c in categories if c != "__base__"])

    repo_root = try_repo_root(base_dir) or base_dir
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out = os.path.join(export_dir, "benchmarks", ts)
    out_dir = os.path.abspath(args.output_dir or default_out)
    ensure_dir(out_dir)

    log_path = os.path.join(out_dir, "events.jsonl")
    results_path = os.path.join(out_dir, "results.json")
    logger = JsonlLogger(log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    model_dtype = dtype_from_arg(args.dtype, device)

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
    logger.write(
        {
            "event": "base_model_loaded",
            "seconds": base_load_sec,
            "vram": torch_vram_stats(device),
        }
    )

    base_state_cpu: Optional[Dict[str, torch.Tensor]] = None
    shared_mask_cpu: Optional[Dict[str, torch.Tensor]] = None
    if args.use_masks:
        base_state_cpu = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
        shared_mask_cpu = safe_torch_load(shared_mask_path, map_location="cpu")

    def load_category_into_model(category: str) -> Tuple[float, Dict[str, Any]]:
        maybe_reset_peak(device)
        cuda_sync()
        before = torch_vram_stats(device)
        start = time.perf_counter()

        if category == "__base__":
            # No-op: keep model weights as base.
            pass
        elif args.use_masks:
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
                if m is None:
                    masked_state[name] = t
                else:
                    masked_state[name] = t * m.to(dtype=t.dtype)
            model.load_state_dict(masked_state, strict=False)
        else:
            expert_path = os.path.join(experts_dir, f"{category}_expert.pt")
            expert_state = safe_torch_load(expert_path, map_location="cpu")
            model.load_state_dict(expert_state, strict=False)

        cuda_sync()
        seconds = time.perf_counter() - start
        after = torch_vram_stats(device)
        return seconds, {"before": before, "after": after}

    def run_prompt(prompt: str) -> Dict[str, Any]:
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
                max_new_tokens=int(args.max_new_tokens),
                do_sample=True,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
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

    results: Dict[str, Any] = {"meta": meta, "categories": {}}

    for category in categories:
        logger.write({"event": "category_start", "category": category})
        try:
            load_sec, load_mem = load_category_into_model(category)
        except FileNotFoundError as e:
            logger.write({"event": "category_skip", "category": category, "reason": "missing_artifact", "error": repr(e)})
            continue

        cat_obj: Dict[str, Any] = {
            "load_seconds": load_sec,
            "load_vram": load_mem,
            "runs": [],
        }

        if category == "__base__":
            prompts = default_prompts_for("Science")[:1]
        else:
            prompts = prompts_map.get(category, default_prompts_for(category))

        for prompt in prompts:
            logger.write({"event": "prompt_start", "category": category})
            run = run_prompt(prompt)
            logger.write({"event": "prompt_done", "category": category, "seconds": run["seconds"]})
            cat_obj["runs"].append(run)

        results["categories"][category] = cat_obj
        logger.write({"event": "category_done", "category": category, "load_seconds": load_sec})

    logger.write({"event": "run_done"})
    logger.close()
    json_dump(results_path, results)
    print(f"Wrote results to: {results_path}")
    print(f"Wrote logs to:    {log_path}")


if __name__ == "__main__":
    main()
