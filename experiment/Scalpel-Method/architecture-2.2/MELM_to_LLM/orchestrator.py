#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aggregator import aggregate_melm_outputs, build_llm_context
from router import TrainedRouter, resolve_path


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any):
    ensure_dir(os.path.dirname(path) or ".")
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
    return {
        "available": True,
        "device": torch.cuda.get_device_name(device),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def safe_torch_load(path: str, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location=map_location)


def generate(
    *,
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    maybe_reset_peak(device)
    cuda_sync()
    v_before = torch_vram_stats(device)
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            pad_token_id=tokenizer.eos_token_id,
        )
    cuda_sync()
    sec = time.perf_counter() - start
    v_after = torch_vram_stats(device)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return {
        "text": text,
        "seconds": sec,
        "tokens": {
            "input": int(input_ids.shape[-1]),
            "output": int(out.shape[-1]),
            "new": int(out.shape[-1] - input_ids.shape[-1]),
        },
        "vram": {"before": v_before, "after": v_after},
    }


def load_melm_descriptor(melms_dir: str, category: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(melms_dir, category, "melm.json")
    if not os.path.isfile(path):
        return None
    return load_json(path)


def main():
    parser = argparse.ArgumentParser(description="Architecture-3 orchestrator (runtime harness).")
    parser.add_argument("--config", default="../config.orchestrator.json", help="Architecture-3 orchestrator config")
    parser.add_argument("--prompt", default=None, help="Run a single prompt and exit")
    parser.add_argument("--interactive", action="store_true", help="Interactive prompt loop (type /help)")
    parser.add_argument("--categories", default=None, help="Comma-separated categories to force (bypass router)")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    cfg = load_json(config_path)
    base_dir = os.path.dirname(config_path)

    # Resolve artifact paths strictly within architecture-3.
    artifacts = cfg["artifacts"]
    router_dir = resolve_path(base_dir, artifacts["router_dir"])
    experts_dir = resolve_path(base_dir, artifacts["experts_dir"])
    melms_dir = resolve_path(base_dir, artifacts.get("melms_dir", "artifacts/melms"))
    runs_dir = resolve_path(base_dir, artifacts["runs_dir"])

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(runs_dir, ts)
    ensure_dir(run_dir)
    events_path = os.path.join(run_dir, "events.jsonl")
    results_path = os.path.join(run_dir, "results.json")
    logger = JsonlLogger(events_path) if cfg.get("logging", {}).get("write_events_jsonl", True) else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # One model instance; we swap weights for each expert (same as Arch-2 style).
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"]).to(device)
    model.eval()
    cuda_sync()
    base_state_cpu = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}

    # Router: trained router artifacts are produced by architecture-3/MELM_Generator/router_train.py
    router_cfg = cfg.get("router", {})
    router_type = str(router_cfg.get("type", "trained"))
    trained_router: Optional[TrainedRouter] = None
    if router_type == "trained":
        checkpoint_path = resolve_path(base_dir, router_cfg["checkpoint_path"])
        manifest_path = resolve_path(base_dir, router_cfg["manifest_path"]) if router_cfg.get("manifest_path") else None
        trained_router = TrainedRouter.load(
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            device=device,
            top_k=int(router_cfg.get("top_k", 2)),
            threshold=float(router_cfg.get("threshold", 0.0)),
            embedder_max_length=256,
        )

    meta = {
        "timestamp": ts,
        "config_path": config_path,
        "device": str(device),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "transformers": None,
        "paths": {
            "router_dir": router_dir,
            "experts_dir": experts_dir,
            "melms_dir": melms_dir,
            "run_dir": run_dir,
        },
    }
    try:
        import transformers  # noqa: F401

        meta["transformers"] = transformers.__version__  # type: ignore[attr-defined]
    except Exception:
        pass

    if logger:
        logger.write({"event": "run_start", "meta": meta, "vram": torch_vram_stats(device)})

    def route(prompt: str) -> List[Dict[str, Any]]:
        if args.categories:
            cats = [c.strip() for c in args.categories.split(",") if c.strip()]
            return [{"category": c, "score": None} for c in cats]
        if trained_router is None:
            raise RuntimeError("router.type is not 'trained' or trained router failed to load.")
        return trained_router.route(tokenizer=tokenizer, prompt=prompt)

    def load_expert(category: str) -> Dict[str, Any]:
        # Prefer MELM descriptor if present, otherwise fall back to experts_dir.
        desc = load_melm_descriptor(melms_dir, category)
        ckpt_path = None
        if desc:
            ckpt_path = ((desc.get("artifacts") or {}).get("expert_checkpoint")) or None
        if not ckpt_path:
            ckpt_path = os.path.join(experts_dir, f"{category}_expert.pt")

        maybe_reset_peak(device)
        cuda_sync()
        v_before = torch_vram_stats(device)
        start = time.perf_counter()
        expert_state = safe_torch_load(ckpt_path, map_location="cpu")
        model.load_state_dict(expert_state, strict=False)
        cuda_sync()
        sec = time.perf_counter() - start
        v_after = torch_vram_stats(device)
        return {"checkpoint_path": ckpt_path, "seconds": sec, "vram": {"before": v_before, "after": v_after}}

    def restore_base():
        model.load_state_dict(base_state_cpu, strict=False)

    def run_prompt(user_prompt: str) -> Dict[str, Any]:
        selected = route(user_prompt)
        if logger:
            logger.write({"event": "route", "selected": selected})

        melm_cfg = cfg["melm"]
        melm_outputs: List[Dict[str, Any]] = []
        for sel in selected:
            cat = str(sel["category"])
            load_info = load_expert(cat)
            out = generate(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=user_prompt,
                max_new_tokens=int(melm_cfg["max_new_tokens"]),
                temperature=float(melm_cfg["temperature"]),
                top_p=float(melm_cfg["top_p"]),
            )
            melm_outputs.append(
                {
                    "schema_version": "arch3.melm_output.v1",
                    "category": cat,
                    "prompt": user_prompt,
                    "content": out["text"],
                    "metadata": {
                        "router_score": sel.get("score"),
                        "load": load_info,
                        "generation": {k: out[k] for k in ("seconds", "tokens", "vram")},
                    },
                }
            )
            if logger:
                logger.write({"event": "melm_done", "category": cat, "seconds": out["seconds"]})

        aggregate = aggregate_melm_outputs(melm_outputs)

        final = None
        final_cfg = cfg.get("final_llm", {})
        if bool(final_cfg.get("enabled", True)):
            # Restore base model weights before producing final answer.
            restore_base()
            context = build_llm_context(aggregate, user_prompt, max_items=12)
            final = generate(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=context,
                max_new_tokens=int(final_cfg.get("max_new_tokens", 160)),
                temperature=float(final_cfg.get("temperature", 0.7)),
                top_p=float(final_cfg.get("top_p", 0.95)),
            )
            final = {"text": final["text"], "seconds": final["seconds"], "vram": final["vram"]}

        return {
            "schema_version": "arch3.orchestrator_result.v1",
            "prompt": user_prompt,
            "selected": selected,
            "melm_outputs": melm_outputs,
            "aggregate": aggregate,
            "final": final,
        }

    if args.prompt:
        res = run_prompt(str(args.prompt))
        save_json(results_path, {"meta": meta, "result": res})
        if logger:
            logger.write({"event": "run_done"})
            logger.close()
        print(f"Wrote results to: {results_path}")
        print(f"Wrote logs to:    {events_path}")
        return

    if not args.interactive:
        print("Nothing to do. Use --prompt or --interactive.", file=sys.stderr)
        return

    print("Interactive mode. Commands: /help, /cats, /exit")
    while True:
        try:
            line = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd = line.strip().lower()
            if cmd in {"/exit", "/quit"}:
                break
            if cmd == "/help":
                print("  /cats  - list known categories (from router checkpoint)")
                print("  /exit  - quit")
                continue
            if cmd == "/cats":
                if trained_router is None:
                    print("router=unavailable")
                else:
                    print("categories=" + ", ".join(trained_router.domains))
                continue
            print("Unknown command. Type /help")
            continue

        res = run_prompt(line)
        if res.get("final") and isinstance(res["final"], dict):
            print(res["final"].get("text", ""))
        else:
            for o in res.get("melm_outputs", []):
                print(f"[{o.get('category')}] {o.get('content','')}")
        save_json(results_path, {"meta": meta, "last_result": res})

    if logger:
        logger.write({"event": "run_done"})
        logger.close()
    print(f"Wrote logs to:    {events_path}")
    print(f"Wrote last to:    {results_path}")


if __name__ == "__main__":
    main()

