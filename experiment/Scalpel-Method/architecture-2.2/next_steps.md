# Architecture-3 Next Steps (Debug + Improve)

This file captures what to investigate next based on current Arch-3 runs and artifacts.

## 0) Sanity: confirm artifacts exist

Expected after running generator:
- `artifacts/router/router.pt`
- `artifacts/router/router_manifest.json`
- `artifacts/masks/shared_mask.pt`
- `artifacts/masks/<Domain>_mask.pt`
- `artifacts/experts/<Domain>_expert.pt`
- `artifacts/manifest.json`

Optional (descriptor export):
- `artifacts/melms/<Domain>/melm.json`

## 1) Routing looks plausible, but tune selection first

Current behavior: router often selects an unrelated second category because:
- `top_k` is > 1, and/or
- `threshold` is low enough that extra domains pass.

Actions:
- In `config.orchestrator.json`, temporarily set:
  - `router.top_k = 1`
  - or raise `router.threshold` until only the obvious category triggers.
- Use `--categories <A,B>` on the orchestrator to bypass routing and isolate expert quality.

Why this matters:
- If routing is noisy, you can’t tell whether failures come from the expert itself or from the wrong expert being selected.

## 2) The big issue: expert text quality is degraded / gibberish

If MELM outputs look like nonsense, the root cause is usually one of:
- masks are too aggressive (too many parameters zeroed)
- training data is too small / too noisy / mis-labeled
- Fisher estimation is under-sampled (`max_batches` too low)
- generation parameters amplify instability (sampling too hot)

Actions to isolate:
- Force a single domain:
  - `orchestrator.py --categories Programming --prompt "..."`
- Compare base model vs expert for same prompt:
  - add a quick “baseline generate” mode (or temporarily disable expert swapping).
- Reduce generation randomness for debugging:
  - set `temperature=0.1` and `top_p=0.9` (or even greedy decoding) to see if coherence returns.

Generator knobs to tune (in `config.generator.json`):
- `split.shared_ratio`
- `split.expert_ratio`
- `split.max_batches`
- dataset size/quality in `data/domain_train.jsonl` and `data/domain_valid.jsonl`

Expectation:
- With unstructured masks, “expert” checkpoints may still behave badly as pure generators; you may need structured masking or fine-tuning of experts after splitting.

## 3) Know what Arch-3 is (and isn’t) proving about VRAM

Current runtime swaps full-shaped expert checkpoints into a single loaded base model instance.
- This does NOT demonstrate “partial model loaded into VRAM”.
- VRAM may remain roughly constant across experts.

Actions:
- Treat current VRAM numbers as “swap overhead” and “inference cost per expert”, not true modular memory savings.
- For real VRAM modularity, plan a future runtime with:
  - structured pruning (smaller model per MELM), or
  - independent MELM runtimes (tool/RAG/small model) that can run without the base LLM.

## 4) Make MELM outputs truly structured (JSON-first)

Right now `melm_outputs[*].content` is plain text.
To support reliable downstream aggregation:
- define a strict JSON schema for MELM outputs
- have MELMs produce structured fields (`facts`, `steps`, `code_snippets`, `sources`, etc.)
- add validation + repair (or reject) at orchestrator boundary

This is crucial for your “augmentation” goal:
Prompt → MELMs → JSON packets → final LLM

## 5) Short, practical experiment plan

1) Set `router.top_k=1` in `config.orchestrator.json`.
2) Run 3 prompts per domain (Programming, Biology, History).
3) Force each domain with `--categories <Domain>` to check expert quality.
4) If experts are gibberish:
   - reduce `split.expert_ratio` (less aggressive)
   - increase `split.max_batches`
   - rerun `split_llms.py`
5) If expert quality improves, re-enable router and tune `threshold`.

