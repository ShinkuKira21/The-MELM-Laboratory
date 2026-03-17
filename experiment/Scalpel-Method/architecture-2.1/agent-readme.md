# Architecture-2: MELM Training + Weight-Splitting (Scalpel Method)

Goal: replace the assumption-heavy demo in architecture-1 with a training-first pipeline that actually splits a pre-trained LLM into domain experts while keeping a shared backbone. This aligns with the paper's MELM framing: JIT routing, domain specialization, and weight-splitting with measurable latency and accuracy tradeoffs.

## Core ideas
- Use real domain-labeled data (multi-label) to train a router.
- Derive domain-specific parameter masks using gradient-based importance (Fisher) rather than guessing domains from prompts.
- Produce experts by combining a shared backbone + per-domain masks, then optionally fine-tune each expert (LoRA or full) on its domain.
- Keep the pipeline reproducible and measurable: TTFT, domain accuracy, cross-domain entropy.

## Architecture at a glance

Text Query
    -> Router (lightweight classifier on base-model embeddings)
        -> Top-K Domain Experts (shared backbone + domain masks)
            -> Response Synthesizer (optional, not in scope here)

Training pipeline
1. Domain taxonomy + dataset creation (multi-label examples).
2. Router training on embeddings from the base LLM.
3. Per-domain importance estimation (Fisher or gradient-based saliency).
4. Mask generation:
   - Shared mask = parameters important across domains.
   - Expert mask = parameters uniquely important to a domain.
5. Expert export: apply masks to the base model weights.
6. Expert fine-tuning (LoRA or full) per domain.
7. Router calibration (top-k / threshold).
8. Evaluation (TTFT, accuracy, cross-domain entropy).

## Why this is better than architecture-1
- Replaces toy prompt vectors with real embeddings from a base LLM.
- Uses multi-label routing (queries can be multi-domain).
- Splitting is driven by parameter importance, not a dummy classifier.
- Produces a shared backbone + experts, which is essential for JIT loading.

## Files
- config.example.json: editable pipeline configuration
- data/README.md: expected dataset format
- scripts/router_train.py: router training
- scripts/split_llm.py: Fisher-based importance, mask building, expert export
- scripts/bench_modular_infer.py: load a subset of experts and benchmark inference
- scripts/utils_data.py: JSONL loading + labeling helpers

## Output artifacts (default)
- artifacts/router/router.pt
- artifacts/masks/shared_mask.pt
- artifacts/masks/<Domain>_mask.pt
- artifacts/experts/<Domain>_expert.pt
- artifacts/manifest.json

## Notes
- Unstructured masks are simplest to implement but hard to deploy; structured masking (heads/MLP blocks) is the next step for real-world pruning.
- If TTFT does not improve, your router is still loading too much of the base model; focus on lazy-loading experts and keeping the backbone minimal.
