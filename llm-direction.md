# LLM direction notes

Angle check: focusing on training + splitting a current LLM is correct, but only if the split is attribution-driven (importance or saliency) and you preserve a shared backbone. A naive "domain classifier + separate tiny experts" will not reflect how knowledge is distributed in a dense LLM.

Course corrections to avoid a dead end:
- Do not split weights using prompt heuristics alone. Use gradient/Fisher importance or similar attribution to decide what belongs to each expert.
- Keep shared weights for universal language and reasoning. Purely disjoint experts tend to regress on common behaviors.
- Favor structured splits (heads/MLP blocks) over unstructured masks for deployability and TTFT gains.
- Treat routing as multi-label. Most real queries span multiple domains.

Minimal viable path:
1. Domain taxonomy + multi-label dataset.
2. Router trained on base-model embeddings.
3. Per-domain Fisher importance and mask creation.
4. Shared backbone + domain experts exported as separate checkpoints.
5. Expert fine-tuning (LoRA) to recover accuracy.
6. Measure TTFT, domain accuracy, and cross-domain entropy.

If you follow the steps above, the "training + splitting" direction is sound and aligns with the MELM hypotheses.
