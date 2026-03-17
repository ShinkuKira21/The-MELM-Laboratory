# MELM Architectures (Modular Expert Language Models)
**Note:** All these architectures are based on the first path found in the "The Bottleneck of Monolithic LLMs" report.

This repository explores multiple approaches to building modular LLM systems by decomposing intelligence into domain-specific expert units (MELMs).

---

## Architecture 1 — Router + Independent Experts
- A routing model predicts domain relevance per prompt
- Each domain is handled by a separate expert model
- Experts are independently trainable, replaceable, and deployable

---

## Architecture 2 & 3 — Fisher-Split Experts
- A single base model is decomposed into domain-specific subspaces
- Uses Fisher information to identify weight importance per domain
- Produces shared + domain-specific parameter masks

---

## Core Question
Which approach leads to better:
- Modularity?
- Scalability?
- Performance?
- Interpretability?

---

## Status
- **Arch1**: Prototype (MLP-based demo, conceptual validation)
- **Arch2**: Experimental tuning (LLM-based splitting, Codex-assisted)
- **Arch3**: Experimental implementation (LLM-based splitting, Codex-assisted)

## My Viewpoint

Architecture-1 provides a strong foundational framework for modular LLM systems.

It demonstrates a clear and testable structure where:
- Domain routing is explicit
- Experts are isolated
- Behavior can be evaluated and iterated independently

This makes it a practical starting point for experimentation and system design.

In contrast, Architectures 2 & 3 explore a more advanced idea:
- Decomposing a single model into functional subspaces

While more efficient and potentially more powerful, they are:
- Harder to interpret
- More tightly coupled to the original model
- More complex to validate

At this stage, Architecture-1 feels more controllable and debuggable,  
while Architectures 2 & 3 feel more exploratory and research-oriented.

## Experimental Environment (Current Setup)

**Hardware:**
- CPU: Ryzen 9 9950X3D
- GPU: Radeon RX 9070 XT
- RAM: 64GB

**Software:**
- OS: Arch Linux
- Kernel: 6.x

**Note:** `ROCm_Setup/` is designed for RDNA4 (tested on RX 9070 XT).

---

## Notes
- Arch1 demonstrates *structural modularity* (separate models)
- Arch2/3 explore *functional modularity* (weight-space decomposition)
- Expert models in Arch1 are currently placeholders (not domain-trained)

---

## Why this exists
Most LLMs are monolithic.

This project explores whether intelligence can be:
- Split
- Routed
- Recombined

...without losing capability.

---

## Feedback
Open to ideas, criticism, and direction.

Building in the dark is tiring :D