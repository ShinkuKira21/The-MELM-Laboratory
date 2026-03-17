# Architecture-3: Fully Modular MELMs (Scaffold)

Architecture-3 is a product-oriented scaffold that separates:
- **MELM_Generator/**: produces standalone MELM artifacts (JSON + optional weights) per category.
- **MELM_to_LLM/**: loads relevant MELMs, executes them, aggregates JSON, and optionally feeds a high-level LLM.

Goals:
- MELMs can run independently of the main LLM (artifact-first execution).
- Orchestration is modular and JSON-native for downstream pipelines.
- VRAM-efficient execution by loading only selected category MELMs.

Entry points (intended to be run via `rocm-python`):
- `MELM_Generator/router_train.py`
- `MELM_Generator/split_llms.py`
- `MELM_Generator/export_melms.py` (optional MELM descriptor export)
- `MELM_to_LLM/orchestrator.py`

Configs:
- `config.generator.json` (training/splitting/export; mirrors Arch-2 config structure)
- `config.orchestrator.json` (runtime orchestration settings)
