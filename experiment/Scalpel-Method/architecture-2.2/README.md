# Architecture-3 (Scaffold): Fully Modular MELMs

This is a **project scaffold** for a product-style MELM system where:
- MELMs are stored as standalone artifacts under `artifacts/`
- an orchestrator selects + runs MELMs and aggregates structured JSON
- a high-level LLM can optionally generate the final answer from aggregated MELM outputs

## ROCm usage

From `experiment/ROCm_Setup/`:

```sh
source ./run_rocm
```

### 1) Generate scaffold MELM artifacts

```sh
rocm-python ./Scalpel-Method/architecture-3/MELM_Generator/split_llms.py \
  --config ./Scalpel-Method/architecture-3/config.generator.json
```

### 2) Generate a placeholder router

```sh
rocm-python ./Scalpel-Method/architecture-3/MELM_Generator/router_train.py \
  --config ./Scalpel-Method/architecture-3/config.generator.json
```

### 3) Export MELM descriptors (optional)

If you want per-category descriptor JSON under `artifacts/melms/<Category>/melm.json`:

```sh
rocm-python ./Scalpel-Method/architecture-3/MELM_Generator/export_melms.py \
  --config ./Scalpel-Method/architecture-3/config.generator.json
```

### 3) Orchestrate MELMs → aggregate JSON → final answer

Single prompt:

```sh
rocm-python ./Scalpel-Method/architecture-3/MELM_to_LLM/orchestrator.py \
  --config ./Scalpel-Method/architecture-3/config.orchestrator.json \
  --prompt "Explain the scientific method and why it matters."
```

Interactive loop:

```sh
rocm-python ./Scalpel-Method/architecture-3/MELM_to_LLM/orchestrator.py \
  --config ./Scalpel-Method/architecture-3/config.orchestrator.json \
  --interactive
```

Outputs go under `Scalpel-Method/architecture-3/artifacts/runs/<timestamp>/`.

Design note (vision for independent MELM augmentation):
- `documentation/architecture-3/independent-melm-augmentation.md`

Migration/tuning checklist (what to tune in Arch-2, what to move into Arch-3):
- `documentation/architecture-3/arch2_to_arch3_migration.md`

Hard boundary rule (do not mix Arch-2 and Arch-3 at runtime):
- `documentation/architecture-3/BOUNDARIES.md`

Next steps (what to investigate when results look wrong):
- `next_steps.md`
