from typing import Any, Dict, List, Optional


def aggregate_melm_outputs(melm_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Coherent dataset that can be fed into a high-level LLM.
    #
    # NOTE (future hardening ideas):
    # - deduplicate near-identical facts
    # - detect conflicts between MELMs (and either flag or reconcile)
    # - attach provenance/citations (which MELM produced which claim)
    # - rank MELM outputs by router confidence or MELM self-confidence
    #
    # Keep it JSON-first so you can benchmark, diff, and audit.
    return {
        "schema_version": "arch3.aggregate.v1",
        "melm_outputs": melm_outputs,
        "summary": {
            "num_melms": len({o.get("category") for o in melm_outputs}),
            "num_items": len(melm_outputs),
        },
    }


def build_llm_context(aggregate: Dict[str, Any], user_prompt: str, max_items: Optional[int] = None) -> str:
    # Converts aggregated MELM JSON into a plain-text context for the base LLM.
    #
    # Migration note:
    # - When MELMs become truly independent modules, this function stays mostly the same.
    # - The key stability point is the MELM output schema.
    items = aggregate.get("melm_outputs") or []
    if max_items is not None:
        items = items[: int(max_items)]

    lines: List[str] = []
    lines.append("You are given JSON outputs from modular MELMs (category experts).")
    lines.append("Use them as evidence, reconcile conflicts, and answer the user.")
    lines.append("")
    lines.append(f"USER PROMPT: {user_prompt}")
    lines.append("")
    lines.append("MELM OUTPUTS:")
    for it in items:
        cat = it.get("category", "?")
        content = it.get("content", "")
        lines.append(f"[{cat}] {content}")
    return "\n".join(lines).strip()
