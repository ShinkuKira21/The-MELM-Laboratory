#!/usr/bin/env python3
import argparse
import glob
import json
import os
import statistics
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_results_json(path: str) -> bool:
    return os.path.basename(path) == "results.json"


def _iter_results_paths(inputs: List[str]) -> List[str]:
    paths: List[str] = []
    for inp in inputs:
        inp = os.path.expanduser(inp)
        if os.path.isdir(inp):
            direct = os.path.join(inp, "results.json")
            if os.path.isfile(direct):
                paths.append(direct)
                continue
            paths.extend(sorted(glob.glob(os.path.join(inp, "**", "results.json"), recursive=True)))
            continue

        if any(ch in inp for ch in ["*", "?", "["]):
            for p in sorted(glob.glob(inp)):
                if os.path.isdir(p):
                    direct = os.path.join(p, "results.json")
                    if os.path.isfile(direct):
                        paths.append(direct)
                    else:
                        paths.extend(sorted(glob.glob(os.path.join(p, "**", "results.json"), recursive=True)))
                elif os.path.isfile(p) and _is_results_json(p):
                    paths.append(p)
            continue

        if os.path.isfile(inp) and _is_results_json(inp):
            paths.append(inp)
            continue

    # de-dupe while preserving order
    seen = set()
    out: List[str] = []
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            out.append(ap)
    return out


def _safe_stat(fn, xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    try:
        return float(fn(xs))
    except Exception:
        return None


def _vram_peak_bytes(run: Dict[str, Any]) -> Optional[int]:
    after = (run.get("vram") or {}).get("after") or {}
    for k in ("max_allocated_bytes", "allocated_bytes", "max_reserved_bytes", "reserved_bytes"):
        v = after.get(k)
        if isinstance(v, int):
            return v
    return None


def _agg_category(cat_obj: Dict[str, Any]) -> Dict[str, Any]:
    runs = cat_obj.get("runs") or []
    run_seconds: List[float] = []
    peaks: List[int] = []
    new_tokens: List[int] = []
    for r in runs:
        s = r.get("seconds")
        if isinstance(s, (int, float)):
            run_seconds.append(float(s))
        p = _vram_peak_bytes(r)
        if isinstance(p, int):
            peaks.append(int(p))
        toks = (r.get("tokens") or {}).get("new")
        if isinstance(toks, int):
            new_tokens.append(int(toks))

    return {
        "num_prompts": int(len(runs)),
        "load_seconds": cat_obj.get("load_seconds"),
        "inference_seconds": {
            "mean": _safe_stat(statistics.mean, run_seconds),
            "median": _safe_stat(statistics.median, run_seconds),
            "min": _safe_stat(min, run_seconds),
            "max": _safe_stat(max, run_seconds),
        },
        "new_tokens": {
            "mean": _safe_stat(statistics.mean, [float(x) for x in new_tokens]) if new_tokens else None,
            "sum": int(sum(new_tokens)) if new_tokens else None,
        },
        "vram_peak_bytes": {
            "max": int(max(peaks)) if peaks else None,
        },
    }


def _format_bytes(n: Optional[int]) -> str:
    if n is None:
        return "-"
    # MiB for readability
    return f"{n / (1024 * 1024):.1f} MiB"


def _format_sec(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x:.3f}s"


def _print_table(rows: List[Tuple[str, str, Dict[str, Any]]]):
    headers = ["run", "category", "n", "load", "mean", "median", "peak_vram"]
    out_rows: List[List[str]] = []
    for run_id, category, agg in rows:
        inf = agg.get("inference_seconds") or {}
        peak = (agg.get("vram_peak_bytes") or {}).get("max")
        out_rows.append(
            [
                run_id,
                category,
                str(agg.get("num_prompts", "-")),
                _format_sec(agg.get("load_seconds") if isinstance(agg.get("load_seconds"), (int, float)) else None),
                _format_sec(inf.get("mean")),
                _format_sec(inf.get("median")),
                _format_bytes(peak if isinstance(peak, int) else None),
            ]
        )

    widths = [len(h) for h in headers]
    for r in out_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(r: List[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r))

    print(fmt_row(headers))
    print(fmt_row(["-" * w for w in widths]))
    for r in out_rows:
        print(fmt_row(r))


def main():
    parser = argparse.ArgumentParser(description="Compare one or more Arch-2 benchmark runs (results.json).")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more results.json paths, benchmark run directories, or globs.",
    )
    parser.add_argument("--output", default=None, help="Write comparison JSON to this path (optional).")
    parser.add_argument("--print_table", action="store_true", help="Print a human-readable table to stdout.")
    args = parser.parse_args()

    paths = _iter_results_paths(args.inputs)
    if not paths:
        print("No results.json files found from --inputs.", file=sys.stderr)
        return 2

    runs: List[Dict[str, Any]] = []
    for p in paths:
        try:
            data = _read_json(p)
        except Exception as e:
            runs.append({"path": p, "error": repr(e)})
            continue

        meta = data.get("meta") or {}
        run_id = meta.get("timestamp") or os.path.basename(os.path.dirname(p))
        categories = data.get("categories") or {}
        cat_aggs: Dict[str, Any] = {}
        for cat, obj in categories.items():
            if not isinstance(obj, dict):
                continue
            cat_aggs[str(cat)] = _agg_category(obj)
        runs.append({"path": p, "run_id": run_id, "meta": meta, "categories": cat_aggs})

    report = {"generated_by": "compare_benchmarks.py", "runs": runs}

    if args.output:
        out_path = os.path.abspath(os.path.expanduser(args.output))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Wrote comparison to: {out_path}")

    if args.print_table:
        table_rows: List[Tuple[str, str, Dict[str, Any]]] = []
        for r in runs:
            if "categories" not in r:
                continue
            run_id = str(r.get("run_id") or "?")
            cats = r.get("categories") or {}
            for cat, agg in cats.items():
                table_rows.append((run_id, str(cat), agg))
        _print_table(table_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

