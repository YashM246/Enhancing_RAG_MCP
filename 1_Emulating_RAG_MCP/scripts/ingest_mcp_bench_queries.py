#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ingest_mcp_bench.py 
- Extract benchmark queries (with servers & distraction_servers) from Accenture/mcp-bench task JSONs
- Writes: 1_Emulating_RAG_MCP/data/queries/queries.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# ---------- paths ----------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DATA_DIR = REPO_ROOT / "data"
QUERIES_DIR = DATA_DIR / "queries"

# Search roots in the mcp-bench checkout that may contain task JSONs
FOLDERS_WITH_TASKS = ["tasks", "task", "benchmark", "benchmarks", "configs", "config"]


# ---------- helpers ----------
def ensure_dirs() -> None:
    QUERIES_DIR.mkdir(parents=True, exist_ok=True)


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _strip_json_comments(text: str) -> str:
    # remove // line comments
    text = re.sub(r"(?m)//.*$", "", text)
    # remove /* ... */ block comments
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    # remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def read_json(p: Path) -> Optional[Any]:
    try:
        raw = p.read_text(encoding="utf-8")
        try:
            return json.loads(raw)
        except Exception:
            cleaned = _strip_json_comments(raw)
            return json.loads(cleaned)
    except Exception:
        return None


def iter_all_dicts(obj: Any) -> Iterable[Dict[str, Any]]:
    """Generic DFS over a JSON-like structure yielding all dicts."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from iter_all_dicts(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from iter_all_dicts(it)


def normalize_query_item(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Fallback normalizer for non-`server_tasks` formats.
    Picks a reasonable 'query' string and a 'ground_truth_tool' if available.
    """
    qid = raw.get("query_id") or raw.get("id") or raw.get("qid") or raw.get("uid") or raw.get("task_id")
    text = (
        raw.get("query")
        or raw.get("prompt")
        or raw.get("question")
        or raw.get("instruction")
        or raw.get("task_description")
        or raw.get("fuzzy_description")
    )
    cat = raw.get("category") or raw.get("task_type") or raw.get("type") or raw.get("class")
    gt = raw.get("ground_truth_tool") or raw.get("gold_tool") or raw.get("target_tool") or raw.get("tool") or raw.get("expected_tool")

    if not isinstance(text, str) or not text.strip():
        # As a last resort, try to pull user message text if present
        msgs = raw.get("messages")
        if isinstance(msgs, list):
            for m in reversed(msgs):
                if isinstance(m, dict) and m.get("role") == "user":
                    c = m.get("content")
                    if isinstance(c, str) and c.strip():
                        text = c
                        break

    if not isinstance(text, str) or not text.strip():
        return None

    return {
        "query_id": qid,
        "query": text.strip(),
        "category": cat,
        "ground_truth_tool": gt,
    }


def find_task_jsons(mcp_root: Path) -> List[Any]:
    """Collect all JSON files that might contain server_tasks or task-like structures."""
    out: List[Path] = []
    for folder in FOLDERS_WITH_TASKS:
        d = mcp_root / folder
        if d.exists():
            out += list(d.rglob("*.json"))
    # Prefer the canonical task bundles first
    out.sort(
        key=lambda p: (
            "mcpbench_tasks_multi_3server_runner_format.json" not in str(p),
            "mcpbench_tasks_multi_2server_runner_format.json" not in str(p),
            "mcpbench_tasks_single_runner_format.json" not in str(p),
            str(p),
        )
    )
    return out


def _emit_query(
    bucket: List[Dict[str, Any]],
    seen: set,
    *,
    variant_text: str,
    tid: Optional[str],
    server_name: Optional[str],
    combo_type: Optional[str],
    combo_name: Optional[str],
    allowed_servers: List[str],
    distractors: List[str]
) -> None:
    key = (variant_text, server_name)
    if key in seen:
        return
    seen.add(key)
    bucket.append(
        {
            "query_id": tid,
            "query": variant_text.strip(),
            "category": combo_type,
            # lead/primary server for the block (single-server tasks) or the combo label
            "ground_truth_tool": server_name,
            # all valid servers for multi-server combos (accept any of these as correct)
            "servers": allowed_servers,
            # explicit hard negatives
            "distraction_servers": distractors,
            "combination_name": combo_name,
        }
    )


def extract_queries(mcp_root: Path) -> List[Dict[str, Any]]:
    queries: List[Dict[str, Any]] = []
    seen: set = set()

    task_files = find_task_jsons(mcp_root)
    print(f"🔎 Found {len(task_files)} task JSON files")

    for jf in task_files:
        data = read_json(jf)
        if not data:
            continue

        # Preferred structure: { "server_tasks": [ { "server_name":..., "tasks":[...], "servers":[...], "combination_name":..., "combination_type":... }, ... ] }
        if isinstance(data, dict) and isinstance(data.get("server_tasks"), list):
            for block in data["server_tasks"]:
                server_name = block.get("server_name")  # primary/lead server or combo label
                combo_name = block.get("combination_name")
                combo_type = block.get("combination_type")
                allowed_servers = block.get("servers") or []

                for task in block.get("tasks", []):
                    tid = task.get("task_id")
                    tdesc = task.get("task_description")
                    fdesc = task.get("fuzzy_description")
                    distractors = task.get("distraction_servers") or []

                    for variant in (tdesc, fdesc):
                        if isinstance(variant, str) and variant.strip():
                            _mpl = dict(
                                variant_text=variant,
                                tid=tid,
                                server_name=server_name,
                                combo_type=combo_type,
                                combo_name=combo_name,
                                allowed_servers=allowed_servers,
                                distractors=distractors,
                            )
                            _emit_query(queries, seen, **_mpl)
        else:
            # Fallback: walk all dicts and try to coerce into a query
            for d in iter_all_dicts(data):
                norm = normalize_query_item(d)
                if not norm:
                    continue
                key = (norm["query"], norm.get("ground_truth_tool"))
                if key in seen:
                    continue
                seen.add(key)
                queries.append(norm)

    print(f" Extracted {len(queries)} total queries")
    return queries


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Ingest tasks from a local clone of Accenture/mcp-bench and write queries.json")
    ap.add_argument("--mcp-bench-path", required=True, help="Path to local mcp-bench clone")
    args = ap.parse_args()

    mcp_root = Path(args.mcp_bench_path).resolve()
    if not mcp_root.exists():
        raise SystemExit(f"Path not found: {mcp_root}")

    ensure_dirs()

    print(f"\n📂 Using mcp-bench at: {mcp_root}")
    print(" - tasks/ exists?      ", (mcp_root / "tasks").exists())

    queries = extract_queries(mcp_root)

    write_json(
        QUERIES_DIR / "queries.json",
        {
            "count": len(queries),
            "queries": queries,
        },
    )
    print(f"[OK] queries → {QUERIES_DIR / 'queries.json'}")
    print("Done.\n")


if __name__ == "__main__":
    main()
