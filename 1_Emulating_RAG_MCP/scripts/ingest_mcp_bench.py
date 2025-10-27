#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ingest_mcp_bench.py
- Extract benchmark queries (with servers & distraction_servers) from Accenture/mcp-bench task JSONs
- Build a basic tool catalog (prefer catalogs if present; else parse mcp_servers/*/server.py)
Outputs:
  1_Emulating_RAG_MCP/data/queries/queries.json
  1_Emulating_RAG_MCP/data/tools/tool_catalog.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --- Paths (relative to this script)
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DATA_DIR = REPO_ROOT / "data"
TOOLS_DIR = DATA_DIR / "tools"
QUERIES_DIR = DATA_DIR / "queries"

# Where to look for task jsons
FOLDERS_WITH_TASKS = ["tasks", "task", "benchmark", "benchmarks", "configs", "config"]

# ---------------- helpers ----------------

def ensure_dirs() -> None:
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    QUERIES_DIR.mkdir(parents=True, exist_ok=True)

def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _strip_json_comments(text: str) -> str:
    # remove // line comments
    text = re.sub(r"(?m)//.*$", "", text)
    # remove /* block comments */
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

# ---------------- Generic JSON walkers ----------------

def iter_all_dicts(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from iter_all_dicts(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from iter_all_dicts(it)

def normalize_query_item(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Fallback normalizer for non-server_tasks shapes (kept for robustness).
    """
    qid = raw.get("query_id") or raw.get("id") or raw.get("qid") or raw.get("uid") or raw.get("task_id")
    text = (
        raw.get("query") or raw.get("prompt") or raw.get("question") or raw.get("instruction")
        or raw.get("task_description") or raw.get("fuzzy_description")
    )
    cat  = raw.get("category") or raw.get("task_type") or raw.get("type") or raw.get("class")
    gt   = (
        raw.get("ground_truth_tool") or raw.get("gold_tool") or raw.get("target_tool")
        or raw.get("tool") or raw.get("expected_tool")
    )
    if not isinstance(text, str) or not text.strip():
        # try chat messages fallback
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
        "ground_truth_tool": gt
    }

# ---------------- Search for task JSONs ----------------

def find_task_jsons(mcp_root: Path) -> List[Path]:
    out: List[Path] = []
    for folder in FOLDERS_WITH_TASKS:
        d = mcp_root / folder
        if d.exists():
            out += list(d.rglob("*.json"))
    # prioritize canonical multi_3server runner file if present
    out_sorted = sorted(
        out,
        key=lambda p: (
            "mcpbench_tasks_multi_3server_runner_format.json" not in str(p),
            "mcpbench_tasks_multi_2server_runner_format.json" not in str(p),
            "mcpbench_tasks_single_runner_format.json" not in str(p),
            str(p)
        )
    )
    return out_sorted

# ---------------- Queries extraction (server_tasks-aware) ----------------

def _emit_query(
    queries: List[Dict[str, Any]],
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
    queries.append({
        "query_id": tid,
        "query": variant_text.strip(),
        "category": combo_type,
        # Lead/primary — for single-server this is the right answer; for multi it’s the block label
        "ground_truth_tool": server_name,
        # Valid servers set (accept any of these as correct in multi-server combos)
        "servers": allowed_servers,
        # Hard negatives
        "distraction_servers": distractors,
        "combination_name": combo_name
    })

def extract_queries(mcp_root: Path) -> List[Dict[str, Any]]:
    queries: List[Dict[str, Any]] = []
    seen: set = set()
    task_files = find_task_jsons(mcp_root)
    print(f"🔎 Found {len(task_files)} task JSON files")

    for jf in task_files:
        data = read_json(jf)
        if not data:
            continue

        # Preferred shape: {"server_tasks":[ { "server_name":..., "tasks":[...], "servers":[...], ... } ]}
        if isinstance(data, dict) and isinstance(data.get("server_tasks"), list):
            for block in data["server_tasks"]:
                server_name = block.get("server_name")  # primary/lead label
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
                            _emit_query(
                                queries, seen,
                                variant_text=variant,
                                tid=tid,
                                server_name=server_name,
                                combo_type=combo_type,
                                combo_name=combo_name,
                                allowed_servers=allowed_servers,
                                distractors=distractors
                            )
        else:
            # Fallback scan for any other formats
            found_here = 0
            for d in iter_all_dicts(data):
                norm = normalize_query_item(d)
                if not norm:
                    continue
                k = (norm["query"], norm.get("ground_truth_tool"))
                if k in seen:
                    continue
                seen.add(k)
                queries.append(norm)
                found_here += 1

    print(f"✅ Extracted {len(queries)} total queries")
    return queries

# ---------------- Tool catalog extraction ----------------

# Regex for Python Tool(...) registrations in server.py
PY_TOOL_BLOCK_RE = re.compile(r"Tool\s*\(.*?\)", re.S)
NAME_RE = re.compile(r"name\s*=\s*['\"]([^'\"]+)['\"]")
DESC_RE = re.compile(r"description\s*=\s*['\"]([^'\"]+)['\"]")

# Simple TS/JS pattern (best effort)
TSJS_NAME_RE = re.compile(r"\bname\s*:\s*['\"]([^'\"]+)['\"]")
TSJS_DESC_RE = re.compile(r"\bdescription\s*:\s*['\"]([^'\"]+)['\"]")

def parse_python_server_tools(server_py: Path) -> List[Dict[str, Any]]:
    try:
        text = server_py.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    tools: List[Dict[str, Any]] = []
    blocks = PY_TOOL_BLOCK_RE.findall(text)
    for block in blocks:
        name = None
        desc = None
        m1 = NAME_RE.search(block)
        m2 = DESC_RE.search(block)
        if m1:
            name = m1.group(1)
        if m2:
            desc = m2.group(1)
        if name:
            tools.append({"tool_name": name, "tool_desc": desc})
    return tools

def parse_tsjs_server_tools(server_file: Path) -> List[Dict[str, Any]]:
    try:
        text = server_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    names = TSJS_NAME_RE.findall(text)
    descs = TSJS_DESC_RE.findall(text)
    out: List[Dict[str, Any]] = []
    for i, nm in enumerate(names):
        ds = descs[i] if i < len(descs) else None
        out.append({"tool_name": nm, "tool_desc": ds})
    return out

def extract_tools(mcp_root: Path) -> List[Dict[str, Any]]:
    """
    Best-effort tool catalog:
      1) Look for any consolidated catalogs under synthesis/configs/tools/catalog/data (jsons).
      2) Else parse mcp_servers/*/server.(py|ts|js) and collect Tool registrations.
    Produces a flat list with server_id.
    """
    # 1) Prefer existing catalogs (if mcp-bench ships them)
    catalogs: List[Path] = []
    for guess in ["synthesis", "configs", "tools", "catalog", "data"]:
        g = mcp_root / guess
        if g.exists():
            catalogs += list(g.rglob("*.json"))

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for jf in catalogs:
        data = read_json(jf)
        if not data:
            continue
        if isinstance(data, list) and data and isinstance(data[0], dict) and "tool_name" in data[0]:
            grouped.setdefault("unknown_server_catalog", []).extend(data)
        elif isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
            # assume dict of server_id -> list[tools]
            for k, v in data.items():
                if isinstance(v, list):
                    grouped.setdefault(k, []).extend(v)

    if grouped:
        flat = []
        for server_id, items in grouped.items():
            for it in items:
                x = dict(it)
                x["server_id"] = server_id
                flat.append(x)
        return flat

    # 2) Fallback: parse mcp_servers/*/server.py|ts|js
    servers_root = mcp_root / "mcp_servers"
    flat: List[Dict[str, Any]] = []
    if servers_root.exists():
        for server_dir in sorted(servers_root.iterdir()):
            if not server_dir.is_dir():
                continue
            server_id = server_dir.name
            py = server_dir / "server.py"
            ts = server_dir / "server.ts"
            js = server_dir / "server.js"
            items: List[Dict[str, Any]] = []
            if py.exists():
                items = parse_python_server_tools(py)
            elif ts.exists():
                items = parse_tsjs_server_tools(ts)
            elif js.exists():
                items = parse_tsjs_server_tools(js)
            for it in items:
                x = dict(it)
                x["server_id"] = server_id
                flat.append(x)

    return flat

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="Ingest tools & queries from a local clone of Accenture/mcp-bench")
    ap.add_argument("--mcp-bench-path", required=True, help="Path to local mcp-bench clone")
    args = ap.parse_args()

    mcp_root = Path(args.mcp_bench_path).resolve()
    if not mcp_root.exists():
        raise SystemExit(f"Path not found: {mcp_root}")

    ensure_dirs()

    print(f"\n📂 Using mcp-bench at: {mcp_root}")
    print(" - tasks/ exists?      ", (mcp_root / "tasks").exists())
    print(" - mcp_servers/ exists?", (mcp_root / "mcp_servers").exists())

    # 1) Queries
    queries = extract_queries(mcp_root)
    write_json(QUERIES_DIR / "queries.json", {
        "count": len(queries),
        "queries": queries
    })
    print(f"[OK] queries → {QUERIES_DIR / 'queries.json'}")

    # 2) Tool catalog (best effort)
    catalog = extract_tools(mcp_root)
    write_json(TOOLS_DIR / "tool_catalog.json", catalog)
    print(f"[OK] tools   → {TOOLS_DIR / 'tool_catalog.json'}")

    print("\nDone.")

if __name__ == "__main__":
    main()
