#!/usr/bin/env python3
"""
Create Server Subsets for Benchmarking

This script creates progressively nested server subsets from the full tools and queries datasets.
Subsets are designed to analyze how RAG-MCP performance scales with toolset size.

Strategy:
- Nested: Smaller subsets are contained in larger ones
- Progressive by size: Start with smallest servers, add larger ones incrementally
- Filter distraction_servers to maintain data integrity
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


# Define nested server subsets (progressive by size)
# Each subset contains all servers from previous subset + new servers
SERVER_SUBSETS = {
    5: [
        # Subset 1: 5 smallest servers (tiny: 1-4 tools)
        "OpenAPI Explorer",      # 1 tool
        "Time MCP",              # 2 tools
        "Car Price Evaluator",   # 3 tools
        "Weather Data",          # 4 tools
        "Bibliomantic",          # 6 tools (small)
    ],
    10: [
        # Previous 5 + 5 more (complete tiny, add small)
        "OpenAPI Explorer", "Time MCP", "Car Price Evaluator", "Weather Data", "Bibliomantic",
        "National Parks",        # 6 tools (small)
        "Google Maps",           # 7 tools (small)
        "Game Trends",           # 7 tools (small)
        "Wikipedia",             # 9 tools (small)
        "Huge Icons",            # 10 tools (medium)
    ],
    15: [
        # Previous 10 + 5 more (add more medium)
        "OpenAPI Explorer", "Time MCP", "Car Price Evaluator", "Weather Data", "Bibliomantic",
        "National Parks", "Google Maps", "Game Trends", "Wikipedia", "Huge Icons",
        "Hugging Face",          # 10 tools (medium)
        "DEX Paprika",           # 11 tools (medium)
        "Unit Converter",        # 16 tools (medium)
        "NixOS",                 # 17 tools (medium)
        "Paper Search",          # 19 tools (medium)
    ],
    20: [
        # Previous 15 + 5 more (add large servers + remaining tiny)
        "OpenAPI Explorer", "Time MCP", "Car Price Evaluator", "Weather Data", "Bibliomantic",
        "National Parks", "Google Maps", "Game Trends", "Wikipedia", "Huge Icons",
        "Hugging Face", "DEX Paprika", "Unit Converter", "NixOS", "Paper Search",
        "Reddit",                # 2 tools (tiny)
        "OKX Exchange",          # 2 tools (tiny)
        "NASA Data",             # 20 tools (large)
        "Scientific Computing",  # 27 tools (large)
        "BioMCP",                # 36 tools (large)
    ],
    27: [
        # All servers
        "OpenAPI Explorer", "Time MCP", "Car Price Evaluator", "Weather Data", "Bibliomantic",
        "National Parks", "Google Maps", "Game Trends", "Wikipedia", "Huge Icons",
        "Hugging Face", "DEX Paprika", "Unit Converter", "NixOS", "Paper Search",
        "Reddit", "OKX Exchange", "NASA Data", "Scientific Computing", "BioMCP",
        "Medical Calculator",    # 51 tools (large)
        "Call for Papers",       # 1 tool (tiny)
        "Context7",              # 1 tool (tiny)
        "FruityVice",            # 1 tool (tiny)
        "Movie Recommender",     # 1 tool (tiny)
        "Metropolitan Museum",   # 3 tools (tiny)
        "OSINT Intelligence",    # 7 tools (small)
    ]
}


def load_json(file_path: str) -> Any:
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: str) -> None:
    """Save data to JSON file with pretty formatting"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Saved: {file_path}")


def filter_tools(all_tools: List[Dict], selected_servers: List[str]) -> List[Dict]:
    """
    Filter tools belonging to selected servers

    Args:
        all_tools: List of all tool dictionaries
        selected_servers: List of server names to include

    Returns:
        List of filtered tool dictionaries
    """
    filtered = [tool for tool in all_tools if tool['server'] in selected_servers]
    return filtered


def filter_queries(all_queries: Dict, selected_servers: List[str]) -> Dict:
    """
    Filter queries and adjust distraction_servers based on selected servers

    Strategy:
    - Include query only if query['server_name'] in selected_servers
    - Filter distraction_servers to only include servers in selected_servers

    Args:
        all_queries: Complete queries dictionary with server_tasks structure
        selected_servers: List of server names to include

    Returns:
        Filtered queries dictionary maintaining original structure
    """
    selected_set = set(selected_servers)
    filtered_queries = {
        "generation_info": all_queries.get("generation_info", {}),
        "server_tasks": []
    }

    for server_task in all_queries.get("server_tasks", []):
        server_name = server_task.get("server_name", "")

        # Only include if server is in selected set
        if server_name not in selected_set:
            continue

        # Filter tasks and adjust distraction_servers
        filtered_tasks = []
        for task in server_task.get("tasks", []):
            # Filter distraction_servers to only include servers in subset
            # Exclude the task's own server from distractions
            original_distractions = task.get("distraction_servers", [])
            filtered_distractions = [
                d for d in original_distractions
                if d in selected_set and d != server_name
            ]

            # Create filtered task
            filtered_task = task.copy()
            filtered_task["distraction_servers"] = filtered_distractions
            filtered_tasks.append(filtered_task)

        # Only include server_task if it has tasks
        if filtered_tasks:
            filtered_server_task = {
                "server_name": server_name,
                "tasks": filtered_tasks,
                "servers": [server_name],  # Keep original structure
                "combination_name": server_task.get("combination_name", f"Single Server: {server_name}"),
                "combination_type": server_task.get("combination_type", "single_server")
            }
            filtered_queries["server_tasks"].append(filtered_server_task)

    return filtered_queries


def validate_subset(tools: List[Dict], queries: Dict, selected_servers: List[str]) -> bool:
    """
    Validate filtered subset for data integrity

    Returns:
        True if valid, False otherwise
    """
    errors = []

    # Check that all tools belong to selected servers
    tool_servers = set(tool['server'] for tool in tools)
    selected_set = set(selected_servers)
    if not tool_servers.issubset(selected_set):
        extra_servers = tool_servers - selected_set
        errors.append(f"Tools contain servers not in selected list: {extra_servers}")

    # Check that all query servers are in selected list
    query_servers = set(st['server_name'] for st in queries['server_tasks'])
    if not query_servers.issubset(selected_set):
        extra_servers = query_servers - selected_set
        errors.append(f"Queries contain servers not in selected list: {extra_servers}")

    # Check that distraction_servers don't reference non-existent servers
    for server_task in queries['server_tasks']:
        for task in server_task['tasks']:
            distraction_servers = set(task.get('distraction_servers', []))
            if not distraction_servers.issubset(selected_set):
                extra = distraction_servers - selected_set
                errors.append(f"Task {task.get('task_id')} has invalid distractions: {extra}")
                break  # Only report once per subset
        if errors:
            break

    if errors:
        for error in errors:
            print(f"  [ERROR] Validation error: {error}")
        return False

    return True


def analyze_subset(tools: List[Dict], queries: Dict, selected_servers: List[str]) -> Dict:
    """
    Analyze subset statistics

    Returns:
        Dictionary with subset statistics
    """
    # Count tools per server
    tools_per_server = defaultdict(int)
    for tool in tools:
        tools_per_server[tool['server']] += 1

    # Count queries per server
    queries_per_server = defaultdict(int)
    for server_task in queries['server_tasks']:
        server_name = server_task['server_name']
        queries_per_server[server_name] = len(server_task['tasks'])

    # Total queries
    total_queries = sum(len(st['tasks']) for st in queries['server_tasks'])

    return {
        "num_servers": len(selected_servers),
        "num_tools": len(tools),
        "num_queries": total_queries,
        "tools_per_server": dict(tools_per_server),
        "queries_per_server": dict(queries_per_server),
        "servers_with_no_queries": [s for s in selected_servers if s not in queries_per_server]
    }


def create_subset(
    all_tools: List[Dict],
    all_queries: Dict,
    subset_size: int,
    selected_servers: List[str],
    output_dir: str
) -> None:
    """
    Create a single subset with filtered tools and queries

    Args:
        all_tools: Complete list of tools
        all_queries: Complete queries dictionary
        subset_size: Number of servers in subset (for naming)
        selected_servers: List of server names to include
        output_dir: Base output directory
    """
    print(f"\n{'='*80}")
    print(f"Creating Subset: {subset_size} servers")
    print(f"{'='*80}")

    # Filter data
    print("Filtering tools...")
    filtered_tools = filter_tools(all_tools, selected_servers)

    print("Filtering queries...")
    filtered_queries = filter_queries(all_queries, selected_servers)

    # Analyze
    print("Analyzing subset...")
    stats = analyze_subset(filtered_tools, filtered_queries, selected_servers)

    # Validate
    print("Validating subset...")
    if not validate_subset(filtered_tools, filtered_queries, selected_servers):
        print("  [ERROR] Validation failed! Skipping this subset.")
        return
    print("  [OK] Validation passed")

    # Create filenames with tool and query counts
    tools_filename = f"tools_{subset_size}servers_{stats['num_tools']}tools.json"
    queries_filename = f"queries_{subset_size}servers_{stats['num_queries']}queries.json"

    # Save files
    tools_path = os.path.join(output_dir, "tools", "subsets", tools_filename)
    queries_path = os.path.join(output_dir, "queries", "subsets", queries_filename)

    print(f"\nSaving subset files...")
    save_json(filtered_tools, tools_path)
    save_json(filtered_queries, queries_path)

    # Print statistics
    print(f"\nSubset Statistics:")
    print(f"  * Servers: {stats['num_servers']}")
    print(f"  * Tools: {stats['num_tools']}")
    print(f"  * Queries: {stats['num_queries']}")
    print(f"  * Avg tools/server: {stats['num_tools'] / stats['num_servers']:.1f}")
    print(f"  * Avg queries/server: {stats['num_queries'] / stats['num_servers']:.1f}")

    if stats['servers_with_no_queries']:
        print(f"  [WARNING] Servers with no queries: {', '.join(stats['servers_with_no_queries'])}")

    print(f"\n  [OK] Subset {subset_size} created successfully!")


def verify_nested_property(subsets: Dict[int, List[str]]) -> bool:
    """
    Verify that subsets are properly nested (smaller ⊂ larger)

    Returns:
        True if nested property holds, False otherwise
    """
    print(f"\n{'='*80}")
    print("Verifying Nested Property")
    print(f"{'='*80}")

    sizes = sorted(subsets.keys())

    for i in range(len(sizes) - 1):
        smaller_size = sizes[i]
        larger_size = sizes[i + 1]

        smaller_set = set(subsets[smaller_size])
        larger_set = set(subsets[larger_size])

        if not smaller_set.issubset(larger_set):
            missing = smaller_set - larger_set
            print(f"  [ERROR] Subset {smaller_size} is NOT contained in subset {larger_size}")
            print(f"    Missing servers: {missing}")
            return False
        else:
            print(f"  [OK] Subset {smaller_size} is subset of Subset {larger_size}")

    print(f"\n  [OK] All subsets are properly nested!")
    return True


def main():
    """Main execution"""
    print("="*80)
    print("Server Subset Creation Tool")
    print("="*80)

    # Paths
    base_dir = Path(__file__).parent.parent
    tools_path = base_dir / "data" / "tools" / "tools_corrected.json"
    queries_path = base_dir / "data" / "queries" / "mcp_task_description.json"
    output_dir = str(base_dir / "data")

    # Load data
    print(f"\nLoading source data...")
    print(f"  Tools: {tools_path}")
    print(f"  Queries: {queries_path}")

    all_tools = load_json(str(tools_path))
    all_queries = load_json(str(queries_path))

    print(f"  [OK] Loaded {len(all_tools)} tools")
    print(f"  [OK] Loaded {len(all_queries.get('server_tasks', []))} server tasks")

    # Verify nested property
    if not verify_nested_property(SERVER_SUBSETS):
        print("\n[ERROR] Subsets are not properly nested. Fix SERVER_SUBSETS definition.")
        return

    # Create each subset
    for subset_size in sorted(SERVER_SUBSETS.keys()):
        selected_servers = SERVER_SUBSETS[subset_size]
        create_subset(all_tools, all_queries, subset_size, selected_servers, output_dir)

    # Final summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"[OK] Created {len(SERVER_SUBSETS)} server subsets")
    print(f"[OK] Generated {len(SERVER_SUBSETS) * 2} files (5 tool files + 5 query files)")
    print(f"\nOutput locations:")
    print(f"  * Tools: data/tools/subsets/")
    print(f"  * Queries: data/queries/subsets/")
    print(f"\nNext steps:")
    print(f"  1. Verify file contents")
    print(f"  2. Modify benchmarker.py to accept custom paths")
    print(f"  3. Create SLURM script for subset benchmarking")
    print(f"  4. Run benchmarks on CARC")


if __name__ == "__main__":
    main()
