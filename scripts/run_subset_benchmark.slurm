#!/bin/bash
#SBATCH --job-name=rag-subset
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=6:00:00
#SBATCH --output=logs/subset_benchmark_%j.out
#SBATCH --error=logs/subset_benchmark_%j.err

# ============================================================================
# RAG-MCP Server Subset Benchmarking Job
# ============================================================================
# This script runs benchmarks on progressively nested server subsets
# to analyze how performance scales with toolset size.
#
# Usage: sbatch scripts/run_subset_benchmark.slurm <subset_size>
# Example: sbatch scripts/run_subset_benchmark.slurm 5
#
# Subset sizes: 5, 10, 15, 20, 27 (servers)
# ============================================================================

# ============================================================================
# Parse Subset Parameter
# ============================================================================
SUBSET_SIZE=${1:-5}  # Default to 5 if no argument provided

# Validate subset size
if [[ ! "$SUBSET_SIZE" =~ ^(5|10|15|20|27)$ ]]; then
    echo "ERROR: Invalid subset size '$SUBSET_SIZE'"
    echo "Valid sizes: 5, 10, 15, 20, 27"
    exit 1
fi

# ============================================================================
# Configure Paths and Time Limits Based on Subset Size
# ============================================================================
case $SUBSET_SIZE in
    5)
        TOOLS_PATH="data/tools/subsets/tools_5servers_16tools.json"
        QUERIES_PATH="data/queries/subsets/queries_5servers_10queries.json"
        TIME_LIMIT="1:00:00"
        NUM_TOOLS=16
        NUM_QUERIES=10
        ;;
    10)
        TOOLS_PATH="data/tools/subsets/tools_10servers_55tools.json"
        QUERIES_PATH="data/queries/subsets/queries_10servers_20queries.json"
        TIME_LIMIT="2:00:00"
        NUM_TOOLS=55
        NUM_QUERIES=20
        ;;
    15)
        TOOLS_PATH="data/tools/subsets/tools_15servers_128tools.json"
        QUERIES_PATH="data/queries/subsets/queries_15servers_30queries.json"
        TIME_LIMIT="3:00:00"
        NUM_TOOLS=128
        NUM_QUERIES=30
        ;;
    20)
        TOOLS_PATH="data/tools/subsets/tools_20servers_215tools.json"
        QUERIES_PATH="data/queries/subsets/queries_20servers_40queries.json"
        TIME_LIMIT="5:00:00"
        NUM_TOOLS=215
        NUM_QUERIES=40
        ;;
    27)
        TOOLS_PATH="data/tools/subsets/tools_27servers_280tools.json"
        QUERIES_PATH="data/queries/subsets/queries_27servers_54queries.json"
        TIME_LIMIT="6:00:00"
        NUM_TOOLS=280
        NUM_QUERIES=54
        ;;
esac

OUTPUT_DIR="data/results/subset_${SUBSET_SIZE}servers"

echo "=========================================="
echo "Server Subset Benchmark"
echo "=========================================="
echo "Job ID: $SLURM_JOBID"
echo "Subset Size: $SUBSET_SIZE servers"
echo "Tools: $NUM_TOOLS"
echo "Queries: $NUM_QUERIES"
echo "Time Limit: $TIME_LIMIT"
echo "Node: $SLURM_NODELIST"
echo "GPU: $SLURM_GPUS"
echo "Start Time: $(date)"
echo "=========================================="

# ============================================================================
# Environment Setup
# ============================================================================
echo "Setting up environment..."

# Load conda
module purge
module load conda

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate rag_mcp_env

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "=========================================="

# ============================================================================
# Navigate to Project Directory
# ============================================================================
# IMPORTANT: Update this path with your actual project path
# Format: /project2/<PI_username>_<Project_ID>/team35/Enhancing_RAG_MCP
PROJECT_DIR="/project2/jieyuz_1727/team35/Enhancing_RAG_MCP"

cd "$PROJECT_DIR" || exit 1
echo "Working directory: $(pwd)"
echo "=========================================="

# Create necessary directories
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# Verify subset files exist
if [ ! -f "$TOOLS_PATH" ]; then
    echo "ERROR: Tools file not found: $TOOLS_PATH"
    exit 1
fi

if [ ! -f "$QUERIES_PATH" ]; then
    echo "ERROR: Queries file not found: $QUERIES_PATH"
    exit 1
fi

echo "Subset files verified:"
echo "  Tools: $TOOLS_PATH"
echo "  Queries: $QUERIES_PATH"
echo "  Output: $OUTPUT_DIR"
echo "=========================================="

# ============================================================================
# GPU Information
# ============================================================================
echo "GPU Information:"
nvidia-smi
echo "=========================================="

# ============================================================================
# Start vLLM Server
# ============================================================================
echo "Starting vLLM server..."
echo "Model: mistralai/Mistral-7B-Instruct-v0.3"
echo "Max context length: 16384 tokens"
echo "Start time: $(date)"

# Start vLLM server in background
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 16384 \
    --dtype float16 > logs/vllm_subset${SUBSET_SIZE}_${SLURM_JOBID}.log 2>&1 &

# Get vLLM server PID for cleanup later
VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

# Wait for server to start (with health check)
echo "Waiting for vLLM server to initialize..."
MAX_WAIT=300  # 5 minutes max wait
WAIT_INTERVAL=10
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    sleep $WAIT_INTERVAL
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))

    # Check if process is still running
    if ! ps -p $VLLM_PID > /dev/null; then
        echo "ERROR: vLLM server process died!"
        echo "Last 50 lines of vLLM log:"
        tail -50 logs/vllm_subset${SUBSET_SIZE}_${SLURM_JOBID}.log
        exit 1
    fi

    # Try to connect to server
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "[OK] vLLM server is ready! (waited ${ELAPSED}s)"
        curl -s http://localhost:8000/v1/models
        break
    else
        echo "  Still waiting... (${ELAPSED}s elapsed)"
    fi
done

# Final check
if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server did not start within ${MAX_WAIT}s"
    echo "Last 50 lines of vLLM log:"
    tail -50 logs/vllm_subset${SUBSET_SIZE}_${SLURM_JOBID}.log
    exit 1
fi

echo "=========================================="

# ============================================================================
# Run Benchmarking
# ============================================================================
echo "Starting benchmarking for subset ${SUBSET_SIZE}..."
echo "Benchmark start time: $(date)"
echo "=========================================="

# Run the benchmarker with custom paths
python benchmarking/benchmarker.py \
    --tools-path "$TOOLS_PATH" \
    --queries-path "$QUERIES_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --server-url http://localhost:8000 \
    --limit-queries -1 \
    --k-values 3 5 7

BENCHMARK_EXIT_CODE=$?

echo "=========================================="
echo "Benchmarking completed with exit code: $BENCHMARK_EXIT_CODE"
echo "Benchmark end time: $(date)"
echo "=========================================="

# ============================================================================
# Cleanup
# ============================================================================
echo "Cleaning up..."

# Kill vLLM server
if ps -p $VLLM_PID > /dev/null; then
    echo "Stopping vLLM server (PID: $VLLM_PID)..."
    kill $VLLM_PID
    sleep 5
    # Force kill if still running
    if ps -p $VLLM_PID > /dev/null; then
        kill -9 $VLLM_PID
    fi
    echo "vLLM server stopped"
else
    echo "vLLM server already stopped"
fi

echo "=========================================="

# ============================================================================
# Results Summary
# ============================================================================
echo "Results Summary:"
echo "  Subset: $SUBSET_SIZE servers ($NUM_TOOLS tools, $NUM_QUERIES queries)"
echo "  Output directory: $OUTPUT_DIR"
echo "  Expected files: 13 (7 approaches with k variations)"
echo ""
echo "Result files:"
ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null || echo "  (No results generated)"

echo ""
echo "Result count: $(ls "$OUTPUT_DIR"/*.json 2>/dev/null | wc -l) / 13"

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="

exit $BENCHMARK_EXIT_CODE
