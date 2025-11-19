#!/bin/bash
#SBATCH --job-name=rag-mcp-test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

# ============================================================================
# RAG-MCP Test Benchmarking Job (5 queries only)
# ============================================================================
# Quick test run to verify everything works before full benchmark
# ============================================================================

echo "=========================================="
echo "TEST RUN - Limited to 5 queries per approach"
echo "Job ID: $SLURM_JOBID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# ============================================================================
# Environment Setup
# ============================================================================
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate rag_mcp_env

# ============================================================================
# Navigate to Project Directory
# ============================================================================
PROJECT_DIR="/project2/jieyuz_1727/team35/Enhancing_RAG_MCP"
cd "$PROJECT_DIR" || exit 1

mkdir -p logs
mkdir -p data/results

# ============================================================================
# Start vLLM Server
# ============================================================================
echo "Starting vLLM server..."
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --dtype float16 > logs/vllm_test_${SLURM_JOBID}.log 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"
sleep 120

# ============================================================================
# Run Test Benchmark (5 queries only)
# ============================================================================
echo "Running test benchmark (5 queries)..."

python benchmarking/benchmarker.py \
    --server-url http://localhost:8000 \
    --output-dir data/results \
    --limit-queries 5 \
    --k-values 3

BENCHMARK_EXIT_CODE=$?

# ============================================================================
# Cleanup
# ============================================================================
if ps -p $VLLM_PID > /dev/null; then
    kill $VLLM_PID
    sleep 5
    kill -9 $VLLM_PID 2>/dev/null
fi

echo "=========================================="
echo "Test completed at: $(date)"
echo "Exit code: $BENCHMARK_EXIT_CODE"
echo "=========================================="

exit $BENCHMARK_EXIT_CODE