#!/bin/bash
#SBATCH --job-name=rag-mcp-benchmark
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

# ============================================================================
# RAG-MCP Benchmarking Job
# ============================================================================
# This script runs benchmarks for all 7 tool selection approaches:
# 1. Dense Retrieval Only
# 2. BM25 Only
# 3. BM25 + Dense Hybrid (No LLM)
# 4. LLM Only (Full Context)
# 5. Dense + LLM
# 6. BM25 + LLM
# 7. Hybrid + LLM
# ============================================================================

echo "=========================================="
echo "Job ID: $SLURM_JOBID"
echo "Job Name: $SLURM_JOB_NAME"
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

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p data/results

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
echo "Start time: $(date)"

# Start vLLM server in background
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --dtype float16 > logs/vllm_server_${SLURM_JOBID}.log 2>&1 &

# Get vLLM server PID for cleanup later
VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

# Wait for server to start (120 seconds should be enough)
echo "Waiting for vLLM server to initialize..."
sleep 120

# Check if server is running
if ! ps -p $VLLM_PID > /dev/null; then
    echo "ERROR: vLLM server failed to start!"
    cat logs/vllm_server_${SLURM_JOBID}.log
    exit 1
fi

# Test server connectivity
echo "Testing vLLM server connectivity..."
curl -s http://localhost:8000/v1/models || echo "Warning: Server might not be ready yet"
echo "=========================================="

# ============================================================================
# Run Benchmarking
# ============================================================================
echo "Starting benchmarking..."
echo "Benchmark start time: $(date)"
echo "=========================================="

# Run the unified benchmarker
python benchmarking/benchmarker.py \
    --server-url http://localhost:8000 \
    --output-dir data/results \
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
echo "Results saved to: data/results/"
ls -lh data/results/

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="

exit $BENCHMARK_EXIT_CODE
