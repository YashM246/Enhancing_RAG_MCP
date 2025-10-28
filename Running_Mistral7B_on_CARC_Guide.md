# Running Mistral7B on CARC Discovery Cluster - Complete Guide

Based on the CARC documentation and your execution plan, here's exactly how you should set up and run Mistral7B for both testing and benchmarking:

## Phase 1: Initial Setup & Access

### 1. **Get Access to CARC**
- Login to CARC OnDemand: https://ondemand.carc.usc.edu/
- Must be connected via USC Secure Wireless or VPN
- Use USC NetID credentials + Duo 2FA

### 2. **Access Discovery Cluster Shell**
From OnDemand, request "Discovery Cluster Shell Access" - this gives you terminal access to the cluster.

---

## Phase 2: Environment Setup (One-time)

### 3. **Initialize Conda Environment**

```bash
# Load conda module
module load conda

# Initialize conda for bash (first time only)
conda init bash
source ~/.bashrc
```

### 4. **Create Python Environment for Your Project**

```bash
# Create new environment with Python
conda create --name rag-mcp-env python=3.10

# Activate environment
conda activate rag-mcp-env

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 5. **Install vLLM and Your Project Dependencies**

```bash
# Install vLLM for efficient LLM serving
pip install vllm

# Install your project requirements
# Transfer your requirements.txt to CARC first, then:
pip install sentence-transformers faiss-cpu pandas numpy tqdm python-dotenv requests pytest

# Or if you upload requirements.txt:
pip install -r requirements.txt
```

### 6. **Download Mistral-7B Model**

```bash
# Install huggingface-cli if not already installed
pip install huggingface-hub

# Download the model (will be cached in ~/.cache/huggingface/)
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3
```

**Storage consideration:** The model will be ~14GB. Use your `/project2` directory (5TB quota) for large files if needed.

---

## Phase 3: Running Mistral7B - Two Approaches

### **Approach A: Interactive Testing (For Initial Testing)**

Use this for quick testing and debugging:

```bash
# Request interactive GPU session
salloc --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32GB --time=2:00:00

# Once allocated, activate environment
conda activate rag-mcp-env

# Start vLLM server
vllm serve mistralai/Mistral-7B-Instruct-v0.3 --host 0.0.0.0 --port 8000
```

**When to use:**
- Initial testing
- Debugging your code
- Quick experiments
- Model verification

**Limitations:**
- Session ends when time expires
- Must keep terminal open
- Not suitable for long benchmarking runs

---

### **Approach B: Batch Job Submission (For Actual Benchmarking)**

This is the recommended approach for your actual experiments.

#### **Step 1: Create SLURM Job Script**

Create a file `run_mistral_server.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=mistral-server
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=8:00:00
#SBATCH --output=logs/mistral_server_%j.log
#SBATCH --error=logs/mistral_server_%j.err

# Load conda
module load conda
source ~/.bashrc
conda activate rag-mcp-env

# Create logs directory if it doesn't exist
mkdir -p logs

# Start vLLM server
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096
```

#### **Step 2: Create Experiment Runner Script**

Create `run_rag_experiment.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=rag-mcp-experiment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=10:00:00
#SBATCH --output=logs/experiment_%j.log
#SBATCH --error=logs/experiment_%j.err

# Load conda
module load conda
source ~/.bashrc
conda activate rag-mcp-env

# Navigate to project directory
cd /project2/<your-username>/Enhancing_RAG_MCP/1_Emulating_RAG_MCP

# Start vLLM server in background
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 &

# Wait for server to start
sleep 30

# Get server PID for cleanup
VLLM_PID=$!

# Run your experiments
python src/main.py --mode rag-mcp --k 3 --server-url http://localhost:8000
python src/main.py --mode rag-mcp --k 5 --server-url http://localhost:8000
python src/main.py --mode all-tools --server-url http://localhost:8000
python src/main.py --mode random --server-url http://localhost:8000

# Kill server when done
kill $VLLM_PID
```

#### **Step 3: Submit Jobs**

```bash
# Submit the job
sbatch run_rag_experiment.slurm

# Check job status
squeue -u $USER

# View running jobs
watch -n 5 squeue -u $USER

# Cancel a job if needed
scancel <job-id>
```

---

## Phase 4: Project File Organization on CARC

### **Recommended Directory Structure:**

```bash
/project2/<your-username>/
└── Enhancing_RAG_MCP/
    └── 1_Emulating_RAG_MCP/
        ├── src/
        │   ├── indexing/
        │   ├── llm/
        │   ├── retrieval/
        │   └── main.py
        ├── data/
        │   ├── tools/
        │   └── queries/
        ├── results/
        ├── logs/
        ├── slurm_scripts/
        │   ├── run_mistral_server.slurm
        │   └── run_rag_experiment.slurm
        ├── requirements.txt
        └── README.md
```

### **Transfer Files to CARC:**

```bash
# From your local machine (Windows), use SCP or OnDemand file browser

# Option 1: Using OnDemand web interface
# Go to Files → Home Directory → Upload

# Option 2: Using SCP from Windows (Git Bash or PowerShell)
scp -r "C:\Users\gr8my\Desktop\Projects\Enhancing_RAG_MCP\1_Emulating_RAG_MCP" \
    <your-usc-id>@discovery.usc.edu:/project2/<your-usc-id>/
```

---

## Phase 5: Optimization Tips from CARC Documentation

### **1. Data Loading Optimization**
The CARC workshop emphasized that **data loading is often the bottleneck**, not model computation.

```python
# In your data loaders, use multiple workers
# Match this with --cpus-per-task in SLURM script

from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,  # Match --cpus-per-task=8
    pin_memory=True
)
```

### **2. GPU Monitoring**
```bash
# Monitor GPU usage during job
watch -n 1 nvidia-smi
```

### **3. Profile Your Code First**
Before running full benchmarks:

```bash
# Install profiler
conda install line_profiler --channel conda-forge

# Profile your code
kernprof -o ${SLURM_JOBID}.lprof -l src/main.py --mode rag-mcp --k 3
python -m line_profiler -rmt *.lprof
```

---

## Phase 6: Testing & Benchmarking Workflow

### **Step 1: Quick Test (Interactive)**
```bash
# Get interactive session
salloc --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32GB --time=1:00:00

# Activate environment
conda activate rag-mcp-env

# Test vLLM server
vllm serve mistralai/Mistral-7B-Instruct-v0.3 --host 0.0.0.0 --port 8000

# In another terminal, test connection
curl http://localhost:8000/v1/models
```

### **Step 2: Test Your Python Code**
```bash
# Test with small dataset (10 queries)
python src/main.py --mode rag-mcp --k 3 --server-url http://localhost:8000 --limit 10
```

### **Step 3: Run Full Benchmarking (Batch Job)**
```bash
# Submit full experiment
sbatch slurm_scripts/run_rag_experiment.slurm

# Monitor progress
tail -f logs/experiment_<job-id>.log
```

---

## Key SLURM Parameters Explained

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `--partition=gpu` | gpu | Use GPU partition (required for GPU access) |
| `--gres=gpu:1` | 1 GPU | Request 1 GPU (Mistral-7B fits on 1 GPU) |
| `--cpus-per-task=8` | 8 cores | For data loading parallelization |
| `--mem=32GB` | 32GB RAM | Enough for model + data (increase if needed) |
| `--time=8:00:00` | 8 hours | Maximum runtime (adjust based on experiment) |

---

## Monitoring & Debugging

### **Check Job Status:**
```bash
# View your jobs
squeue -u $USER

# View detailed job info
scontrol show job <job-id>

# View job history
sacct -j <job-id> --format=JobID,JobName,State,ExitCode,Start,End
```

### **View Logs:**
```bash
# Real-time log monitoring
tail -f logs/experiment_<job-id>.log

# Search for errors
grep -i error logs/experiment_<job-id>.err
```

### **Test Server Connectivity:**
```bash
# From within SLURM job or interactive session
curl http://localhost:8000/v1/models

# Test inference
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

---

## Recommended Workflow for Your Project

### **Week 1: Setup & Initial Testing**
1. Set up CARC account and access
2. Create conda environment
3. Install all dependencies (vLLM, PyTorch, your requirements)
4. Download Mistral-7B model
5. Test interactive session with vLLM
6. Transfer your code to CARC
7. Test your Python code with small dataset (10 queries)

### **Week 2-3: Benchmarking**
1. Create SLURM batch scripts
2. Run RAG-MCP experiments (k=3, k=5)
3. Run baseline experiments (all-tools, random)
4. Run ablation studies
5. Monitor GPU utilization and optimize

### **Week 4: Analysis**
1. Download results from CARC
2. Perform analysis locally or on CARC
3. Generate visualizations

---

## Storage Locations on CARC

| Location | Quota | Use For |
|----------|-------|---------|
| `/home1/<username>` | 100 GB | Code, configs, scripts |
| `/project2/<group>` | 5 TB+ | Models, datasets, results |
| `/scratch1/<username>` | 10 TB | Temporary files, I/O-intensive work |

**Recommendation:**
- Store code in `/home1`
- Store model cache and results in `/project2`
- Use `/scratch1` for intermediate processing

---

## Cost Optimization Tips

1. **Start with short time limits** - Test with `--time=1:00:00` first
2. **Optimize before scaling** - Profile and optimize single-GPU performance before requesting more resources
3. **Use interactive sessions sparingly** - They keep resources allocated even when idle
4. **Clean up old files** - Delete old logs and temporary files
5. **Monitor queue times** - Submit jobs during off-peak hours if possible

---

## Troubleshooting Common Issues

### **Issue 1: Out of Memory**
```bash
# Increase memory allocation
#SBATCH --mem=64GB

# Or reduce model memory usage
vllm serve model --gpu-memory-utilization 0.8
```

### **Issue 2: Server Won't Start**
```bash
# Check if port is in use
netstat -tuln | grep 8000

# Use different port
vllm serve model --port 8001
```

### **Issue 3: Job Timeout**
```bash
# Increase time limit
#SBATCH --time=24:00:00

# Or optimize your code to run faster
```

### **Issue 4: Can't Connect to Server**
```bash
# Make sure server is running
ps aux | grep vllm

# Check logs
cat logs/mistral_server_<job-id>.log
```

---

## Summary: Quick Start Commands

```bash
# 1. First time setup
module load conda
conda create --name rag-mcp-env python=3.10
conda activate rag-mcp-env
pip install vllm torch sentence-transformers faiss-cpu
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3

# 2. Interactive testing
salloc --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32GB --time=2:00:00
conda activate rag-mcp-env
vllm serve mistralai/Mistral-7B-Instruct-v0.3 --host 0.0.0.0 --port 8000

# 3. Batch job submission
sbatch run_rag_experiment.slurm
squeue -u $USER
tail -f logs/experiment_*.log
```

---

## Key Differences: Testing vs Benchmarking

**Testing:** Use interactive sessions (`salloc`) for quick feedback and debugging

**Benchmarking:** Use batch jobs (`sbatch`) for long-running, reproducible experiments

---

## Additional Resources

- **CARC Documentation**: https://www.carc.usc.edu/user-guides/
- **CARC Quick Start**: https://www.carc.usc.edu/user-guides/quick-start-guides/intro-to-carc
- **Running DL Applications**: https://github.com/uschpc/Running-DL-Applications
- **vLLM Documentation**: https://docs.vllm.ai/
- **SLURM Cheatsheet**: Available on CARC website
