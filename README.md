# RAG-MCP: Scalable Tool Selection for Large Language Models

A research implementation exploring Retrieval-Augmented Generation (RAG) approaches for efficient external tool selection in Large Language Models, with planned extensions using hybrid search techniques.

## 🎯 Problem Statement

As LLMs integrate with growing toolsets through protocols like Model Context Protocol (MCP), **prompt bloat** becomes a critical issue:

- Including all tool descriptions in prompts overwhelms the LLM's context window
- Tool selection accuracy drops dramatically (from ~90% → 13.62% as tools scale)
- Token costs and latency increase proportionally
- Decision complexity causes model confusion and hallucinations

**Example:** An LLM with access to 1,000 tools cannot efficiently determine which tool to use for "Find recent papers about climate change" when all 1,000 tool descriptions are in the prompt.

## 💡 Solution Overview

This project implements and compares **6 different approaches** to tool selection for LLMs:

**Pure Retrieval Methods (No LLM):**
1. **Dense Retrieval Only (top-1)** - Cosine similarity on embeddings, select top-1 tool
2. **BM25 Only (top-1)** - Lexical search, select top-1 tool

**LLM-Based Methods:**
3. **LLM Only (Full Context)** - All tools provided to LLM (naive MCP baseline)
4. **Dense Retrieval + LLM (top-k)** - RAG-MCP: Embedding-based retrieval → LLM selects from top-k
5. **BM25 + LLM (top-k)** - BM25 retrieval → LLM selects from top-k
6. **Hybrid Retrieval + LLM (top-k)** - Combined dense + BM25 retrieval → LLM selects from top-k

**Key Benefits:**
- Systematic comparison from pure retrieval to hybrid approaches
- Demonstrates trade-offs between speed, accuracy, and context efficiency
- Validates RAG-MCP methodology and explores improvements

## 📊 Expected Results

Based on Gan & Sun (2025) and our experimental design:

| Approach | Accuracy (Expected) | Token Usage | Latency | k Values Tested | Notes |
|----------|---------------------|-------------|---------|----------------|-------|
| 1. Dense Retrieval Only | Low (~20-30%) | Minimal (0 LLM tokens) | Fastest | Retrieve top-7, select top-1 | Reports Recall@1/3/5/7 |
| 2. BM25 Only | Low (~15-25%) | Minimal (0 LLM tokens) | Fastest | Retrieve top-7, select top-1 | Reports Recall@1/3/5/7 |
| 3. LLM Only (Full Context) | ~13% | Highest (100% baseline) | Slowest | All tools | Prompt bloat baseline |
| 4. Dense Retrieval + LLM | ~43% | ~50% reduction | Fast | k = 3, 5, 7 | RAG-MCP from paper |
| 5. BM25 + LLM | ~35-40% | ~50% reduction | Fast | k = 3, 5, 7 | Lexical filtering |
| 6. Hybrid Retrieval + LLM | **>50%** (goal) | ~50% reduction | Fast | k = 3, 5, 7 | Best of both worlds |

**Key Hypothesis:** Hybrid approach (6) should outperform both pure retrieval and single-retrieval methods by combining semantic understanding with keyword matching.

## 🏗️ Architecture

```
                    User Query
                        |
        +---------------+---------------+
        |               |               |
   Dense Retrieval   BM25 Search   Hybrid Fusion
   (Embeddings)      (Lexical)    (Both Combined)
        |               |               |
        +-------+-------+-------+-------+
                |               |
         Direct Selection   LLM Selection
         (top-1 only)      (top-k reasoning)
                |               |
                v               v
           Tool Selection   Tool Selection
```

**Components:**
1. **Tool Indexer:**
   - Dense: Embeds tool descriptions into vector space (FAISS)
   - Sparse: BM25 index for keyword matching
2. **Retriever:** Multiple strategies (dense, BM25, hybrid)
3. **LLM Selector:** Optional reasoning layer for top-k candidates
4. **Evaluator:** Measures accuracy, token usage, and latency across all 6 approaches

## 📈 Evaluation Metrics

**Comparison Level:**
- Server-level comparison (not individual tool level)
- Evaluates if the approach selects tools from the correct server

**Accuracy Metrics:**
- **Accuracy**: Top-1 server selection correctness (%)
- **Recall@k**: Is correct server in top-k candidates? (k = 1, 3, 5, 7)
- **Mean Reciprocal Rank (MRR)**: Average rank of correct server (0-1, higher is better)

**Efficiency Metrics (LLM approaches only):**
- Average Prompt Tokens
- Average Completion Tokens
- Total Token Usage
- Token Reduction vs Baseline (%)
- Cost per Query ($)

**Latency Metrics:**
- Total Query Latency (seconds)
- Retrieval Latency (approaches 1, 2, 4, 5, 6)
- LLM Inference Latency (approaches 3, 4, 5, 6)

## 📚 Research Background

This project is based on:

**Primary Paper:**
> Gan, T., & Sun, Q. (2025). RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation. *arXiv preprint arXiv:2505.03275*.

**Supporting Work:**
- Luo et al. (2025) - MCPBench evaluation framework
- Lewis et al. (2020) - RAG foundations
- Gao et al. (2023) - Hybrid retrieval survey

**Key Insights:**
1. Tool selection degrades significantly as toolsets scale (13.62% accuracy at 11,100 tools)
2. Semantic retrieval restores accuracy to ~43% while reducing tokens by >50%
3. Hybrid approaches may further improve by combining semantic + keyword matching

## 📁 Project Structure

```
Enhancing_RAG_MCP/
├── src/
│   ├── indexing/              # Index building components
│   │   ├── tool_indexer.py    # Dense embeddings (FAISS)
│   │   └── bm25_indexer.py    # Sparse BM25 index
│   ├── retrieval/             # Retrieval components
│   │   ├── dense_retriever.py # Dense/semantic retrieval
│   │   ├── bm25_retriever.py  # Sparse/lexical retrieval
│   │   └── hybrid_retriever.py # Hybrid fusion (RRF)
│   ├── approaches/            # Core implementations of 6 approaches
│   │   ├── dense_only.py      # Approach 1: Dense Retrieval Only
│   │   ├── bm25_only.py       # Approach 2: BM25 Only
│   │   ├── llm_only.py        # Approach 3: LLM Only (Full Context)
│   │   ├── dense_llm.py       # Approach 4: Dense + LLM
│   │   ├── bm25_llm.py        # Approach 5: BM25 + LLM
│   │   └── hybrid_llm.py      # Approach 6: Hybrid + LLM
│   └── llm/                   # LLM integration
│       └── llm_selector.py    # LLM tool selection logic
├── benchmarking/              # Evaluation framework
│   └── benchmarker.py         # Unified benchmarking suite
├── data/
│   ├── tools/                 # Tool definitions (JSON)
│   ├── queries/               # Test queries with ground truth
│   ├── indexes/               # Pre-built FAISS and BM25 indexes
│   └── results/               # Experiment results
└── tests/                     # Unit and integration tests
```

## 🛠️ Technology Stack

**Core Libraries:**
- `sentence-transformers` - Dense embeddings (semantic search)
- `faiss-cpu` - Vector similarity search
- `rank-bm25` - Sparse lexical search (BM25)
- `vllm` or `ollama` - Open-source LLM serving
- `pandas`, `numpy` - Data processing

**Retrieval Components:**
- **Dense:** `all-MiniLM-L6-v2` (fast baseline) or `all-mpnet-base-v2` (higher quality)
- **Sparse:** BM25 with custom tokenization
- **Hybrid:** Reciprocal Rank Fusion (RRF) or weighted combination

**LLMs (Self-Hosted):**
- Primary: Mistral 7B Instruct / Mixtral 8x7B Instruct
- Alternative: Qwen2.5-7B-Instruct / LLaMA 3.1-8B-Instruct
- Deployment: vLLM server via SSH (GPU-accelerated)

**Infrastructure:**
- Remote GPU server access via SSH
- Model serving: vLLM / Text Generation Inference / Ollama
- GPU Requirements: 40GB+ VRAM (A100 or equivalent)

## 🚀 Quick Start

### Local Development & Testing

**1. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**2. Install Ollama for Local Testing:**
```bash
# See Ollama_Setup_Guide.md for detailed instructions
ollama run mistral:7b-instruct-q4_0
```

**3. Test Individual Approaches:**
```bash
# Test Approach 1 (Dense Retrieval Only - no LLM needed)
python src/approaches/dense_only.py

# Test Approach 2 (BM25 Only - no LLM needed)
python src/approaches/bm25_only.py

# Test Approach 3-6 (LLM-based - requires Ollama running)
python src/approaches/llm_only.py
python src/approaches/dense_llm.py
python src/approaches/bm25_llm.py
python src/approaches/hybrid_llm.py
```

**4. Run on HPC Cluster (after local testing):**
```bash
# Create benchmarking script
python scripts/run_full_benchmark.py

# Submit SLURM job
sbatch scripts/submit_benchmark.sh
```

### Development vs Production

| Environment | Purpose | Tools | Dataset |
|-------------|---------|-------|---------|
| **Local** | Development & debugging | Sample tools (3-5) | Ollama (CPU/small GPU) |
| **HPC** | Final benchmarking | Full dataset (200+ tools) | vLLM (A100 GPU) |

---

## 📊 Current Status

**Implementation Progress:**
- [x] Project setup and infrastructure
- [x] Dense retrieval implementation (FAISS + embeddings)
- [x] BM25 retrieval implementation (sparse lexical search)
- [x] LLM integration (vLLM + Ollama support, multi-tool selection)
- [x] Approach 1: Dense Retrieval Only (100% on sample data)
- [x] Approach 2: BM25 Only (75% on sample data)
- [x] Approach 3: LLM Only (Full Context)
- [x] Approach 4: Dense + LLM (RAG-MCP) (100% on sample data)
- [x] Approach 5: BM25 + LLM (100% on sample data)
- [ ] Approach 6: Hybrid Retrieval + LLM
- [ ] Unified benchmarking script for all 6 approaches
- [ ] HPC cluster deployment & large-scale evaluation

**Development Workflow:**

**Phase 1: Local Development** (Current)
1. Build all 6 approaches with modular design
2. Test each approach locally with Ollama (3-5 sample tools)
3. Debug and validate implementation
4. Commit each working approach

**Phase 2: HPC Benchmarking** (After all approaches complete)
1. Create unified benchmarking script
2. Prepare SLURM job submission script
3. Deploy to university HPC cluster
4. Run comprehensive evaluation on full dataset (200+ tools)
5. Collect results and perform analysis

