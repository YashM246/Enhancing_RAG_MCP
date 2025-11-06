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

| Approach | Accuracy (Expected) | Token Usage | Latency | Notes |
|----------|---------------------|-------------|---------|-------|
| 1. Dense Retrieval Only (top-1) | Low (~20-30%) | Minimal (0 LLM tokens) | Fastest | No reasoning, pure similarity |
| 2. BM25 Only (top-1) | Low (~15-25%) | Minimal (0 LLM tokens) | Fastest | Keyword-only matching |
| 3. LLM Only (Full Context) | ~13% | Highest (100% baseline) | Slowest | Prompt bloat baseline |
| 4. Dense Retrieval + LLM (top-k) | ~43% | ~50% reduction | Fast | RAG-MCP from paper |
| 5. BM25 + LLM (top-k) | ~35-40% | ~50% reduction | Fast | Lexical filtering |
| 6. Hybrid Retrieval + LLM (top-k) | **>50%** (goal) | ~50% reduction | Fast | Best of both worlds |

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

**Accuracy Metrics:**
- Tool Selection Accuracy (exact match)
- Top-k Accuracy (ground truth in top-k)
- Mean Reciprocal Rank (MRR)

**Efficiency Metrics:**
- Average Prompt Tokens
- Average Completion Tokens
- Query Latency (seconds)
- Cost per Query ($)

**Retrieval Quality:**
- Recall@k (is correct tool retrieved?)
- Precision@k
- Retrieval latency

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

## 📊 Current Status

**Implementation Progress:**
- [x] Project setup and infrastructure
- [x] Dense retrieval implementation (FAISS + embeddings)
- [x] LLM integration (vLLM server + prompt engineering)
- [ ] BM25 retrieval implementation
- [ ] Hybrid fusion implementation
- [ ] All 6 approaches implementation
- [ ] Comprehensive evaluation framework
- [ ] Experimental validation

**Experimental Timeline:**
1. **Baseline Methods** (Approaches 1-3): 1-2 weeks
2. **LLM-Augmented Methods** (Approaches 4-6): 2-3 weeks
3. **Analysis & Reporting**: 1 week

