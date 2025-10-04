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

**Phase 1: RAG-MCP Baseline (Current)**
- Use semantic retrieval to filter tools *before* presenting them to the LLM
- Store tool descriptions in an external vector index
- Retrieve only top-k most relevant tools for each query
- Dramatically reduce prompt size while maintaining accuracy

**Phase 2: Hybrid RAG-MCP (Planned)**
- Extend semantic retrieval with BM25 sparse retrieval
- Combine dense embeddings (semantic) + keyword matching (BM25)
- Apply fusion techniques (Reciprocal Rank Fusion, weighted combination)
- Optional: Add reranking models for further accuracy improvements

## 📊 Expected Results

Based on Gan & Sun (2025), we expect:

| Approach | Accuracy | Token Reduction | Speed |
|----------|----------|----------------|-------|
| All Tools Baseline | ~13% | 0% (baseline) | Slow |
| RAG-MCP (Semantic) | ~43% | >50% | Fast |
| Hybrid RAG-MCP | **>50%** (goal) | >50% | Fast |

## 🏗️ Architecture

```
User Query → Retriever → Top-k Tools → LLM → Tool Selection
                ↑
         Vector Index
      (50-100 MCP Tools)
```

**Components:**
1. **Tool Indexer:** Embeds tool descriptions into vector space (FAISS)
2. **Retriever:** Semantic search to find relevant tools (top-k)
3. **LLM Selector:** Final tool selection from retrieved candidates
4. **Evaluator:** Measures accuracy, token usage, and latency

## 📁 Project Structure

```
rag-mcp-project/
├── data/
│   ├── tools/              # MCP tool descriptions (50-100 tools)
│   ├── queries/            # Test queries with ground truth
│   └── indexes/            # Pre-built vector indexes
├── src/
│   ├── indexing/           # Tool embedding and indexing
│   ├── retrieval/          # Semantic + hybrid retrieval
│   ├── llm/                # LLM integration and prompting
│   ├── evaluation/         # Metrics and experiment runner
│   └── utils/              # Helper functions
├── notebooks/              # Analysis and visualization
├── results/                # Experiment outputs
├── reports/                # Analysis reports
├── tests/                  # Unit and integration tests
└── docs/                   # Technical documentation
```

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
- `sentence-transformers` - Text embeddings
- `faiss-cpu` - Vector similarity search
- `vllm` or `ollama` - Open-source LLM serving
- `pandas`, `numpy` - Data processing

**Embedding Models:**
- Phase 1: `all-MiniLM-L6-v2` (fast, baseline)
- Optional: `all-mpnet-base-v2` (higher quality)

**LLMs (Self-Hosted):**
- Primary: Mistral 7B Instruct / Mixtral 8x7B Instruct
- Alternative: Qwen2.5-7B-Instruct / LLaMA 3.1-8B-Instruct
- Deployment: vLLM server via SSH (GPU-accelerated)

**Infrastructure:**
- Remote GPU server access via SSH
- Model serving: vLLM / Text Generation Inference / Ollama
- GPU Requirements: 40GB+ VRAM (A100 or equivalent)

## 📊 Current Status

**Phase 1 Progress:**
- [x] Project setup and data collection
- [x] Core implementation (indexing, retrieval, LLM integration)
- [x] Baseline experiments completed
- [x] Results analysis and validation
- [x] Documentation

**Phase 2 Timeline:** 3-4 weeks (after Phase 1 completion)

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

**Team Structure:**
- 6-person team
- Modular codebase for parallel development
- Code reviews required for all PRs