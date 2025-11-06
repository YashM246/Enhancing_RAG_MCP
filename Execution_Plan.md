# RAG-MCP Emulation Project - Execution Plan

## Project Overview
**Goal:** Compare 6 different approaches to tool selection for LLMs on 50-100 tools

**Success Criteria:**
- Achieve ~40-45% accuracy with RAG-MCP (Approach 4)
- Demonstrate >50% token reduction vs. all-tools baseline (Approach 3)
- Hybrid approach (Approach 6) should achieve >50% accuracy
- Complete reproducible implementation with comprehensive comparison

---

## 6 Comparison Approaches

| # | Approach Name | Retrieval Method | Selection Method | Top-K | LLM Required | Expected Accuracy |
|---|---------------|------------------|------------------|-------|--------------|-------------------|
| 1 | Dense Retrieval Only | Embeddings (Cosine) | Direct (top-1) | 1 | No | Low (~20-30%) |
| 2 | BM25 Only | BM25 (Lexical) | Direct (top-1) | 1 | No | Low (~15-25%) |
| 3 | LLM Only (Full Context) | None (all tools) | LLM Selection | All | Yes | Low (~13%) |
| 4 | Dense Retrieval + LLM | Embeddings (Cosine) | LLM Selection | 3-10 | Yes | Medium (~43%) |
| 5 | BM25 + LLM | BM25 (Lexical) | LLM Selection | 3-10 | Yes | Medium (~35-40%) |
| 6 | Hybrid Retrieval + LLM | Dense + BM25 Fusion | LLM Selection | 3-10 | Yes | **High (>50% goal)** |

**Key Insights:**
- Approaches 1-2: Fast but limited by pure retrieval quality
- Approach 3: Suffers from prompt bloat (baseline from paper)
- Approach 4: RAG-MCP from Gan & Sun (2025) - our validation target
- Approach 5: Variant using lexical retrieval instead of semantic
- Approach 6: Novel hybrid approach - our main contribution

---

## Phase 1: Setup & Data Preparation

### Environment Setup

**Project Infrastructure:**
- [X] Create GitHub repository
- [X] Set up project structure:
  ```
  Enhancing_RAG_MCP/
  ├── src/
  │   ├── indexing/              # Index building components
  │   │   ├── tool_indexer.py    # Dense embeddings (FAISS) [✓]
  │   │   └── bm25_indexer.py    # Sparse BM25 index [✓]
  │   ├── retrieval/             # Retrieval components
  │   │   ├── dense_retriever.py # Dense/semantic retrieval [TODO]
  │   │   ├── bm25_retriever.py  # Sparse/lexical retrieval [✓]
  │   │   └── hybrid_retriever.py # Hybrid fusion (RRF) [TODO]
  │   ├── approaches/            # Core implementations of 6 approaches
  │   │   ├── dense_only.py      # Approach 1: Dense Retrieval Only [TODO]
  │   │   ├── bm25_only.py       # Approach 2: BM25 Only [IN PROGRESS]
  │   │   ├── llm_only.py        # Approach 3: LLM Only (Full Context) [TODO]
  │   │   ├── dense_llm.py       # Approach 4: Dense + LLM [TODO]
  │   │   ├── bm25_llm.py        # Approach 5: BM25 + LLM [TODO]
  │   │   └── hybrid_llm.py      # Approach 6: Hybrid + LLM [TODO]
  │   ├── llm/                   # LLM integration
  │   │   └── llm_selector.py    # LLM tool selection logic [✓]
  │   └── evaluation/            # Evaluation utilities [TODO]
  ├── benchmarking/              # Evaluation framework
  │   └── benchmarker.py         # Benchmarking suite (dense-only) [✓]
  ├── data/
  │   ├── tools/                 # Tool definitions (JSON)
  │   ├── queries/               # Test queries with ground truth
  │   ├── indexes/               # Pre-built FAISS and BM25 indexes
  │   └── results/               # Experiment results
  ├── tests/                     # Unit and integration tests
  ├── notebooks/                 # Analysis notebooks [TODO]
  └── configs/                   # Configuration files [TODO]
  ```
- [x] Create requirements.txt with dependencies:
  - sentence-transformers
  - faiss-cpu (or faiss-gpu)
  - vllm / ollama (for local model serving)
  - pandas, numpy
  - tqdm
  - python-dotenv
- [x] Set up .env.example for server credentials
- [x] Initialize Git and .gitignore

**Development Environment:**
- [x] Install all dependencies
- [ ] Set up SSH access to GPU server
- [ ] Install vLLM on server: `pip install vllm`
- [ ] Download model: `huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3`
- [ ] Start model server: `vllm serve mistralai/Mistral-7B-Instruct-v0.3 --host 0.0.0.0 --port 8000`
- [ ] Test server connectivity from local machine
- [ ] Test basic imports and model inference

---

### Data Collection

**Tool Collection:**
- [ ] Collect 50-100 MCP tool descriptions from:
  - mcp.so registry
  - GitHub MCP servers
  - Official MCP documentation
- [ ] Organize tools by category:
  - [ ] 20 tools: Web search
  - [ ] 20 tools: Database/data access
  - [ ] 20 tools: File operations
  - [ ] 20 tools: Communication/APIs
  - [ ] 10-20 tools: Miscellaneous

**Tool Data Schema:**
```json
{
  "tool_id": "brave_search_001",
  "tool_name": "Brave Web Search",
  "description": "A web search tool using Brave's Search API...",
  "parameters": {
    "query": "string",
    "count": "integer"
  },
  "usage_example": "Search for recent news about AI",
  "category": "web_search"
}
```

**Query Dataset Creation/Collection:**
- [ ] Create 50-100 test queries with ground truth
- [ ] Distribute queries by difficulty:
  - [ ] 40% easy (clear, unambiguous)
  - [ ] 40% medium (require semantic understanding)
  - [ ] 20% hard (ambiguous/edge cases)

**Query Data Schema:**
```json
{
  "query_id": "q001",
  "query": "Find recent papers about climate change",
  "ground_truth_tool": "academic_search_api",
  "category": "academic",
  "difficulty": "easy"
}
```

**Data Validation:**
- [ ] Validate all tool descriptions
- [ ] Verify ground truth labels
- [ ] Create data statistics report
- [ ] Save as: `data/tools/mcp_tools.json` and `data/queries/test_queries.json`

---

### Initial Implementation

**Indexing Module:**
- [x] Create `src/indexing/tool_indexer.py`
- [x] Implement `ToolIndexer` class:
  - [x] Embedding model initialization (configurable: all-MiniLM-L6-v2, intfloat/e5-base-v2)
  - [x] Text combination method (name + description + examples)
  - [x] Embedding generation with batching
  - [x] FAISS index creation
  - [x] Index save/load functionality
- [x] Test basic indexing on sample tools

**Retrieval Module:**
- [ ] Create `src/retrieval/dense_retriever.py`
- [ ] Implement `DenseRetriever` class:
  - [ ] Query encoding with sentence-transformers
  - [ ] Top-k retrieval from FAISS
  - [ ] Result ranking and formatting
- [ ] Create `src/retrieval/sparse_retriever.py`
- [ ] Implement `BM25Retriever` class:
  - [ ] BM25 index building
  - [ ] Query tokenization
  - [ ] Top-k retrieval with BM25 scoring
- [ ] Create `src/retrieval/hybrid_retriever.py`
- [ ] Implement `HybridRetriever` class:
  - [ ] Reciprocal Rank Fusion (RRF)
  - [ ] Weighted combination option
  - [ ] Top-k retrieval from fused results
- [ ] Create `src/retrieval/retrieval_metrics.py` for Recall@k, MRR

**LLM Integration Module:**
- [x] Create `src/llm/llm_selector.py`
- [x] Implement `LLMToolSelector` class:
  - [x] HTTP client setup for vLLM server
  - [x] Prompt template design (with strict JSON formatting for open models)
  - [x] Tool formatting for prompt
  - [x] Response parsing (JSON extraction with fallback regex)
  - [x] Error handling and retry logic
- [x] Token tracking integrated into LLMToolSelector
- [x] Create comprehensive unit tests (19 tests, all passing)
- [x] Test with mocked responses (no server required)

**Evaluation Module:**
- [ ] Create `src/evaluation/evaluator.py`
- [ ] Implement `Evaluator` class:
  - [ ] Accuracy calculation (exact match)
  - [ ] Top-k accuracy
  - [ ] Mean Reciprocal Rank (MRR)
  - [ ] Token usage tracking
  - [ ] Latency measurement
- [ ] Create `src/evaluation/experiment_runner.py`

**Utilities:**
- [ ] Create `src/utils/data_loader.py` for loading tools/queries
- [ ] Create `src/utils/logger.py` for experiment logging

---

## Phase 2: Integration & Baseline Implementation

### Complete Core Components

**Finalize Data:**
- [ ] Ensure 50+ tools collected
- [ ] Ensure 50+ queries with ground truth
- [ ] Create data validation script: `scripts/validate_data.py`
- [ ] Document data sources in `data/README.md`

**Complete Indexing:**
- [ ] Test multiple embedding models:
  - [ ] all-MiniLM-L6-v2 (fast, baseline)
  - [ ] intfloat/e5-base-v2 (semantic search optimized)
  - [ ] Compare quality in notebook
- [ ] Build full index with all collected tools
- [ ] Implement index versioning
- [ ] Save index: `data/indexes/tools_v1.index`
- [ ] Create `notebooks/embedding_model_comparison.ipynb`

**Complete Retrieval:**
- [ ] Integrate retriever with indexer
- [ ] Test retrieval on sample queries
- [ ] Measure retrieval quality (is ground truth in top-k?)
- [ ] Create visualization tools
- [ ] Create `notebooks/retrieval_analysis.ipynb`

**Complete LLM Integration:**
- [ ] Finalize prompt engineering for Mistral 7B
- [ ] Test different prompt formats (strict JSON output)
- [ ] Implement robust JSON parsing with fallbacks
- [ ] Add comprehensive error handling
- [ ] Test batching capabilities if needed
- [ ] Create `notebooks/prompt_engineering.ipynb`

**Complete Evaluation:**
- [ ] Implement all metrics functions
- [ ] Create experiment configuration system: `configs/experiment_config.yaml`
- [ ] Build results aggregation utilities
- [ ] Test evaluation on sample runs

---

### End-to-End Integration

**Main Pipeline:**
- [ ] Create `src/main.py` with CLI interface
- [ ] Implement end-to-end workflow:
  1. Load data
  2. Build/load indexes (FAISS + BM25)
  3. Initialize retrievers (dense, sparse, hybrid)
  4. Initialize LLM selector (connect to vLLM server)
  5. Run experiments
  6. Save results
- [ ] Add command-line arguments:
  - `--approach` (1-6: dense-only, bm25-only, llm-only, dense-llm, bm25-llm, hybrid-llm)
  - `--k` (top-k value for LLM-based approaches)
  - `--output` (results directory)
  - `--server-url` (vLLM server URL, only needed for approaches 3-6)
- [ ] Test pipeline on 10 sample queries for each approach

**Integration Testing:**
- [ ] Create `tests/test_integration.py`
- [ ] Test full workflow end-to-end
- [ ] Verify results format
- [ ] Check for error handling
- [ ] Test server connection resilience

**All 6 Approach Implementations:**
- [ ] **Approach 1:** `src/approaches/dense_only.py` - Dense Retrieval Only (top-1)
  - Use cosine similarity on embeddings, return top-1 tool directly
- [ ] **Approach 2:** `src/approaches/bm25_only.py` - BM25 Only (top-1)
  - Use BM25 lexical search, return top-1 tool directly
- [ ] **Approach 3:** `src/approaches/llm_only.py` - LLM Only (Full Context)
  - Give LLM ALL tools at once (naive MCP baseline)
- [ ] **Approach 4:** `src/approaches/dense_llm.py` - Dense Retrieval + LLM (top-k)
  - RAG-MCP from paper: retrieve top-k with embeddings, LLM selects
- [ ] **Approach 5:** `src/approaches/bm25_llm.py` - BM25 + LLM (top-k)
  - Retrieve top-k with BM25, LLM selects
- [ ] **Approach 6:** `src/approaches/hybrid_llm.py` - Hybrid Retrieval + LLM (top-k)
  - Combine dense + BM25 with fusion, LLM selects from top-k
- [ ] Test all approaches on sample queries
- [ ] Verify correct behavior and output format consistency

**Documentation:**
- [ ] Update `README.md` with:
  - Project overview
  - Setup instructions (including server setup)
  - Usage examples
  - Project structure
- [ ] Create `docs/api_interfaces.md`
- [ ] Create `docs/architecture.md`
- [ ] Create `docs/server_setup.md` (vLLM configuration)

---

## Phase 3: Experimentation & Analysis

### Run Experiments

**Prepare for Experiments:**
- [ ] Verify all 50+ queries are ready
- [ ] Ensure server is running and stable
- [ ] Configure logging and result saving
- [ ] Create experiment tracking sheet

**Experiment 1: Dense Retrieval Only (top-1)**
- [ ] Run full query dataset
- [ ] Save results: `results/exp1_dense_only.json`
- [ ] Track: Accuracy, Latency (no LLM tokens)
- [ ] Note: Fastest baseline, no reasoning

**Experiment 2: BM25 Only (top-1)**
- [ ] Run full query dataset
- [ ] Save results: `results/exp2_bm25_only.json`
- [ ] Track: Accuracy, Latency (no LLM tokens)
- [ ] Compare with dense retrieval

**Experiment 3: LLM Only (Full Context)**
- [ ] Run full query dataset
- [ ] Save results: `results/exp3_llm_only.json`
- [ ] Track: Accuracy, Token usage (highest), Latency (slowest)
- [ ] Expect: Low accuracy (~13%), prompt bloat

**Experiment 4: Dense Retrieval + LLM (k=3,5,10)**
- [ ] Run with k=3: `results/exp4_dense_llm_k3.json`
- [ ] Run with k=5: `results/exp4_dense_llm_k5.json`
- [ ] Run with k=10: `results/exp4_dense_llm_k10.json`
- [ ] Track: Accuracy (~43% expected), Token reduction (>50%), Latency
- [ ] This is RAG-MCP from paper

**Experiment 5: BM25 + LLM (k=3,5,10)**
- [ ] Run with k=3: `results/exp5_bm25_llm_k3.json`
- [ ] Run with k=5: `results/exp5_bm25_llm_k5.json`
- [ ] Run with k=10: `results/exp5_bm25_llm_k10.json`
- [ ] Track: Accuracy, Token reduction, Latency
- [ ] Compare with dense+LLM

**Experiment 6: Hybrid Retrieval + LLM (k=3,5,10)**
- [ ] Run with k=3: `results/exp6_hybrid_llm_k3.json`
- [ ] Run with k=5: `results/exp6_hybrid_llm_k5.json`
- [ ] Run with k=10: `results/exp6_hybrid_llm_k10.json`
- [ ] Track: Accuracy (>50% goal), Token reduction, Latency
- [ ] Expected: Best overall performance

**Experiment 7: Ablation Studies**
- [ ] Test different fusion methods (RRF vs weighted)
- [ ] Test different embedding models
- [ ] Test different BM25 tokenization strategies
- [ ] Save results: `results/ablation_studies.json`

---

### Analysis & Visualization

**Results Aggregation:**
- [ ] Aggregate all experiment results
- [ ] Calculate summary statistics
- [ ] Create comparison tables
- [ ] Save: `results/aggregated_results.csv`

**Accuracy Analysis:**
- [ ] Calculate accuracy for each experiment
- [ ] Compare RAG-MCP vs. baselines
- [ ] Statistical significance testing
- [ ] Create accuracy comparison chart

**Token Usage Analysis:**
- [ ] Calculate average prompt tokens per experiment
- [ ] Calculate average completion tokens
- [ ] Create token usage comparison chart
- [ ] Verify >50% reduction with RAG-MCP

**Latency Analysis:**
- [ ] Calculate average latency per query
- [ ] Identify bottlenecks (retrieval vs. LLM vs. network)
- [ ] Create latency distribution plots
- [ ] Analyze server response times

**Retrieval Quality Analysis:**
- [ ] Calculate Recall@k for different k values
- [ ] Calculate Mean Reciprocal Rank (MRR)
- [ ] Identify cases where ground truth not retrieved
- [ ] Analyze false negatives
- [ ] Create retrieval quality report: `reports/retrieval_analysis.md`

**LLM Selection Analysis:**
- [ ] Cases where tool retrieved but not selected
- [ ] Prompt effectiveness evaluation
- [ ] JSON parsing success rate
- [ ] Error pattern identification
- [ ] Create LLM analysis report: `reports/llm_selection_analysis.md`

**Error Analysis:**
- [ ] Categorize all failures:
  - Retrieval failures (correct tool not in top-k)
  - Selection failures (retrieved but not chosen)
  - Parsing failures (invalid JSON output)
  - Execution failures (errors)
- [ ] Analyze by query difficulty
- [ ] Analyze by query category
- [ ] Identify problematic queries
- [ ] Create error analysis report: `reports/error_analysis.md`

**Embedding Analysis:**
- [ ] Create t-SNE/UMAP visualization of tool embeddings
- [ ] Identify tool clusters
- [ ] Analyze semantic similarity issues
- [ ] Create `notebooks/embedding_analysis.ipynb`

**Visualization Creation:**
- [ ] Accuracy comparison bar chart
- [ ] Token usage comparison (prompt vs. completion)
- [ ] Latency distribution boxplot
- [ ] Retrieval quality curves (Recall@k)
- [ ] Embedding space visualization
- [ ] Save all charts in `visualizations/`

**Reports:**
- [ ] Create `reports/results_summary.md`
- [ ] Create `reports/executive_summary.md`
- [ ] Create `docs/comparison_with_paper.md` (compare to Gan & Sun results)

---

## Success Metrics

### Technical Metrics:
- ✅ RAG-MCP accuracy: 40-45% (target: match paper's 43%)
- ✅ Token reduction: >50% vs. all-tools baseline
- ✅ Retrieval Recall@k: >80%
- ✅ All 50+ queries processed successfully
- ✅ Reproducible results

### Quality Metrics:
- ✅ Code coverage: >70%
- ✅ Documentation: Complete
- ✅ Tests: Passing
- ✅ Clean code: No linting errors

### Process Metrics:
- ✅ All milestones completed
- ✅ Results validated against paper
- ✅ Ready for Phase 2 (Hybrid Search)

---

## Key Commands Reference

```bash
# Setup
pip install -r requirements.txt
python scripts/validate_data.py

# Server setup (on GPU server - only needed for approaches 3-6)
pip install vllm
vllm serve mistralai/Mistral-7B-Instruct-v0.3 --host 0.0.0.0 --port 8000

# Build indexes
python src/main.py --build-index --index-type all  # Build both FAISS and BM25

# Run experiments - All 6 approaches

# Approach 1: Dense Retrieval Only (no LLM needed)
python src/main.py --approach 1

# Approach 2: BM25 Only (no LLM needed)
python src/main.py --approach 2

# Approach 3: LLM Only (Full Context)
python src/main.py --approach 3 --server-url http://your-server:8000

# Approach 4: Dense Retrieval + LLM (RAG-MCP)
python src/main.py --approach 4 --k 3 --server-url http://your-server:8000
python src/main.py --approach 4 --k 5 --server-url http://your-server:8000
python src/main.py --approach 4 --k 10 --server-url http://your-server:8000

# Approach 5: BM25 + LLM
python src/main.py --approach 5 --k 3 --server-url http://your-server:8000
python src/main.py --approach 5 --k 5 --server-url http://your-server:8000

# Approach 6: Hybrid Retrieval + LLM
python src/main.py --approach 6 --k 3 --server-url http://your-server:8000
python src/main.py --approach 6 --k 5 --server-url http://your-server:8000

# Run evaluation across all approaches
python src/evaluation/evaluate.py --results results/

# Generate comparison reports
python scripts/generate_report.py --compare-all

# Run tests
pytest tests/
```

---

## Key Implementation Notes

**Approach Dependencies:**
- Approaches 1 & 4: Require FAISS dense index
- Approaches 2 & 5: Require BM25 sparse index
- Approach 3 & 6: Require both indexes
- Approaches 3-6: Require vLLM server connection

**Recommended Implementation Order:**
1. Complete Approaches 1 & 2 (pure retrieval, no LLM needed)
2. Complete Approach 3 (LLM-only baseline)
3. Complete Approach 4 (RAG-MCP from paper - validate against published results)
4. Complete Approach 5 (BM25+LLM variant)
5. Complete Approach 6 (Hybrid - expected best performance)

**Critical Success Factors:**
- Approach 4 accuracy should match paper (~43%)
- Approach 3 should show low accuracy (~13%) due to prompt bloat
- Approach 6 should outperform all others (>50% target)
- Token reduction consistent across approaches 4-6 (~50%)

---

