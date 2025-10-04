# RAG-MCP Emulation Project - Execution Plan

## Project Overview
**Goal:** Emulate RAG-MCP methodology on 50-100 tools to establish baseline performance

**Timeline:** 3-4 weeks  
**Success Criteria:** 
- Achieve ~40-45% accuracy with RAG-MCP
- Demonstrate >50% token reduction vs. all-tools baseline
- Complete reproducible implementation

---

## WEEK 1: Setup & Data Preparation

### Environment Setup (Days 1-2)

**Project Infrastructure:**
- [X] Create GitHub repository
- [ ] Set up project structure:
  ```
  rag-mcp-project/
  ├── data/
  │   ├── tools/
  │   ├── queries/
  │   └── results/
  ├── src/
  │   ├── indexing/
  │   ├── retrieval/
  │   ├── llm/
  │   ├── evaluation/
  │   └── utils/
  ├── notebooks/
  ├── tests/
  ├── configs/
  └── docs/
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

### Data Collection (Days 3-5)

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

**Query Dataset Creation:**
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

### Initial Implementation (Days 3-5)

**Indexing Module:**
- [ ] Create `src/indexing/tool_indexer.py`
- [ ] Implement `ToolIndexer` class:
  - [ ] Embedding model initialization (start with all-MiniLM-L6-v2)
  - [ ] Text combination method (name + description + examples)
  - [ ] Embedding generation with batching
  - [ ] FAISS index creation
  - [ ] Index save/load functionality
- [ ] Test basic indexing on sample tools

**Retrieval Module:**
- [ ] Create `src/retrieval/retriever.py`
- [ ] Implement `ToolRetriever` class:
  - [ ] Query encoding
  - [ ] Top-k retrieval from FAISS
  - [ ] Result ranking and formatting
  - [ ] Parameter tuning utilities (different k values)
- [ ] Create `src/retrieval/retrieval_metrics.py` for Recall@k, MRR

**LLM Integration Module:**
- [ ] Create `src/llm/llm_selector.py`
- [ ] Implement `LLMToolSelector` class:
  - [ ] HTTP client setup for vLLM server
  - [ ] Prompt template design (with strict JSON formatting for open models)
  - [ ] Tool formatting for prompt
  - [ ] Response parsing (JSON extraction with fallback regex)
  - [ ] Error handling and retry logic
- [ ] Create `src/utils/token_counter.py` for tracking

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

## WEEK 2: Integration & Baseline Implementation

### Complete Core Components (Days 1-3)

**Finalize Data:**
- [ ] Ensure 50+ tools collected
- [ ] Ensure 50+ queries with ground truth
- [ ] Create data validation script: `scripts/validate_data.py`
- [ ] Document data sources in `data/README.md`

**Complete Indexing:**
- [ ] Test multiple embedding models:
  - [ ] all-MiniLM-L6-v2 (fast)
  - [ ] all-mpnet-base-v2 (better quality)
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

### End-to-End Integration (Days 4-5)

**Main Pipeline:**
- [ ] Create `src/main.py` with CLI interface
- [ ] Implement end-to-end workflow:
  1. Load data
  2. Build/load index
  3. Initialize retriever
  4. Initialize LLM selector (connect to vLLM server)
  5. Run experiments
  6. Save results
- [ ] Add command-line arguments:
  - `--mode` (rag-mcp, all-tools, random)
  - `--k` (top-k value)
  - `--output` (results directory)
  - `--server-url` (vLLM server URL)
- [ ] Test pipeline on 10 sample queries

**Integration Testing:**
- [ ] Create `tests/test_integration.py`
- [ ] Test full workflow end-to-end
- [ ] Verify results format
- [ ] Check for error handling
- [ ] Test server connection resilience

**Baseline Implementations:**
- [ ] Create `src/baselines/all_tools_baseline.py`
  - Give LLM ALL tools at once
- [ ] Create `src/baselines/random_baseline.py`
  - Random k tools selection
- [ ] Test baselines on sample queries
- [ ] Compare baseline vs RAG-MCP qualitatively

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

## WEEK 3: Experimentation & Analysis

### Run Experiments (Days 1-2)

**Prepare for Experiments:**
- [ ] Verify all 50+ queries are ready
- [ ] Ensure server is running and stable
- [ ] Configure logging and result saving
- [ ] Create experiment tracking sheet

**Experiment 1: RAG-MCP (k=3)**
- [ ] Run full query dataset
- [ ] Save results: `results/experiment_1_rag_k3.json`
- [ ] Track:
  - Accuracy
  - Token usage (prompt + completion)
  - Latency per query
  - Server response times
- [ ] Monitor for errors and handle gracefully

**Experiment 2: RAG-MCP (k=5)**
- [ ] Run full query dataset
- [ ] Save results: `results/experiment_2_rag_k5.json`
- [ ] Compare with k=3 results
- [ ] Analyze if more tools help or hurt

**Experiment 3: All Tools Baseline**
- [ ] Run full query dataset
- [ ] Save results: `results/experiment_3_all_tools.json`
- [ ] Expect:
  - Much higher token usage
  - Lower accuracy
- [ ] Document performance degradation

**Experiment 4: Random Baseline**
- [ ] Run full query dataset (should be quick)
- [ ] Save results: `results/experiment_4_random.json`
- [ ] Establish lower bound

**Experiment 5: Ablation Studies**
- [ ] Test different embedding models
- [ ] Test different k values (1, 3, 5, 10)
- [ ] Test different text combinations for indexing
- [ ] Test different prompt formats for Mistral
- [ ] Save results: `results/ablation_studies.json`

---

### Analysis & Visualization (Days 3-5)

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

## WEEK 4: Validation & Documentation

### Validation (Days 1-2)

**Validate Against Paper:**
- [ ] Compare accuracy: Target ~43% (paper result)
- [ ] Compare token reduction: Target >50%
- [ ] Document any deviations
- [ ] If results don't match:
  - [ ] Try different embedding models
  - [ ] Adjust top-k parameter
  - [ ] Review prompt engineering for Mistral
  - [ ] Debug retrieval pipeline
- [ ] Create `reports/validation_against_paper.md`

**Code Quality:**
- [ ] Code review and cleanup
- [ ] Add docstrings to all functions
- [ ] Add type hints
- [ ] Remove debug code
- [ ] Optimize performance where possible
- [ ] Add caching for repeated operations

**Testing:**
- [ ] Write unit tests for core components
- [ ] Test edge cases
- [ ] Verify reproducibility (run same experiment twice)
- [ ] Create `tests/test_*.py` files
- [ ] Target >70% code coverage

---

### Documentation (Days 3-5)

**Code Documentation:**
- [ ] Complete all docstrings
- [ ] Add inline comments for complex logic
- [ ] Create API documentation
- [ ] Document all configuration options

**User Documentation:**
- [ ] Update `README.md` with:
  - [ ] Project description
  - [ ] Installation instructions
  - [ ] Server setup guide
  - [ ] Quick start guide
  - [ ] Usage examples
  - [ ] Configuration options
  - [ ] Results summary
- [ ] Create `CONTRIBUTING.md`
- [ ] Create `docs/user_guide.md`

**Technical Documentation:**
- [ ] Create `docs/architecture.md`:
  - System architecture diagram
  - Component interactions
  - Data flow
- [ ] Create `docs/indexing_pipeline.md`:
  - Embedding model choices
  - Text combination strategies
  - FAISS setup
- [ ] Create `docs/retrieval_documentation.md`:
  - Retrieval algorithms
  - Parameter tuning guide
  - Best practices
- [ ] Create `docs/llm_integration.md`:
  - Prompt engineering guide for Mistral
  - vLLM server setup
  - Response parsing strategies
- [ ] Create `docs/evaluation_methodology.md`:
  - Metrics definitions
  - Evaluation process
  - Results interpretation

**Data Documentation:**
- [ ] Complete `data/README.md`:
  - Data sources and licenses
  - Collection methodology
  - Schema documentation
  - Usage examples
- [ ] Create `docs/ground_truth_methodology.md`

**Reproducibility:**
- [ ] Create `docs/reproducibility_guide.md`:
  - Step-by-step instructions
  - Environment requirements
  - Server setup steps
  - Expected results
  - Troubleshooting guide
- [ ] Test reproduction on clean environment
- [ ] Create reproduction checklist

**Phase 2 Preparation:**
- [ ] Create `docs/phase_2_handoff.md`:
  - Current implementation summary
  - What worked well
  - Known limitations
  - Suggestions for hybrid search
- [ ] Design hybrid retrieval interface (skeleton)
- [ ] Create `configs/phase_2_evaluation_config.yaml`

**Project Retrospective:**
- [ ] Document lessons learned
- [ ] Identify what worked well
- [ ] Identify challenges faced
- [ ] Suggestions for improvement
- [ ] Create `docs/retrospective.md`

---

## Final Deliverables Checklist

### Code:
- [ ] Complete, working RAG-MCP implementation
- [ ] All baseline implementations
- [ ] Comprehensive evaluation framework
- [ ] Unit and integration tests
- [ ] Clean, documented codebase

### Data:
- [ ] 50-100 MCP tool descriptions
- [ ] 50-100 test queries with ground truth
- [ ] Data validation scripts
- [ ] Data documentation

### Results:
- [ ] All experiment results (JSON/CSV)
- [ ] Aggregated results and statistics
- [ ] Comparison tables
- [ ] Visualizations (charts, plots)
- [ ] Analysis reports

### Documentation:
- [ ] README.md (comprehensive)
- [ ] Architecture documentation
- [ ] API documentation
- [ ] User guide
- [ ] Server setup guide
- [ ] Reproducibility guide
- [ ] Technical reports

### Reports:
- [ ] Executive summary
- [ ] Results summary
- [ ] Retrieval analysis
- [ ] LLM selection analysis
- [ ] Error analysis
- [ ] Validation against paper
- [ ] Project retrospective

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

# Server setup (on GPU server)
pip install vllm
vllm serve mistralai/Mistral-7B-Instruct-v0.3 --host 0.0.0.0 --port 8000

# Build index
python src/main.py --build-index

# Run experiments
python src/main.py --mode rag-mcp --k 3 --server-url http://your-server:8000
python src/main.py --mode rag-mcp --k 5 --server-url http://your-server:8000
python src/main.py --mode all-tools --server-url http://your-server:8000
python src/main.py --mode random --server-url http://your-server:8000

# Run evaluation
python src/evaluation/evaluate.py --results results/

# Generate reports
python scripts/generate_report.py

# Run tests
pytest tests/
```

---

## Next Steps: Phase 2 Preview

After completing Phase 1, implement hybrid search:
1. Add BM25 sparse retrieval
2. Implement hybrid fusion (RRF, weighted)
3. Add reranking models (optional)
4. Compare hybrid vs. semantic-only
5. Optimize for best accuracy/efficiency balance

---

