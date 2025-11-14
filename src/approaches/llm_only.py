# Approach 1: LLM Only (Full Context)
# Gives LLM all tools at once without retrieval (often has prompt bloat problem)

from typing import Dict, Any, List
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.llm.llm_selector import LLMToolSelector

class LLMOnlyApproach:
    # Gives the LLM all available tools without any retrieval filtering
    # This demonstrates the prompt bloat problem when tool count scales

    def __init__(self,
                 tools: List[Dict[str, Any]],
                 server_url: str= "http://localhost:11434",
                 model_name: str= "mistral:7b-instruct-q4_0",
                 backend: str= "ollama"):
        self.tools = tools
        self.llm_selector = LLMToolSelector(
            server_url= server_url,
            model_name= model_name,
            backend= backend,
            temperature= 0.1,
            max_tokens= 500
        )
        self.approach_name = "LLM Only"
        self.num_tools = len(tools)

    
    def select_tool(self, query:str)-> Dict[str, Any]:
        # Selects tools using LLM with full context (all tools)
        #
        # Returns Dictionary containing:
        #           - selected_tools: List of selected tool IDs
        #           - num_tools_selected: Number of tools selected
        #           - prompt_tokens: Tokens used in prompt
        #           - completion_tokens: Tokens in response
        #           - latency: Time taken in seconds
        #           - approach: Name of approach
        #           - num_tools_in_prompt: Total tools given to LLM

        import time
        start_time = time.perf_counter()

        # Pass all tools to LLM
        result = self.llm_selector.select_tool(
            query= query,
            candidate_tools= self.tools
        )

        latency = time.perf_counter() - start_time

        response = {
            "query": query,
            "selected_tools": result["selected_tools"],
            "num_tools_selected": result["num_tools_selected"],
            "prompt_tokens": result["usage"]["prompt_tokens"],
            "completion_tokens": result["usage"]["completion_tokens"],
            "total_tokens": result["usage"]["total_tokens"],
            "latency_seconds": latency,
            "approach": self.approach_name,
            "num_tools_in_prompt": self.num_tools,
            "raw_response": result.get("raw_response", "")
        }
        
        return response
    
    def evaluate_query(self,
                       query:str,
                       ground_truth_server: str) -> Dict[str, Any]:
        # Evaluate tool selection for a single query
        # ground_truth_server: Server name (e.g., "Weather API", "Unit Converter")

        result = self.select_tool(query)

        # Check if selection is correct (server-level comparison)
        # Extract server names from selected tool IDs
        selected_servers = []
        for tool_id in result["selected_tools"]:
            # Find the tool in self.tools
            tool = next((t for t in self.tools if t["tool_id"] == tool_id), None)
            if tool:
                selected_servers.append(tool.get("server", "Unknown"))

        is_correct = ground_truth_server in selected_servers

        evaluation = {
            **result,
            "selected_servers": selected_servers,
            "ground_truth_server": ground_truth_server,
            "is_correct": is_correct,
            "accuracy": 1.0 if is_correct else 0.0
        }

        return evaluation
    
if __name__ == "__main__":
    print("Testing Approach: LLM Only")
    print("=" * 60)
    
    # Sample tools for testing
    sample_tools = [
        {
            "tool_id": "tool_001",
            "tool_name": "Weather API",
            "description": "Get current weather data for any location",
            "usage_example": "Get weather for New York",
            "server": "Weather API"
        },
        {
            "tool_id": "tool_002",
            "tool_name": "Database Query",
            "description": "Execute SQL queries on the database",
            "usage_example": "Query user table",
            "server": "Database Tools"
        },
        {
            "tool_id": "tool_003",
            "tool_name": "File Reader",
            "description": "Read contents from files",
            "usage_example": "Read config.json",
            "server": "File Operations"
        }
    ]
    
    print(f"\nInitializing with {len(sample_tools)} tools...")
    print("NOTE: Make sure Ollama is running with mistral:7b-instruct-q4_0")
    print("Run: ollama run mistral:7b-instruct-q4_0")
    print("-" * 60)
    
    # Initialize approach
    try:
        approach = LLMOnlyApproach(
            tools=sample_tools,
            server_url="http://localhost:11434",
            model_name="mistral:7b-instruct-q4_0",
            backend="ollama"
        )
        
        # Test queries
        test_cases = [
            {
                "query": "get current weather data",
                "ground_truth": "Weather API",
                "description": "Should select Weather API"
            },
            {
                "query": "run SQL query on database",
                "ground_truth": "Database Query",
                "description": "Should select Database Query"
            },
            {
                "query": "read file from disk",
                "ground_truth": "File Reader",
                "description": "Should select File Reader"
            },
            {
                "query": "what is the temperature outside",
                "ground_truth": "Weather API",
                "description": "Semantic query - LLM should understand"
            }
        ]
        
        print("\nRunning test queries...")
        print("=" * 60)
        
        results = []
        for test_case in test_cases:
            print(f"\n{'='*60}")
            print(f"Query: '{test_case['query']}'")
            print(f"Expected: {test_case['ground_truth']} ({test_case['description']})")
            print(f"{'='*60}")
            
            evaluation = approach.evaluate_query(
                test_case["query"],
                test_case["ground_truth"]
            )
            
            results.append(evaluation)

            print(f"Selected tools: {evaluation['selected_tools']}")
            print(f"Selected servers: {evaluation['selected_servers']}")
            print(f"Num selected: {evaluation['num_tools_selected']}")
            print(f"Correct: {'✓' if evaluation['is_correct'] else '✗'}")
            print(f"Prompt tokens: {evaluation['prompt_tokens']}")
            print(f"Completion tokens: {evaluation['completion_tokens']}")
            print(f"Total tokens: {evaluation['total_tokens']}")
            print(f"Latency: {evaluation['latency_seconds']*1000:.2f}ms")
            print(f"Tools in prompt: {evaluation['num_tools_in_prompt']} (ALL tools)")
        
        # Summary statistics
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        total_correct = sum(r['is_correct'] for r in results)
        accuracy = total_correct / len(results) * 100
        avg_latency = sum(r['latency_seconds'] for r in results) / len(results) * 1000
        avg_prompt_tokens = sum(r['prompt_tokens'] for r in results) / len(results)
        avg_total_tokens = sum(r['total_tokens'] for r in results) / len(results)
        
        print(f"Total queries: {len(results)}")
        print(f"Correct: {total_correct}")
        print(f"Accuracy: {accuracy:.1f}%")
        print(f"Average latency: {avg_latency:.2f}ms")
        print(f"Average prompt tokens: {avg_prompt_tokens:.0f}")
        print(f"Average total tokens: {avg_total_tokens:.0f}")
        print(f"Tools in prompt: {approach.num_tools} (no retrieval filtering)")
        
        print(f"\n{'='*60}")
        print("NOTE: With only 3 tools, LLM should perform well.")
        print("With 200+ tools, expect ~13% accuracy due to prompt bloat.")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure:")
        print("1. Ollama is installed and running")
        print("2. Run: ollama run mistral:7b-instruct-q4_0")
        print("3. Ollama is accessible at http://localhost:11434")
        import traceback
        traceback.print_exc()