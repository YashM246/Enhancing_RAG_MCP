import pickle
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import re

class BM25Indexer:

    def __init__(self, tokenizer:str= "simple"):
        self.tokenizer_type = tokenizer
        self.bm25 = None
        self.tools = []
        self.tool_ids = []

    
    def tokenize(self, text:str)-> List[str]:
        
        if self.tokenizer_type == "simple":
            # Simple: lowercase + split by non-alphanumeric
            text = text.lower()
            tokens = re.findall(r'\w+', text)
            return tokens
        
        elif self.tokenizer_type == "advanced":
            # Advanced: lowercase + split + remove stopwords
            text = text.lower()
            tokens = re.findall(r"\w+", text)

            stopwords = {"a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "to", "for"}
            tokens = [t for t in tokens if t not in stopwords and len(t)>1]

            return tokens
        
        else:
            raise ValueError(f"Unknown tokenizer: {self.tokenizer_type}")
    

    def build_index(self, tools:List[Dict[str, Any]])-> None:

        self.tools = tools
        self.tool_ids = [tool["tool_id"] for tool in tools]

        # Combine text field for each tools
        corpus = []
        for tool in tools:
            # Handle usage_example which may be a list or string
            usage_example = tool.get("usage_example", "")
            if isinstance(usage_example, list):
                usage_example = " ".join(usage_example)

            text_parts = [
                tool.get("tool_name", ""),
                tool.get("description", ""),
                usage_example
            ]
            combined_text = " ".join(text_parts)
            corpus.append(combined_text)

        # Tokenize Corpus
        tokenized_corpus = [self.tokenize(doc) for doc in corpus]

        # Build BM25 Index
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"BM25 index built with {len(tools)} tools")
    

    def save_index(self, index_path:str)-> None:
        index_data = {"bm25": self.bm25,
                      "tools": self.tools,
                      "tool_ids": self.tool_ids,
                      "tokenizer_type": self.tokenizer_type}
        
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)

        with open(index_path, "wb") as f:
            pickle.dump(index_data, f)

        print(f"BM25 index saved to {index_path}")

    
    def load_index(self, index_path:str)-> None:
        with open(index_path, "rb") as f:
            index_data = pickle.load(f)

        self.bm25 = index_data["bm25"]
        self.tools = index_data["tools"]
        self.tool_ids = index_data["tool_ids"]
        self.tokenizer_type = index_data["tokenizer_type"]

        print(f"BM25 index loaded from {index_path}")
        print(f"Index contains {len(self.tools)} tools")

if __name__ == "__main__":
    # Example usage
    sample_tools = [
        {
            "tool_id": "tool_001",
            "tool_name": "Weather API",
            "description": "Get current weather data for any location",
            "usage_example": "Get weather for New York"
        },
        {
            "tool_id": "tool_002",
            "tool_name": "Database Query",
            "description": "Execute SQL queries on the database",
            "usage_example": "Query user table"
        },
        {
            "tool_id": "tool_003",
            "tool_name": "File Reader",
            "description": "Read contents from files",
            "usage_example": "Read config.json"
        }
    ]
    
    # Build index
    indexer = BM25Indexer(tokenizer="simple")
    indexer.build_index(sample_tools)
    
    # Save index
    indexer.save_index("data/indexes/bm25_index.pkl")
    
    # Test loading
    indexer2 = BM25Indexer()
    indexer2.load_index("data/indexes/bm25_index.pkl")