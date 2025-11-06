"""
Retrieval Metrics Module
Calculates evaluation metrics for retrieval performance.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from collections import defaultdict


class RetrievalMetrics:
    """
    Calculate retrieval quality metrics including Recall@k and Mean Reciprocal Rank (MRR).
    """

    @staticmethod
    def accuracy_at_1(
        results: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate Accuracy@1 - proportion of queries where the first (top-1) 
        retrieved server matches the ground truth server.

        Args:
            results: List of dictionaries with retrieved servers and ground truth

        Returns:
            Accuracy value between 0.0 and 1.0
        """
        if not results:
            return 0.0

        correct_count = 0
        for result in results:
            # Support both old (IDs) and new (servers) format
            retrieved = result.get('retrieved_servers', result.get('retrieved_ids', []))
            ground_truth = result.get('ground_truth_server', result.get('ground_truth_id', ''))
            
            # Check if first retrieved item matches ground truth
            if len(retrieved) > 0 and retrieved[0] == ground_truth:
                correct_count += 1

        return correct_count / len(results)

    @staticmethod
    def recall_at_k(
        retrieved_items: List[str],
        ground_truth_item: str,
        k: Optional[int] = None
    ) -> float:
        """
        Calculate per-instance recall - whether the first (top) retrieved item matches ground truth.
        This is used as a building block for macro-average recall calculation.

        Args:
            retrieved_items: List of retrieved items (servers or IDs, ordered by relevance)
            ground_truth_item: The correct item (server or ID)
            k: Number of top results to consider (None = all)

        Returns:
            1.0 if first retrieved item matches ground truth, 0.0 otherwise
        """
        if not retrieved_items:
            return 0.0
            
        # Check if the first (top) retrieved item matches ground truth
        return 1.0 if retrieved_items[0] == ground_truth_item else 0.0

    @staticmethod
    def reciprocal_rank(
        retrieved_items: List[str],
        ground_truth_item: str,
        k: Optional[int] = None
    ) -> float:
        """
        Calculate reciprocal rank - 1/rank of the ground truth item.

        Args:
            retrieved_items: List of retrieved items (servers or IDs, ordered by relevance)
            ground_truth_item: The correct item (server or ID)
            k: Number of top results to consider (None = all)

        Returns:
            1/rank if found in top-k, 0.0 otherwise
        """
        if k is not None:
            retrieved_items = retrieved_items[:k]

        try:
            rank = retrieved_items.index(ground_truth_item) + 1  # 1-indexed
            return 1.0 / rank
        except ValueError:
            return 0.0

    @staticmethod
    def mean_reciprocal_rank(
        results: List[Dict[str, Any]],
        k: Optional[int] = None
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR) across multiple queries.

        Args:
            results: List of dictionaries with retrieved items and ground truth
            k: Number of top results to consider (None = all)

        Returns:
            Average reciprocal rank across all queries
        """
        if not results:
            return 0.0

        rr_sum = 0.0
        for result in results:
            # Support both old (IDs) and new (servers) format
            retrieved = result.get('retrieved_servers', result.get('retrieved_ids', []))
            ground_truth = result.get('ground_truth_server', result.get('ground_truth_id', ''))

            rr = RetrievalMetrics.reciprocal_rank(retrieved, ground_truth, k)
            rr_sum += rr

        return rr_sum / len(results)

    @staticmethod
    def average_recall_at_k(
        results: List[Dict[str, Any]],
        k: int
    ) -> float:
        """
        Calculate macro-average Recall@k across all server classes.
        
        Treats this as a multi-class classification problem where:
        - Each server is a class
        - The top-ranked retrieved tool's server is the predicted class
        - Calculates recall for each server class individually
        - Returns the unweighted average (macro-average) of per-class recalls
        
        Formula: Macro-average recall = (1/N) * Σ Recall_i
        where N is the number of classes (servers) and Recall_i is recall for class i

        Args:
            results: List of dictionaries with retrieved items and ground truth
            k: Number of top results to consider (not used, always considers top-1)

        Returns:
            Macro-average recall across all server classes
        """
        if not results:
            return 0.0

        # Step 1: Collect all unique server classes (both ground truth and predicted)
        all_servers = set()
        for result in results:
            ground_truth = result.get('ground_truth_server', result.get('ground_truth_id', ''))
            retrieved = result.get('retrieved_servers', result.get('retrieved_ids', []))
            
            all_servers.add(ground_truth)
            if retrieved:
                all_servers.add(retrieved[0])  # Add predicted server (top-1)
        
        # Step 2: Calculate recall for each server class
        per_class_recall = {}
        
        for server_class in all_servers:
            true_positives = 0  # Correctly predicted as this class
            false_negatives = 0  # Should be this class but predicted as another
            
            for result in results:
                ground_truth = result.get('ground_truth_server', result.get('ground_truth_id', ''))
                retrieved = result.get('retrieved_servers', result.get('retrieved_ids', []))
                predicted_server = retrieved[0] if retrieved else None
                
                # Count TP and FN for this server class
                if ground_truth == server_class:
                    if predicted_server == server_class:
                        true_positives += 1
                    else:
                        false_negatives += 1
            
            # Calculate recall for this class
            # Recall = TP / (TP + FN)
            total_actual = true_positives + false_negatives
            if total_actual > 0:
                per_class_recall[server_class] = true_positives / total_actual
            else:
                # No instances of this class in ground truth (only in predictions)
                per_class_recall[server_class] = 0.0
        
        # Step 3: Calculate macro-average (unweighted mean of per-class recalls)
        if not per_class_recall:
            return 0.0
        
        macro_avg_recall = sum(per_class_recall.values()) / len(per_class_recall)
        
        return macro_avg_recall

    @staticmethod
    def precision_at_k(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: Optional[int] = None
    ) -> float:
        """
        Calculate Precision@k - proportion of retrieved items that are relevant.

        Args:
            retrieved_ids: List of retrieved item IDs
            relevant_ids: List of all relevant item IDs
            k: Number of top results to consider (None = all)

        Returns:
            Precision value between 0.0 and 1.0
        """
        if k is not None:
            retrieved_ids = retrieved_ids[:k]

        if not retrieved_ids:
            return 0.0

        relevant_retrieved = sum(1 for item_id in retrieved_ids if item_id in relevant_ids)
        return relevant_retrieved / len(retrieved_ids)

    @staticmethod
    def ndcg_at_k(
        retrieved_ids: List[str],
        relevance_scores: Dict[str, float],
        k: Optional[int] = None
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@k).

        Args:
            retrieved_ids: List of retrieved item IDs (ordered by relevance)
            relevance_scores: Dictionary mapping item IDs to relevance scores
            k: Number of top results to consider (None = all)

        Returns:
            NDCG value between 0.0 and 1.0
        """
        if k is not None:
            retrieved_ids = retrieved_ids[:k]

        if not retrieved_ids:
            return 0.0

        # Calculate DCG
        dcg = 0.0
        for i, item_id in enumerate(retrieved_ids):
            relevance = relevance_scores.get(item_id, 0.0)
            dcg += relevance / np.log2(i + 2)  # i+2 because i starts at 0

        # Calculate IDCG (ideal DCG with perfect ranking)
        sorted_relevance = sorted(relevance_scores.values(), reverse=True)[:len(retrieved_ids)]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(sorted_relevance))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def calculate_all_metrics(
        results: List[Dict[str, Any]],
        k_values: List[int] = [1]
    ) -> Dict[str, float]:
        """
        Calculate all common retrieval metrics.

        Args:
            results: List of dictionaries with 'retrieved_ids' and 'ground_truth_id'
            k_values: List of k values to evaluate (default: [1] for first tool only)

        Returns:
            Dictionary with all calculated metrics
        """
        metrics = {}

        # Calculate Accuracy@1 (first result must match)
        metrics['accuracy@1'] = RetrievalMetrics.accuracy_at_1(results)

        # Calculate Recall@k for each k value (only k=1 by default)
        for k in k_values:
            recall = RetrievalMetrics.average_recall_at_k(results, k)
            metrics[f'recall@{k}'] = recall

        # Calculate MRR for each k value (only k=1 by default)
        for k in k_values:
            mrr = RetrievalMetrics.mean_reciprocal_rank(results, k)
            metrics[f'mrr@{k}'] = mrr

        # Overall MRR (no k limit)
        metrics['mrr'] = RetrievalMetrics.mean_reciprocal_rank(results, k=None)

        return metrics

    @staticmethod
    def analyze_by_category(
        results: List[Dict[str, Any]],
        k_values: List[int] = [1]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics broken down by category.

        Args:
            results: List of dictionaries with 'retrieved_ids', 'ground_truth_id', and 'category'
            k_values: List of k values to evaluate

        Returns:
            Dictionary mapping categories to their metrics
        """
        # Group results by category
        category_results = defaultdict(list)
        for result in results:
            category = result.get('category', 'unknown')
            category_results[category].append(result)

        # Calculate metrics for each category
        category_metrics = {}
        for category, cat_results in category_results.items():
            category_metrics[category] = RetrievalMetrics.calculate_all_metrics(
                cat_results, k_values
            )

        return category_metrics

    @staticmethod
    def analyze_by_difficulty(
        results: List[Dict[str, Any]],
        k_values: List[int] = [1]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics broken down by difficulty level.

        Args:
            results: List of dictionaries with 'retrieved_ids', 'ground_truth_id', and 'difficulty'
            k_values: List of k values to evaluate

        Returns:
            Dictionary mapping difficulty levels to their metrics
        """
        # Group results by difficulty
        difficulty_results = defaultdict(list)
        for result in results:
            difficulty = result.get('difficulty', 'unknown')
            difficulty_results[difficulty].append(result)

        # Calculate metrics for each difficulty level
        difficulty_metrics = {}
        for difficulty, diff_results in difficulty_results.items():
            difficulty_metrics[difficulty] = RetrievalMetrics.calculate_all_metrics(
                diff_results, k_values
            )

        return difficulty_metrics

    @staticmethod
    def get_failure_cases(
        results: List[Dict[str, Any]],
        k: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Get all cases where ground truth was not retrieved at rank 1 (top result).

        Args:
            results: List of dictionaries with retrieved items, ground truth, and query
            k: Number of top results to consider (default=1 for accuracy@1)

        Returns:
            List of failed retrieval cases with details
        """
        failures = []

        for result in results:
            # Support both old (IDs) and new (servers) format
            retrieved = result.get('retrieved_servers', result.get('retrieved_ids', []))[:k]
            ground_truth = result.get('ground_truth_server', result.get('ground_truth_id', ''))

            # For accuracy@1, check if first result doesn't match
            if k == 1:
                if len(retrieved) == 0 or retrieved[0] != ground_truth:
                    failure = {
                        'query': result.get('query', ''),
                        'query_id': result.get('query_id', ''),
                        'ground_truth_server': ground_truth,
                        'retrieved_servers': result.get('retrieved_servers', result.get('retrieved_ids', []))
                    }
                    failures.append(failure)
            else:
                # For recall@k, check if ground truth is in top-k
                if ground_truth not in retrieved:
                    failure = {
                        'query': result.get('query', ''),
                        'ground_truth_server': ground_truth,
                        'retrieved_servers': retrieved,
                        'query_id': result.get('query_id', 'unknown')
                    }
                    failures.append(failure)

        return failures

    @staticmethod
    def print_summary(
        metrics: Dict[str, float],
        title: str = "Retrieval Metrics Summary"
    ) -> None:
        """
        Print a formatted summary of metrics.

        Args:
            metrics: Dictionary of metric names to values
            title: Title for the summary
        """
        print(f"\n{'=' * 60}")
        print(f"{title:^60}")
        print(f"{'=' * 60}")

        # Group metrics by type
        recall_metrics = {k: v for k, v in metrics.items() if 'recall' in k}
        mrr_metrics = {k: v for k, v in metrics.items() if 'mrr' in k}
        time_metrics = {k: v for k, v in metrics.items() if 'time' in k}
        other_metrics = {k: v for k, v in metrics.items()
                        if 'recall' not in k and 'mrr' not in k and 'time' not in k}

        # Print Recall metrics
        if recall_metrics:
            print(f"\n{'Recall Metrics:':^60}")
            print(f"{'-' * 60}")
            for metric, value in sorted(recall_metrics.items()):
                print(f"  {metric:<30} {value:>10.4f} ({value*100:>6.2f}%)")

        # Print MRR metrics
        if mrr_metrics:
            print(f"\n{'MRR Metrics:':^60}")
            print(f"{'-' * 60}")
            for metric, value in sorted(mrr_metrics.items()):
                print(f"  {metric:<30} {value:>10.4f}")
        
        # Print performance metrics
        if time_metrics:
            print(f"\n{'Performance Metrics:':^60}")
            print(f"{'-' * 60}")
            for metric, value in sorted(time_metrics.items()):
                print(f"  {metric:<30} {value:>10.2f} ms")
        
        # Print other metrics
        if other_metrics:
            print(f"\n{'Other Metrics:':^60}")
            print(f"{'-' * 60}")
            for metric, value in sorted(other_metrics.items()):
                print(f"  {metric:<30} {value:>10.4f}")

        print(f"{'=' * 60}\n")
