import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from difflib import SequenceMatcher
import ast

@dataclass
class OfflineEvaluationSample:
    """Single evaluation sample for offline metrics"""
    task_id: str
    task_description: str
    ground_truth_actions: List[str]
    predicted_actions: List[str]
    ground_truth_output: Any
    predicted_output: Any
    expected_tools: List[str]
    used_tools: List[str]
    context: Dict[str, Any] = None

class OfflineAgenticMetrics:
    """Comprehensive offline evaluation metrics for agentic AI"""
    
    def __init__(self):
        self.samples: List[OfflineEvaluationSample] = []
    
    def add_sample(self, sample: OfflineEvaluationSample):
        """Add an evaluation sample"""
        self.samples.append(sample)
    
    # 1. Exact Match Accuracy
    def exact_match_accuracy(self) -> float:
        """Percentage of predictions that exactly match ground truth"""
        if not self.samples:
            return 0.0
        
        exact_matches = 0
        for sample in self.samples:
            if str(sample.predicted_output).strip() == str(sample.ground_truth_output).strip():
                exact_matches += 1
        
        return exact_matches / len(self.samples)
    
    # 2. BLEU Score for Text Generation
    def bleu_score(self, n_gram: int = 4) -> float:
        """BLEU score for generated text outputs"""
        from collections import Counter
        
        def get_ngrams(text: str, n: int) -> List[str]:
            tokens = text.lower().split()
            return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
        total_score = 0.0
        valid_samples = 0
        
        for sample in self.samples:
            if not isinstance(sample.predicted_output, str) or not isinstance(sample.ground_truth_output, str):
                continue
            
            reference = sample.ground_truth_output
            candidate = sample.predicted_output
            
            score = 0.0
            for n in range(1, n_gram + 1):
                ref_ngrams = Counter(get_ngrams(reference, n))
                cand_ngrams = Counter(get_ngrams(candidate, n))
                
                if not cand_ngrams:
                    continue
                
                overlap = sum((ref_ngrams & cand_ngrams).values())
                precision = overlap / sum(cand_ngrams.values())
                score += precision / n_gram
            
            total_score += score
            valid_samples += 1
        
        return total_score / valid_samples if valid_samples > 0 else 0.0
    
    # 3. Action Sequence Similarity
    def action_sequence_similarity(self) -> float:
        """Similarity between predicted and ground truth action sequences"""
        similarities = []
        
        for sample in self.samples:
            pred_actions = sample.predicted_actions
            true_actions = sample.ground_truth_actions
            
            # Use sequence matcher for similarity
            similarity = SequenceMatcher(None, pred_actions, true_actions).ratio()
            similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    # 4. Tool Usage Accuracy
    def tool_usage_accuracy(self) -> Dict[str, float]:
        """Accuracy metrics for tool usage"""
        tool_precision_scores = []
        tool_recall_scores = []
        tool_f1_scores = []
        
        for sample in self.samples:
            expected_tools = set(sample.expected_tools)
            used_tools = set(sample.used_tools)
            
            if not expected_tools and not used_tools:
                # Both empty - perfect match
                precision = recall = f1 = 1.0
            elif not expected_tools:
                # Should use no tools but used some
                precision = recall = f1 = 0.0
            elif not used_tools:
                # Should use tools but used none
                precision = recall = f1 = 0.0
            else:
                # Standard precision/recall calculation
                true_positives = len(expected_tools & used_tools)
                precision = true_positives / len(used_tools)
                recall = true_positives / len(expected_tools)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            tool_precision_scores.append(precision)
            tool_recall_scores.append(recall)
            tool_f1_scores.append(f1)
        
        return {
            'precision': np.mean(tool_precision_scores),
            'recall': np.mean(tool_recall_scores),
            'f1': np.mean(tool_f1_scores)
        }
    
    # 5. Plan Quality Score
    def plan_quality_score(self) -> float:
        """Evaluate quality of action planning"""
        quality_scores = []
        
        for sample in self.samples:
            pred_actions = sample.predicted_actions
            true_actions = sample.ground_truth_actions
            
            if not true_actions:
                quality_scores.append(1.0 if not pred_actions else 0.0)
                continue
            
            # Check for optimal length (not too many unnecessary actions)
            length_penalty = abs(len(pred_actions) - len(true_actions)) / len(true_actions)
            length_score = max(0, 1 - length_penalty)
            
            # Check for correct action ordering
            order_score = self._calculate_order_similarity(pred_actions, true_actions)
            
            # Check for presence of critical actions
            critical_actions = set(true_actions)
            used_critical = set(pred_actions) & critical_actions
            coverage_score = len(used_critical) / len(critical_actions)
            
            # Combined quality score
            quality = (length_score * 0.3 + order_score * 0.4 + coverage_score * 0.3)
            quality_scores.append(quality)
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _calculate_order_similarity(self, pred_actions: List[str], true_actions: List[str]) -> float:
        """Calculate similarity in action ordering"""
        if not pred_actions or not true_actions:
            return 0.0
        
        # Find longest common subsequence
        def lcs_length(a, b):
            m, n = len(a), len(b)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if a[i-1] == b[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        lcs_len = lcs_length(pred_actions, true_actions)
        return lcs_len / max(len(pred_actions), len(true_actions))
    
    # 6. Semantic Correctness
    def semantic_correctness_score(self, semantic_evaluator=None) -> float:
        """Evaluate semantic correctness of outputs"""
        if semantic_evaluator is None:
            # Simple rule-based semantic evaluation
            return self._rule_based_semantic_evaluation()
        else:
            # Use custom semantic evaluator (e.g., LLM-based)
            return semantic_evaluator(self.samples)
    
    def _rule_based_semantic_evaluation(self) -> float:
        """Simple rule-based semantic evaluation"""
        correct_samples = 0
        
        for sample in self.samples:
            pred_str = str(sample.predicted_output).lower()
            true_str = str(sample.ground_truth_output).lower()
            
            # Extract numerical values
            pred_numbers = re.findall(r'\d+\.?\d*', pred_str)
            true_numbers = re.findall(r'\d+\.?\d*', true_str)
            
            # Check if key information is preserved
            if pred_numbers and true_numbers:
                # For numerical outputs, check if values are close
                try:
                    pred_vals = [float(x) for x in pred_numbers]
                    true_vals = [float(x) for x in true_numbers]
                    
                    if len(pred_vals) == len(true_vals):
                        # Check if values are within 5% tolerance
                        close_enough = all(
                            abs(p - t) / max(abs(t), 1) < 0.05 
                            for p, t in zip(pred_vals, true_vals)
                        )
                        if close_enough:
                            correct_samples += 1
                except ValueError:
                    # Fall back to string similarity
                    similarity = SequenceMatcher(None, pred_str, true_str).ratio()
                    if similarity > 0.8:
                        correct_samples += 1
            else:
                # For text outputs, use similarity
                similarity = SequenceMatcher(None, pred_str, true_str).ratio()
                if similarity > 0.8:
                    correct_samples += 1
        
        return correct_samples / len(self.samples) if self.samples else 0.0
    
    # 7. Hallucination Detection
    def hallucination_rate(self) -> float:
        """Detect and measure hallucination rate"""
        hallucination_count = 0
        
        for sample in self.samples:
            pred_output = str(sample.predicted_output)
            context = sample.context or {}
            
            # Check for factual inconsistencies
            if self._contains_hallucination(pred_output, context):
                hallucination_count += 1
        
        return hallucination_count / len(self.samples) if self.samples else 0.0
    
    def _contains_hallucination(self, output: str, context: Dict[str, Any]) -> bool:
        """Simple hallucination detection"""
        # Extract claims from output
        claims = self._extract_claims(output)
        
        # Check each claim against context
        for claim in claims:
            if not self._verify_claim_against_context(claim, context):
                return True
        
        return False
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text"""
        # Simple implementation - extract sentences with specific patterns
        sentences = re.split(r'[.!?]+', text)
        claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter very short sentences
                # Look for factual patterns
                if any(pattern in sentence.lower() for pattern in 
                      ['is', 'are', 'was', 'were', 'has', 'have', 'costs', 'worth']):
                    claims.append(sentence)
        
        return claims
    
    def _verify_claim_against_context(self, claim: str, context: Dict[str, Any]) -> bool:
        """Verify if claim is supported by context"""
        claim_lower = claim.lower()
        context_str = json.dumps(context).lower()
        
        # Simple keyword overlap check
        claim_words = set(re.findall(r'\w+', claim_lower))
        context_words = set(re.findall(r'\w+', context_str))
        
        overlap = len(claim_words & context_words)
        return overlap / len(claim_words) > 0.3 if claim_words else True
    
    # 8. Efficiency Metrics
    def action_efficiency_score(self) -> float:
        """Measure efficiency in terms of action usage"""
        efficiency_scores = []
        
        for sample in self.samples:
            optimal_actions = len(sample.ground_truth_actions)
            actual_actions = len(sample.predicted_actions)
            
            if optimal_actions == 0:
                efficiency = 1.0 if actual_actions == 0 else 0.0
            else:
                # Penalize both under-use and over-use
                efficiency = min(1.0, optimal_actions / max(actual_actions, 1))
            
            efficiency_scores.append(efficiency)
        
        return np.mean(efficiency_scores) if efficiency_scores else 0.0
    
    # 9. Error Analysis
    def error_analysis(self) -> Dict[str, Any]:
        """Comprehensive error analysis"""
        error_types = {
            'wrong_tool': 0,
            'missing_action': 0,
            'extra_action': 0,
            'wrong_output_format': 0,
            'factual_error': 0,
            'logical_error': 0
        }
        
        for sample in self.samples:
            # Tool usage errors
            expected_tools = set(sample.expected_tools)
            used_tools = set(sample.used_tools)
            
            if used_tools - expected_tools:
                error_types['wrong_tool'] += 1
            
            # Action sequence errors
            if len(sample.predicted_actions) < len(sample.ground_truth_actions):
                error_types['missing_action'] += 1
            elif len(sample.predicted_actions) > len(sample.ground_truth_actions):
                error_types['extra_action'] += 1
            
            # Output format errors
            if not self._check_output_format(sample.predicted_output, sample.ground_truth_output):
                error_types['wrong_output_format'] += 1
            
            # Factual errors (using hallucination detection)
            if self._contains_hallucination(str(sample.predicted_output), sample.context or {}):
                error_types['factual_error'] += 1
        
        # Convert to percentages
        total_samples = len(self.samples)
        error_percentages = {
            error_type: (count / total_samples) * 100 
            for error_type, count in error_types.items()
        }
        
        return error_percentages
    
    def _check_output_format(self, predicted: Any, expected: Any) -> bool:
        """Check if output format matches expected format"""
        pred_type = type(predicted).__name__
        exp_type = type(expected).__name__
        
        return pred_type == exp_type
    
    # 10. Comprehensive Evaluation Report
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive offline evaluation report"""
        if not self.samples:
            return {"error": "No evaluation samples provided"}
        
        report = {
            "metadata": {
                "total_samples": len(self.samples),
                "evaluation_timestamp": str(np.datetime64('now'))
            },
            "accuracy_metrics": {
                "exact_match_accuracy": self.exact_match_accuracy(),
                "bleu_score": self.bleu_score(),
                "semantic_correctness": self.semantic_correctness_score(),
                "action_sequence_similarity": self.action_sequence_similarity()
            },
            "tool_usage_metrics": self.tool_usage_accuracy(),
            "planning_metrics": {
                "plan_quality_score": self.plan_quality_score(),
                "action_efficiency_score": self.action_efficiency_score()
            },
            "reliability_metrics": {
                "hallucination_rate": self.hallucination_rate()
            },
            "error_analysis": self.error_analysis()
        }
        
        return report

# Sample data generation for testing
def generate_sample_data() -> List[OfflineEvaluationSample]:
    """Generate sample evaluation data"""
    samples = [
        OfflineEvaluationSample(
            task_id="weather_001",
            task_description="Get weather for New York",
            ground_truth_actions=["search_weather", "extract_temperature", "format_response"],
            predicted_actions=["search_weather", "extract_temperature", "format_response"],
            ground_truth_output="The temperature in New York is 72°F with sunny skies.",
            predicted_output="The temperature in New York is 72°F with sunny conditions.",
            expected_tools=["weather_api", "text_formatter"],
            used_tools=["weather_api", "text_formatter"],
            context={"location": "New York", "current_temp": 72, "conditions": "sunny"}
        ),
        OfflineEvaluationSample(
            task_id="math_001",
            task_description="Calculate 15% tip on $45.50",
            ground_truth_actions=["parse_calculation", "compute_percentage", "format_currency"],
            predicted_actions=["parse_calculation", "compute_percentage", "round_result", "format_currency"],
            ground_truth_output="$6.83",
            predicted_output="$6.82",
            expected_tools=["calculator"],
            used_tools=["calculator"],
            context={"base_amount": 45.50, "tip_percentage": 15}
        ),
        OfflineEvaluationSample(
            task_id="search_001",
            task_description="Find information about Python programming",
            ground_truth_actions=["web_search", "extract_relevant_info", "summarize"],
            predicted_actions=["web_search", "web_search", "extract_relevant_info", "summarize"],
            ground_truth_output="Python is a high-level programming language known for its simplicity and readability.",
            predicted_output="Python is a popular programming language that's easy to learn and widely used.",
            expected_tools=["search_engine", "text_processor"],
            used_tools=["search_engine", "text_processor", "web_scraper"],
            context={"query": "Python programming", "search_results": ["Python.org", "tutorials"]}
        )
    ]
    
    return samples

# Example usage
def main():
    # Initialize evaluator
    evaluator = OfflineAgenticMetrics()
    
    # Add sample data
    samples = generate_sample_data()
    for sample in samples:
        evaluator.add_sample(sample)
    
    # Generate comprehensive report
    report = evaluator.generate_comprehensive_report()
    
    print("Offline Agentic AI Evaluation Report")
    print("=" * 50)
    print(json.dumps(report, indent=2))
    
    # Individual metric examples
    print(f"\nKey Offline Metrics:")
    print(f"Exact Match Accuracy: {evaluator.exact_match_accuracy():.2%}")
    print(f"BLEU Score: {evaluator.bleu_score():.3f}")
    print(f"Action Sequence Similarity: {evaluator.action_sequence_similarity():.3f}")
    print(f"Plan Quality Score: {evaluator.plan_quality_score():.3f}")
    print(f"Hallucination Rate: {evaluator.hallucination_rate():.2%}")

if __name__ == "__main__":
    main()