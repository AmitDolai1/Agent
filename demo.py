import time
import json
import statistics
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

class TaskStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    ERROR = "error"

class SafetyLevel(Enum):
    SAFE = "safe"
    WARNING = "warning"
    VIOLATION = "violation"

@dataclass
class TaskResult:
    """Represents the result of a single task execution"""
    task_id: str
    status: TaskStatus
    start_time: float
    end_time: float
    steps_taken: int
    expected_steps: int
    goal_achieved: bool
    errors: List[str] = field(default_factory=list)
    safety_level: SafetyLevel = SafetyLevel.SAFE
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    user_satisfaction: Optional[float] = None  # 1-5 scale
    hallucinations: int = 0
    tool_usage_correct: bool = True
    context_maintained: bool = True
    explanation_clarity: Optional[float] = None  # 1-5 scale
    
    @property
    def completion_time(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def action_efficiency(self) -> float:
        if self.steps_taken == 0:
            return 0.0
        return min(self.expected_steps / self.steps_taken, 1.0)

class AgenticEvaluator:
    """Main evaluation framework for agentic AI applications"""
    
    def __init__(self):
        self.results: List[TaskResult] = []
        self.logger = logging.getLogger(__name__)
        
    def add_result(self, result: TaskResult):
        """Add a task result to the evaluation dataset"""
        self.results.append(result)
    
    def task_completion_rate(self) -> float:
        """Calculate percentage of successfully completed tasks"""
        if not self.results:
            return 0.0
        successful = sum(1 for r in self.results if r.status == TaskStatus.SUCCESS)
        return successful / len(self.results)
    
    def goal_achievement_accuracy(self) -> float:
        """Calculate percentage of tasks where goals were achieved"""
        if not self.results:
            return 0.0
        achieved = sum(1 for r in self.results if r.goal_achieved)
        return achieved / len(self.results)
    
    def average_completion_time(self) -> float:
        """Calculate average time to complete tasks"""
        if not self.results:
            return 0.0
        times = [r.completion_time for r in self.results if r.status == TaskStatus.SUCCESS]
        return statistics.mean(times) if times else 0.0
    
    def completion_time_percentiles(self) -> Dict[str, float]:
        """Calculate completion time percentiles"""
        if not self.results:
            return {}
        times = [r.completion_time for r in self.results if r.status == TaskStatus.SUCCESS]
        if not times:
            return {}
        
        times.sort()
        n = len(times)
        return {
            "p50": times[n//2],
            "p75": times[int(n*0.75)],
            "p90": times[int(n*0.90)],
            "p95": times[int(n*0.95)]
        }
    
    def action_efficiency_score(self) -> float:
        """Calculate average action efficiency across all tasks"""
        if not self.results:
            return 0.0
        efficiencies = [r.action_efficiency for r in self.results]
        return statistics.mean(efficiencies)
    
    def error_rate(self) -> float:
        """Calculate overall error rate"""
        if not self.results:
            return 0.0
        errors = sum(1 for r in self.results if r.status == TaskStatus.ERROR or len(r.errors) > 0)
        return errors / len(self.results)
    
    def hallucination_rate(self) -> float:
        """Calculate rate of hallucination incidents"""
        if not self.results:
            return 0.0
        total_hallucinations = sum(r.hallucinations for r in self.results)
        return total_hallucinations / len(self.results)
    
    def safety_compliance_rate(self) -> Dict[str, float]:
        """Calculate safety compliance metrics"""
        if not self.results:
            return {}
        
        total = len(self.results)
        safe = sum(1 for r in self.results if r.safety_level == SafetyLevel.SAFE)
        warnings = sum(1 for r in self.results if r.safety_level == SafetyLevel.WARNING)
        violations = sum(1 for r in self.results if r.safety_level == SafetyLevel.VIOLATION)
        
        return {
            "safe_rate": safe / total,
            "warning_rate": warnings / total,
            "violation_rate": violations / total
        }
    
    def user_satisfaction_score(self) -> Optional[float]:
        """Calculate average user satisfaction score"""
        scores = [r.user_satisfaction for r in self.results if r.user_satisfaction is not None]
        return statistics.mean(scores) if scores else None
    
    def tool_usage_accuracy(self) -> float:
        """Calculate accuracy of tool usage"""
        if not self.results:
            return 0.0
        correct = sum(1 for r in self.results if r.tool_usage_correct)
        return correct / len(self.results)
    
    def context_retention_rate(self) -> float:
        """Calculate context retention rate"""
        if not self.results:
            return 0.0
        retained = sum(1 for r in self.results if r.context_maintained)
        return retained / len(self.results)
    
    def communication_clarity_score(self) -> Optional[float]:
        """Calculate average communication clarity score"""
        scores = [r.explanation_clarity for r in self.results if r.explanation_clarity is not None]
        return statistics.mean(scores) if scores else None
    
    def resource_utilization_stats(self) -> Dict[str, Any]:
        """Calculate resource utilization statistics"""
        if not self.results:
            return {}
        
        # Aggregate resource usage data
        all_resources = {}
        for result in self.results:
            for resource, value in result.resource_usage.items():
                if resource not in all_resources:
                    all_resources[resource] = []
                all_resources[resource].append(value)
        
        stats = {}
        for resource, values in all_resources.items():
            if values and all(isinstance(v, (int, float)) for v in values):
                stats[resource] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "max": max(values),
                    "min": min(values),
                    "total": sum(values)
                }
        
        return stats
    
    def multi_step_reasoning_success(self) -> float:
        """Calculate success rate for multi-step reasoning tasks"""
        multi_step_results = [r for r in self.results if r.expected_steps > 1]
        if not multi_step_results:
            return 0.0
        
        successful = sum(1 for r in multi_step_results if r.status == TaskStatus.SUCCESS and r.goal_achieved)
        return successful / len(multi_step_results)
    
    def robustness_score(self) -> Dict[str, float]:
        """Calculate robustness metrics"""
        if len(self.results) < 2:
            return {}
        
        success_rates = []
        completion_times = []
        
        # Group results by some criteria (you might want to customize this)
        # For now, we'll use time-based grouping
        sorted_results = sorted(self.results, key=lambda x: x.start_time)
        chunk_size = max(10, len(sorted_results) // 5)  # Create 5 chunks minimum
        
        for i in range(0, len(sorted_results), chunk_size):
            chunk = sorted_results[i:i + chunk_size]
            if len(chunk) >= 3:  # Only consider chunks with enough samples
                chunk_success_rate = sum(1 for r in chunk if r.status == TaskStatus.SUCCESS) / len(chunk)
                chunk_avg_time = statistics.mean([r.completion_time for r in chunk if r.status == TaskStatus.SUCCESS])
                
                success_rates.append(chunk_success_rate)
                if chunk_avg_time > 0:
                    completion_times.append(chunk_avg_time)
        
        robustness = {}
        if len(success_rates) > 1:
            robustness["success_rate_std"] = statistics.stdev(success_rates)
            robustness["success_rate_consistency"] = 1 - (statistics.stdev(success_rates) / statistics.mean(success_rates)) if statistics.mean(success_rates) > 0 else 0
        
        if len(completion_times) > 1:
            robustness["completion_time_std"] = statistics.stdev(completion_times)
            robustness["completion_time_consistency"] = 1 - (statistics.stdev(completion_times) / statistics.mean(completion_times)) if statistics.mean(completion_times) > 0 else 0
        
        return robustness
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report"""
        return {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_tasks_evaluated": len(self.results),
            
            # Task Success Metrics
            "task_success": {
                "completion_rate": self.task_completion_rate(),
                "goal_achievement_accuracy": self.goal_achievement_accuracy(),
                "multi_step_reasoning_success": self.multi_step_reasoning_success()
            },
            
            # Efficiency and Performance
            "efficiency": {
                "average_completion_time": self.average_completion_time(),
                "completion_time_percentiles": self.completion_time_percentiles(),
                "action_efficiency_score": self.action_efficiency_score(),
                "resource_utilization": self.resource_utilization_stats()
            },
            
            # Reliability and Safety
            "reliability": {
                "error_rate": self.error_rate(),
                "hallucination_rate": self.hallucination_rate(),
                "safety_compliance": self.safety_compliance_rate(),
                "robustness": self.robustness_score()
            },
            
            # Interaction Quality
            "interaction_quality": {
                "user_satisfaction_score": self.user_satisfaction_score(),
                "communication_clarity_score": self.communication_clarity_score(),
                "tool_usage_accuracy": self.tool_usage_accuracy(),
                "context_retention_rate": self.context_retention_rate()
            }
        }
    
    def export_results(self, filename: str):
        """Export evaluation results to JSON file"""
        report = self.generate_comprehensive_report()
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
    
    def print_summary(self):
        """Print a human-readable summary of the evaluation"""
        report = self.generate_comprehensive_report()
        
        print("=== Agentic AI Evaluation Summary ===")
        print(f"Total Tasks Evaluated: {report['total_tasks_evaluated']}")
        print()
        
        print("Task Success Metrics:")
        print(f"  Task Completion Rate: {report['task_success']['completion_rate']:.2%}")
        print(f"  Goal Achievement: {report['task_success']['goal_achievement_accuracy']:.2%}")
        print(f"  Multi-step Success: {report['task_success']['multi_step_reasoning_success']:.2%}")
        print()
        
        print("Efficiency Metrics:")
        print(f"  Avg Completion Time: {report['efficiency']['average_completion_time']:.2f}s")
        print(f"  Action Efficiency: {report['efficiency']['action_efficiency_score']:.2%}")
        print()
        
        print("Reliability Metrics:")
        print(f"  Error Rate: {report['reliability']['error_rate']:.2%}")
        print(f"  Hallucination Rate: {report['reliability']['hallucination_rate']:.2f} per task")
        print(f"  Safety Compliance: {report['reliability']['safety_compliance']['safe_rate']:.2%}")
        print()
        
        print("Interaction Quality:")
        if report['interaction_quality']['user_satisfaction_score']:
            print(f"  User Satisfaction: {report['interaction_quality']['user_satisfaction_score']:.2f}/5.0")
        if report['interaction_quality']['communication_clarity_score']:
            print(f"  Communication Clarity: {report['interaction_quality']['communication_clarity_score']:.2f}/5.0")
        print(f"  Tool Usage Accuracy: {report['interaction_quality']['tool_usage_accuracy']:.2%}")
        print(f"  Context Retention: {report['interaction_quality']['context_retention_rate']:.2%}")


# Example usage and testing
def create_sample_data():
    """Create sample data for demonstration"""
    evaluator = AgenticEvaluator()
    
    # Simulate some task results
    import random
    
    for i in range(100):
        start_time = time.time() - random.uniform(0, 1000)
        completion_time = random.uniform(1, 30)
        
        result = TaskResult(
            task_id=f"task_{i:03d}",
            status=random.choices([TaskStatus.SUCCESS, TaskStatus.FAILURE, TaskStatus.PARTIAL, TaskStatus.ERROR], 
                                weights=[70, 15, 10, 5])[0],
            start_time=start_time,
            end_time=start_time + completion_time,
            steps_taken=random.randint(1, 10),
            expected_steps=random.randint(1, 8),
            goal_achieved=random.choice([True, False]),
            errors=[f"Error {j}" for j in range(random.randint(0, 3))],
            safety_level=random.choices([SafetyLevel.SAFE, SafetyLevel.WARNING, SafetyLevel.VIOLATION],
                                      weights=[85, 12, 3])[0],
            resource_usage={
                "api_calls": random.randint(1, 20),
                "tokens_used": random.randint(100, 5000),
                "memory_mb": random.uniform(10, 100)
            },
            user_satisfaction=random.uniform(1, 5) if random.random() > 0.3 else None,
            hallucinations=random.randint(0, 3),
            tool_usage_correct=random.choice([True, False]),
            context_maintained=random.choice([True, False]),
            explanation_clarity=random.uniform(1, 5) if random.random() > 0.4 else None
        )
        
        evaluator.add_result(result)
    
    return evaluator

# Demo
if __name__ == "__main__":
    # Create sample evaluation data
    evaluator = create_sample_data()
    
    # Print summary
    evaluator.print_summary()
    
    # Export detailed report
    evaluator.export_results("agentic_evaluation_report.json")
    print("\nDetailed report exported to 'agentic_evaluation_report.json'")