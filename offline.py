import time
import json
from collections import defaultdict
from typing import List, Dict, Any, Optional, Callable, Tuple

# --- 1. Data Structures for Evaluation ---

class AgentTask:
    """Represents a single task for the agent to perform."""
    def __init__(self,
                 task_id: str,
                 input_data: Dict[str, Any],
                 expected_output: Dict[str, Any],
                 ground_truth_steps: Optional[List[Dict[str, Any]]] = None, # For step-level accuracy
                 expected_tool_calls: Optional[List[Dict[str, Any]]] = None, # For tool selection accuracy
                 context: Optional[str] = None):
        self.task_id = task_id
        self.input_data = input_data
        self.expected_output = expected_output
        self.ground_truth_steps = ground_truth_steps
        self.expected_tool_calls = expected_tool_calls
        self.context = context

class AgentRunResult:
    """Stores the results of an agent's execution for a single task."""
    def __init__(self,
                 task_id: str,
                 agent_output: Dict[str, Any],
                 start_time: float,
                 end_time: float,
                 llm_token_usage: Dict[str, int], # e.g., {'prompt_tokens': 100, 'completion_tokens': 50}
                 agent_logs: List[Dict[str, Any]], # Detailed log of agent's internal decisions/actions
                 tool_calls_made: List[Dict[str, Any]], # Actual tool calls made by the agent
                 final_decision_turn_count: int,
                 recognized_failure: bool = False,
                 intermediate_steps: Optional[List[Dict[str, Any]]] = None): # For step-level analysis
        self.task_id = task_id
        self.agent_output = agent_output
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time
        self.llm_token_usage = llm_token_usage
        self.agent_logs = agent_logs
        self.tool_calls_made = tool_calls_made
        self.final_decision_turn_count = final_decision_turn_count
        self.recognized_failure = recognized_failure
        self.intermediate_steps = intermediate_steps

# --- 2. Metric Calculation Functions ---

class AgentEvaluationMetrics:
    def __init__(self):
        pass

    # --- Effectiveness/Performance ---

    def calculate_task_success_rate(self,
                                    agent_run_results: List[AgentRunResult],
                                    tasks: Dict[str, AgentTask],
                                    success_criteria_fn: Callable[[Dict[str, Any], Dict[str, Any]], bool]) -> float:
        """
        Calculates the percentage of tasks completed correctly end-to-end.
        `success_criteria_fn` is a function that defines what constitutes success
        by comparing agent_output and expected_output.
        """
        successful_tasks = 0
        for result in agent_run_results:
            task = tasks[result.task_id]
            if success_criteria_fn(result.agent_output, task.expected_output):
                successful_tasks += 1
        return (successful_tasks / len(agent_run_results)) * 100 if agent_run_results else 0

    def calculate_task_completion_rate(self, agent_run_results: List[AgentRunResult]) -> float:
        """Percentage of inquiries successfully resolved (not necessarily perfectly, but completed)."""
        completed_tasks = sum(1 for r in agent_run_results if r.agent_output is not None)
        return (completed_tasks / len(agent_run_results)) * 100 if agent_run_results else 0

    def calculate_step_level_accuracy(self,
                                      agent_run_results: List[AgentRunResult],
                                      tasks: Dict[str, AgentTask],
                                      step_accuracy_fn: Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], float]) -> float:
        """
        Calculates the average accuracy of actions taken within a larger workflow.
        `step_accuracy_fn` compares agent's intermediate steps to ground truth steps.
        """
        total_accuracy = 0.0
        evaluated_tasks = 0
        for result in agent_run_results:
            task = tasks[result.task_id]
            if task.ground_truth_steps and result.intermediate_steps:
                total_accuracy += step_accuracy_fn(result.intermediate_steps, task.ground_truth_steps)
                evaluated_tasks += 1
        return total_accuracy / evaluated_tasks if evaluated_tasks else 0.0

    def calculate_precision_recall_f1(self,
                                       agent_run_results: List[AgentRunResult],
                                       tasks: Dict[str, AgentTask],
                                       metric_type: str = "information_retrieval", # or "generation"
                                       get_retrieved_items_fn: Optional[Callable[[Dict[str, Any]], List[Any]]] = None,
                                       get_relevant_items_fn: Optional[Callable[[Dict[str, Any]], List[Any]]] = None) -> Dict[str, float]:
        """
        Calculates Precision, Recall, and F1-score.
        Requires functions to extract retrieved/generated items and relevant ground truth items.
        """
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        count = 0

        for result in agent_run_results:
            task = tasks[result.task_id]
            if metric_type == "information_retrieval" and get_retrieved_items_fn and get_relevant_items_fn:
                retrieved_items = set(get_retrieved_items_fn(result.agent_output))
                relevant_items = set(get_relevant_items_fn(task.expected_output))

                if not relevant_items: # Avoid division by zero if no relevant items exist
                    if not retrieved_items:
                        precision, recall, f1 = 1.0, 1.0, 1.0 # Perfect if nothing to retrieve and nothing retrieved
                    else:
                        precision, recall, f1 = 0.0, 0.0, 0.0 # Retrieved something when nothing was relevant
                else:
                    true_positives = len(retrieved_items.intersection(relevant_items))
                    false_positives = len(retrieved_items.difference(relevant_items))
                    false_negatives = len(relevant_items.difference(retrieved_items))

                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
                    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                total_precision += precision
                total_recall += recall
                total_f1 += f1
                count += 1
            # Add logic for "generation" metric_type if needed (e.g., using ROUGE/BLEU scores, which require NLTK/HuggingFace libraries)

        return {
            "precision": total_precision / count if count else 0.0,
            "recall": total_recall / count if count else 0.0,
            "f1": total_f1 / count if count else 0.0
        }

    def calculate_tool_utilization_efficacy(self,
                                             agent_run_results: List[AgentRunResult],
                                             tasks: Dict[str, AgentTask],
                                             tool_usage_accuracy_fn: Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], float]) -> float:
        """
        Measures how accurately and efficiently the agent uses external tools or APIs.
        `tool_usage_accuracy_fn` compares actual tool calls to expected tool calls.
        """
        total_efficacy = 0.0
        evaluated_tasks = 0
        for result in agent_run_results:
            task = tasks[result.task_id]
            if task.expected_tool_calls and result.tool_calls_made:
                total_efficacy += tool_usage_accuracy_fn(result.tool_calls_made, task.expected_tool_calls)
                evaluated_tasks += 1
        return total_efficacy / evaluated_tasks if evaluated_tasks else 0.0

    def calculate_memory_coherence_retrieval(self,
                                              agent_run_results: List[AgentRunResult],
                                              tasks: Dict[str, AgentTask],
                                              memory_evaluation_fn: Callable[[Dict[str, Any], str, List[Dict[str, Any]]], float]) -> float:
        """
        Assesses how well the agent utilizes retrieved information and maintains context.
        `memory_evaluation_fn` would typically analyze agent_logs and context.
        This is highly specific and would require detailed log parsing.
        """
        total_score = 0.0
        evaluated_tasks = 0
        for result in agent_run_results:
            task = tasks[result.task_id]
            # This function would need to inspect agent_logs for memory access patterns
            # and compare retrieved info with the task's context/ground_truth_steps.
            # Example: memory_evaluation_fn(result.agent_logs, task.context, task.ground_truth_relevant_memories)
            # For now, this is a placeholder.
            # if task.context: # Assuming context is relevant to memory
            #     total_score += memory_evaluation_fn(result.agent_logs, task.context, ...)
            #     evaluated_tasks += 1
            pass # Placeholder for now, as implementation is very specific
        return total_score / evaluated_tasks if evaluated_tasks else 0.0

    # --- Efficiency ---

    def calculate_average_task_duration(self, agent_run_results: List[AgentRunResult]) -> float:
        """Average time taken for tasks."""
        if not agent_run_results:
            return 0.0
        return sum(r.duration for r in agent_run_results) / len(agent_run_results)

    def calculate_response_time(self, agent_run_results: List[AgentRunResult]) -> float:
        """Average time to acknowledge and begin processing requests (initial latency)."""
        # This typically needs to be logged within the agent's execution itself,
        # measuring the time from input reception to the first meaningful internal action.
        # For this example, we'll assume `start_time` in AgentRunResult is when processing truly begins.
        return self.calculate_average_task_duration(agent_run_results) # Simplification for now

    def calculate_llm_cost_per_task(self, agent_run_results: List[AgentRunResult],
                                     llm_cost_per_token: Dict[str, float]) -> float:
        """
        Calculates the average LLM cost per task based on token usage.
        `llm_cost_per_token` example: {'prompt': 0.0001, 'completion': 0.0002}
        """
        total_cost = 0.0
        for result in agent_run_results:
            prompt_tokens = result.llm_token_usage.get('prompt_tokens', 0)
            completion_tokens = result.llm_token_usage.get('completion_tokens', 0)
            total_cost += (prompt_tokens * llm_cost_per_token.get('prompt', 0)) + \
                          (completion_tokens * llm_cost_per_token.get('completion', 0))
        return total_cost / len(agent_run_results) if agent_run_results else 0.0

    def calculate_latency_per_agent_loop(self, agent_run_results: List[AgentRunResult]) -> float:
        """
        Average time taken for each iteration of the agent's decision-making process.
        Requires detailed logging of loop iterations within the agent.
        """
        total_latency = 0.0
        total_loops = 0
        for result in agent_run_results:
            # Assuming agent_logs contain entries like {'event': 'loop_start', 'timestamp': ...}
            # and {'event': 'loop_end', 'timestamp': ...}
            loop_start_times = [log['timestamp'] for log in result.agent_logs if log.get('event') == 'loop_start']
            loop_end_times = [log['timestamp'] for log in result.agent_logs if log.get('event') == 'loop_end']

            # Simple pairing of start/end, more robust parsing needed for complex logs
            for i in range(min(len(loop_start_times), len(loop_end_times))):
                total_latency += (loop_end_times[i] - loop_start_times[i])
                total_loops += 1
        return total_latency / total_loops if total_loops else 0.0

    # --- Autonomy ---

    def calculate_decision_turn_count(self, agent_run_results: List[AgentRunResult]) -> float:
        """Average number of actions taken without human intervention."""
        if not agent_run_results:
            return 0.0
        return sum(r.final_decision_turn_count for r in agent_run_results) / len(agent_run_results)

    def calculate_self_aware_failure_rate(self, agent_run_results: List[AgentRunResult]) -> float:
        """Measures how often the agent recognizes its limitations and handles failures gracefully."""
        if not agent_run_results:
            return 0.0
        total_failures = len(agent_run_results) # Assuming each result represents a task attempt
        recognized_failures = sum(1 for r in agent_run_results if r.recognized_failure)
        return (recognized_failures / total_failures) * 100 if total_failures else 0.0

    # --- Accuracy/Reliability ---

    def calculate_tool_action_selection_accuracy(self,
                                                 agent_run_results: List[AgentRunResult],
                                                 tasks: Dict[str, AgentTask],
                                                 selection_accuracy_fn: Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], float]) -> float:
        """
        Did the agent choose the right tool or action at each step?
        Similar to tool utilization efficacy, but focused purely on selection correctness.
        `selection_accuracy_fn` would compare agent's selected actions (from logs) against ground truth.
        """
        return self.calculate_tool_utilization_efficacy(agent_run_results, tasks, selection_accuracy_fn) # Reusing similar logic

    def calculate_hallucination_rate(self,
                                     agent_run_results: List[AgentRunResult],
                                     tasks: Dict[str, AgentTask],
                                     hallucination_detection_fn: Callable[[Dict[str, Any], Dict[str, Any]], bool]) -> float:
        """
        How often the agent generates incorrect or unsupported information.
        `hallucination_detection_fn` would typically involve comparing generated content
        (e.g., from `agent_output`) against `expected_output` or `context` for factual errors.
        """
        hallucinations = 0
        total_generations = 0
        for result in agent_run_results:
            task = tasks[result.task_id]
            if 'generated_text' in result.agent_output: # Assuming generated text is in agent_output
                total_generations += 1
                if hallucination_detection_fn(result.agent_output, task.expected_output): # Or task.context
                    hallucinations += 1
        return (hallucinations / total_generations) * 100 if total_generations else 0.0

    def calculate_reflection_accuracy(self,
                                     agent_run_results: List[AgentRunResult],
                                     reflection_evaluation_fn: Callable[[List[Dict[str, Any]]], float]) -> float:
        """
        How effectively the agent improves by iterating on its past actions.
        This requires logging of reflection steps and an evaluation function that assesses
        the quality of the reflection (e.g., did it identify the correct error, did it propose a good fix).
        Highly agent-specific.
        """
        total_reflection_accuracy = 0.0
        evaluated_reflections = 0
        for result in agent_run_results:
            # Assuming agent_logs contain information about reflection cycles
            # if 'reflection_steps' in result.agent_logs:
            #     total_reflection_accuracy += reflection_evaluation_fn(result.agent_logs['reflection_steps'])
            #     evaluated_reflections += 1
            pass # Placeholder
        return total_reflection_accuracy / evaluated_reflections if evaluated_reflections else 0.0

    def calculate_intent_resolution(self,
                                    agent_run_results: List[AgentRunResult],
                                    tasks: Dict[str, AgentTask],
                                    intent_resolution_fn: Callable[[Dict[str, Any], Dict[str, Any]], bool]) -> float:
        """
        How well the agent understands and addresses the user's intent.
        `intent_resolution_fn` compares the agent's perceived intent (from logs/output)
        with the true intent (from task.input_data or a dedicated ground truth field).
        """
        resolved_intents = 0
        total_intents = 0
        for result in agent_run_results:
            task = tasks[result.task_id]
            total_intents += 1
            if intent_resolution_fn(result.agent_output, task.input_data): # Or task.expected_output for implied intent
                resolved_intents += 1
        return (resolved_intents / total_intents) * 100 if total_intents else 0.0

    def calculate_context_relevance(self,
                                    agent_run_results: List[AgentRunResult],
                                    tasks: Dict[str, AgentTask],
                                    context_relevance_fn: Callable[[List[str], str, Dict[str, Any]], float]) -> float:
        """
        Assesses if retrieved information is relevant to the current query or task.
        `context_relevance_fn` would evaluate retrieved context snippets (from agent logs)
        against the task's input query and expected output.
        """
        total_relevance_score = 0.0
        evaluated_tasks = 0
        for result in agent_run_results:
            task = tasks[result.task_id]
            # Assuming agent_logs contain a list of retrieved context snippets
            retrieved_contexts = [log['context_snippet'] for log in result.agent_logs if log.get('event') == 'context_retrieval']
            if retrieved_contexts:
                total_relevance_score += context_relevance_fn(retrieved_contexts, task.input_data.get('query', ''), task.expected_output)
                evaluated_tasks += 1
        return total_relevance_score / evaluated_tasks if evaluated_tasks else 0.0

    def calculate_faithfulness(self,
                              agent_run_results: List[AgentRunResult],
                              tasks: Dict[str, AgentTask],
                              faithfulness_fn: Callable[[str, List[str]], bool]) -> float:
        """
        Checks if generated answers are grounded in the retrieved context.
        `faithfulness_fn` would take the generated answer and the context snippets it was based on,
        and verify if the answer can be directly supported by the context.
        """
        faithful_answers = 0
        total_answers = 0
        for result in agent_run_results:
            task = tasks[result.task_id]
            generated_answer = result.agent_output.get('generated_answer')
            retrieved_contexts = [log['context_snippet'] for log in result.agent_logs if log.get('event') == 'context_retrieval']

            if generated_answer and retrieved_contexts:
                total_answers += 1
                if faithfulness_fn(generated_answer, retrieved_contexts):
                    faithful_answers += 1
        return (faithful_answers / total_answers) * 100 if total_answers else 0.0

    def calculate_answer_similarity(self,
                                   agent_run_results: List[AgentRunResult],
                                   tasks: Dict[str, AgentTask],
                                   similarity_fn: Callable[[str, str], float]) -> float:
        """
        Compares generated answers to ground truth answers (when available).
        `similarity_fn` could use metrics like BLEU, ROUGE, cosine similarity on embeddings, etc.
        (Requires libraries like NLTK, scikit-learn, or sentence-transformers).
        """
        total_similarity = 0.0
        compared_answers = 0
        for result in agent_run_results:
            task = tasks[result.task_id]
            generated_answer = result.agent_output.get('generated_answer')
            ground_truth_answer = task.expected_output.get('final_answer')

            if generated_answer and ground_truth_answer:
                total_similarity += similarity_fn(generated_answer, ground_truth_answer)
                compared_answers += 1
        return total_similarity / compared_answers if compared_answers else 0.0

    # --- Robustness ---

    def calculate_error_rate(self, agent_run_results: List[AgentRunResult]) -> float:
        """Percentage of incorrect responses or failed operations."""
        if not agent_run_results:
            return 0.0
        # This assumes 'success' is determined elsewhere or failure is explicitly marked.
        # For a simple error rate, we can count tasks that did not complete successfully.
        failed_tasks = sum(1 for r in agent_run_results if r.agent_output is None or not r.agent_output.get('success', False))
        return (failed_tasks / len(agent_run_results)) * 100 if agent_run_results else 0.0

# --- 3. Helper Functions and Example Implementations ---

# Dummy functions for demonstration. In a real application, these would be sophisticated.

def dummy_success_criteria(agent_output: Dict[str, Any], expected_output: Dict[str, Any]) -> bool:
    """Example success criteria: checks if a 'status' key is 'success' and 'value' matches."""
    return agent_output.get('status') == 'success' and agent_output.get('value') == expected_output.get('value')

def dummy_step_accuracy_fn(agent_steps: List[Dict[str, Any]], ground_truth_steps: List[Dict[str, Any]]) -> float:
    """Simplified step accuracy: checks if the first step's action matches."""
    if not agent_steps or not ground_truth_steps:
        return 0.0
    return 1.0 if agent_steps[0].get('action') == ground_truth_steps[0].get('action') else 0.0

def dummy_tool_usage_accuracy_fn(actual_calls: List[Dict[str, Any]], expected_calls: List[Dict[str, Any]]) -> float:
    """Simplified tool usage accuracy: checks if the first tool call name matches."""
    if not actual_calls or not expected_calls:
        return 0.0
    return 1.0 if actual_calls[0].get('tool_name') == expected_calls[0].get('tool_name') else 0.0

def dummy_hallucination_detection_fn(agent_output: Dict[str, Any], expected_output: Dict[str, Any]) -> bool:
    """Dummy hallucination: if agent output contains 'fictional' keyword."""
    return "fictional" in agent_output.get('generated_text', '').lower()

def dummy_intent_resolution_fn(agent_output: Dict[str, Any], input_data: Dict[str, Any]) -> bool:
    """Dummy intent resolution: checks if 'resolved_intent' in output matches a key in input."""
    return agent_output.get('resolved_intent') == input_data.get('true_intent')

def dummy_context_relevance_fn(retrieved_contexts: List[str], query: str, expected_output: Dict[str, Any]) -> float:
    """Dummy context relevance: checks if query keywords are in retrieved contexts."""
    score = 0.0
    query_keywords = query.lower().split()
    for context in retrieved_contexts:
        if any(keyword in context.lower() for keyword in query_keywords):
            score += 1.0
    return score / len(retrieved_contexts) if retrieved_contexts else 0.0

def dummy_faithfulness_fn(generated_answer: str, retrieved_contexts: List[str]) -> bool:
    """Dummy faithfulness: checks if a keyword from the answer is in any context."""
    if not retrieved_contexts: return False
    answer_keywords = generated_answer.lower().split()[:3] # Check first few keywords
    return any(keyword in context.lower() for keyword in answer_keywords for context in retrieved_contexts)

def dummy_similarity_fn(text1: str, text2: str) -> float:
    """Dummy similarity: simple word overlap ratio."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 and not words2: return 1.0
    return len(words1.intersection(words2)) / len(words1.union(words2))

# --- 4. Simulation of Agent Execution (for offline evaluation) ---

def simulate_agent_run(task: AgentTask) -> AgentRunResult:
    """
    Simulates an agent processing a task.
    In a real scenario, this would be your actual agent's `run` method.
    """
    start_time = time.time()
    # Simulate agent's internal processing
    time.sleep(0.1 + (len(task.input_data.get('query', '')) * 0.001)) # Simulate variable processing time

    # Dummy agent logic
    agent_output = {}
    agent_logs = []
    tool_calls_made = []
    llm_token_usage = {'prompt_tokens': 50, 'completion_tokens': 20}
    final_decision_turn_count = 3
    recognized_failure = False
    intermediate_steps = []

    # Simulate success/failure based on task ID
    if "success" in task.task_id:
        agent_output = {'status': 'success', 'value': task.expected_output.get('value'), 'generated_answer': "This is the correct answer."}
        agent_logs.append({'event': 'loop_start', 'timestamp': time.time() - 0.05})
        agent_logs.append({'event': 'context_retrieval', 'timestamp': time.time() - 0.03, 'context_snippet': "Relevant context found here."})
        tool_calls_made.append({'tool_name': 'search_api', 'parameters': {'query': 'test'}})
        agent_logs.append({'event': 'loop_end', 'timestamp': time.time()})
        intermediate_steps.append({'action': 'retrieve_info'})
        intermediate_steps.append({'action': 'generate_response'})
    elif "fail" in task.task_id:
        agent_output = {'status': 'failure', 'error': 'Could not process request.', 'generated_answer': "I'm sorry, I couldn't complete this."}
        recognized_failure = True
        agent_logs.append({'event': 'loop_start', 'timestamp': time.time() - 0.05})
        agent_logs.append({'event': 'loop_end', 'timestamp': time.time()})
    elif "hallucination" in task.task_id:
        agent_output = {'status': 'success', 'value': 'some_value', 'generated_answer': "This is a fictional fact about the moon."}
        agent_logs.append({'event': 'loop_start', 'timestamp': time.time() - 0.05})
        agent_logs.append({'event': 'loop_end', 'timestamp': time.time()})
    elif "irrelevant" in task.task_id:
        agent_output = {'status': 'success', 'value': 'some_value', 'generated_answer': "Here is some information."}
        agent_logs.append({'event': 'loop_start', 'timestamp': time.time() - 0.05})
        agent_logs.append({'event': 'context_retrieval', 'timestamp': time.time() - 0.03, 'context_snippet': "Completely irrelevant information."})
        agent_logs.append({'event': 'loop_end', 'timestamp': time.time()})
    else:
        agent_output = {'status': 'success', 'value': 'default_value', 'generated_answer': "Default generated answer."}
        agent_logs.append({'event': 'loop_start', 'timestamp': time.time() - 0.05})
        agent_logs.append({'event': 'loop_end', 'timestamp': time.time()})


    end_time = time.time()

    return AgentRunResult(
        task_id=task.task_id,
        agent_output=agent_output,
        start_time=start_time,
        end_time=end_time,
        llm_token_usage=llm_token_usage,
        agent_logs=agent_logs,
        tool_calls_made=tool_calls_made,
        final_decision_turn_count=final_decision_turn_count,
        recognized_failure=recognized_failure,
        intermediate_steps=intermediate_steps
    )

def load_synthetic_tasks(filepath: str) -> Dict[str, AgentTask]:
    """Loads synthetic tasks from a JSON file."""
    with open(filepath, 'r') as f:
        tasks_data = json.load(f)
    tasks = {}
    for task_data in tasks_data:
        tasks[task_data['task_id']] = AgentTask(
            task_id=task_data['task_id'],
            input_data=task_data['input_data'],
            expected_output=task_data['expected_output'],
            ground_truth_steps=task_data.get('ground_truth_steps'),
            expected_tool_calls=task_data.get('expected_tool_calls'),
            context=task_data.get('context')
        )
    return tasks

# --- 5. Main Evaluation Runner ---

def run_offline_evaluation(tasks_filepath: str, llm_cost_per_token: Dict[str, float]) -> Dict[str, Any]:
    """
    Main function to run the offline evaluation.
    """
    tasks = load_synthetic_tasks(tasks_filepath)
    agent_run_results: List[AgentRunResult] = []

    print(f"Running evaluation on {len(tasks)} tasks...")
    for task_id, task in tasks.items():
        print(f"  Processing task: {task_id}")
        run_result = simulate_agent_run(task)
        agent_run_results.append(run_result)

    metrics_calculator = AgentEvaluationMetrics()
    all_metrics = {}

    # Effectiveness/Performance
    all_metrics['task_success_rate'] = metrics_calculator.calculate_task_success_rate(
        agent_run_results, tasks, dummy_success_criteria
    )
    all_metrics['task_completion_rate'] = metrics_calculator.calculate_task_completion_rate(agent_run_results)
    all_metrics['step_level_accuracy'] = metrics_calculator.calculate_step_level_accuracy(
        agent_run_results, tasks, dummy_step_accuracy_fn
    )
    all_metrics['precision_recall_f1'] = metrics_calculator.calculate_precision_recall_f1(
        agent_run_results, tasks, metric_type="information_retrieval",
        get_retrieved_items_fn=lambda x: [x.get('value')], # Example: retrieve 'value'
        get_relevant_items_fn=lambda x: [x.get('value')] # Example: expected 'value'
    )
    all_metrics['tool_utilization_efficacy'] = metrics_calculator.calculate_tool_utilization_efficacy(
        agent_run_results, tasks, dummy_tool_usage_accuracy_fn
    )
    # MCR is highly custom, leaving as placeholder
    # all_metrics['memory_coherence_retrieval'] = metrics_calculator.calculate_memory_coherence_retrieval(...)

    # Efficiency
    all_metrics['average_task_duration'] = metrics_calculator.calculate_average_task_duration(agent_run_results)
    all_metrics['response_time'] = metrics_calculator.calculate_response_time(agent_run_results)
    all_metrics['llm_cost_per_task'] = metrics_calculator.calculate_llm_cost_per_task(
        agent_run_results, llm_cost_per_token
    )
    all_metrics['latency_per_agent_loop'] = metrics_calculator.calculate_latency_per_agent_loop(agent_run_results)

    # Autonomy
    all_metrics['decision_turn_count'] = metrics_calculator.calculate_decision_turn_count(agent_run_results)
    all_metrics['self_aware_failure_rate'] = metrics_calculator.calculate_self_aware_failure_rate(agent_run_results)

    # Accuracy/Reliability
    all_metrics['tool_action_selection_accuracy'] = metrics_calculator.calculate_tool_action_selection_accuracy(
        agent_run_results, tasks, dummy_tool_usage_accuracy_fn # Reusing dummy for tool selection
    )
    all_metrics['hallucination_rate'] = metrics_calculator.calculate_hallucination_rate(
        agent_run_results, tasks, dummy_hallucination_detection_fn
    )
    # Reflection accuracy is highly custom, leaving as placeholder
    # all_metrics['reflection_accuracy'] = metrics_calculator.calculate_reflection_accuracy(...)
    all_metrics['intent_resolution'] = metrics_calculator.calculate_intent_resolution(
        agent_run_results, tasks, dummy_intent_resolution_fn
    )
    all_metrics['context_relevance'] = metrics_calculator.calculate_context_relevance(
        agent_run_results, tasks, dummy_context_relevance_fn
    )
    all_metrics['faithfulness'] = metrics_calculator.calculate_faithfulness(
        agent_run_results, tasks, dummy_faithfulness_fn
    )
    all_metrics['answer_similarity'] = metrics_calculator.calculate_answer_similarity(
        agent_run_results, tasks, dummy_similarity_fn
    )

    # Robustness
    all_metrics['error_rate'] = metrics_calculator.calculate_error_rate(agent_run_results)

    return all_metrics

# --- Example Usage ---

if __name__ == "__main__":
    # Create a dummy synthetic tasks file for demonstration
    dummy_tasks_data = [
        {
            "task_id": "task_1_success",
            "input_data": {"query": "Find the capital of France", "true_intent": "information_retrieval"},
            "expected_output": {"status": "success", "value": "Paris", "final_answer": "The capital of France is Paris."},
            "ground_truth_steps": [{"action": "retrieve_info", "tool": "knowledge_base"}],
            "expected_tool_calls": [{"tool_name": "knowledge_base", "parameters": {"query": "capital of France"}}]
        },
        {
            "task_id": "task_2_fail",
            "input_data": {"query": "Solve this complex math problem: (1000! * 999!) / (1001!)", "true_intent": "math_calculation"},
            "expected_output": {"status": "failure", "value": None},
            "ground_truth_steps": [],
            "expected_tool_calls": []
        },
        {
            "task_id": "task_3_hallucination",
            "input_data": {"query": "Tell me about unicorns"},
            "expected_output": {"status": "success", "value": "mythical creature", "final_answer": "Unicorns are mythical creatures."},
            "ground_truth_steps": [],
            "expected_tool_calls": []
        },
        {
            "task_id": "task_4_irrelevant_context",
            "input_data": {"query": "What is the weather like today?"},
            "expected_output": {"status": "success", "value": "weather_info", "final_answer": "The weather is sunny."},
            "ground_truth_steps": [],
            "expected_tool_calls": []
        }
    ]

    tasks_filepath = "synthetic_tasks.json"
    with open(tasks_filepath, 'w') as f:
        json.dump(dummy_tasks_data, f, indent=4)

    llm_costs = {'prompt': 0.0001, 'completion': 0.0002} # Example costs per token

    evaluation_results = run_offline_evaluation(tasks_filepath, llm_costs)

    print("\n--- Offline Evaluation Results ---")
    for metric, value in evaluation_results.items():
        if isinstance(value, dict):
            print(f"{metric}:")
            for sub_metric, sub_value in value.items():
                print(f"  - {sub_metric}: {sub_value:.4f}")
        else:
            print(f"{metric}: {value:.4f}")

    # Clean up the dummy file
    import os
    os.remove(tasks_filepath)