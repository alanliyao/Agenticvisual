"""
Unified Evaluator

Combines all evaluators into a single interface:
- Objective answer evaluation
- Subjective answer evaluation (LLM Judge)
- Tool call evaluation
- State evaluation

Computes total score based on task type and eval_weights.
"""

import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from .objective_evaluator import ObjectiveEvaluator, EvalResult
from .tool_evaluator import ToolEvaluator, ToolEvalResult
from .state_evaluator import StateEvaluator, StateEvalResult
from .subjective_evaluator import SubjectiveEvaluator, SubjectiveEvalResult


def get_eval_weights_for_task_type(task_type: str) -> Dict[str, float]:
    """Weights by task_type (no fallback; caller must ensure task_type is set)."""
    WEIGHTS_MAP = {
        "clear_single": {"tool_call": 0.8, "final_state": 0.2},
        "clear_multi": {"tool_call": 0.7, "final_state": 0.3},
        "vague_single": {"tool_call": 0.3, "final_state": 0.7},
        "vague_multi": {"tool_call": 0.2, "final_state": 0.8},
    }
    return WEIGHTS_MAP.get(task_type, {"tool_call": 0.6, "final_state": 0.4})


# ============================================================
# Response Extractor
# ============================================================

@dataclass
class ExtractedResponse:
    """Container for extracted structured response"""
    reasoning_rounds: List[str]  # Multiple REASONING blocks (one per iteration)
    key_insights: List[str]      # All KEY_INSIGHTS bullet points
    answer: str                  # Final ANSWER
    raw_response: str            # Original full response


def extract_structured_response(response: str) -> ExtractedResponse:
    """
    Extract ANSWER, multiple REASONING blocks, and KEY_INSIGHTS from response.
    
    Supports:
    - Multiple REASONING: blocks (one per iteration)
    - KEY_INSIGHTS: with bullet points
    - ANSWER: for final answer
    
    Args:
        response: Raw model response text
        
    Returns:
        ExtractedResponse with parsed fields
    """
    if not response:
        return ExtractedResponse(
            reasoning_rounds=[],
            key_insights=[],
            answer="",
            raw_response=""
        )
    
    # Extract ALL REASONING blocks (multiple per response)
    reasoning_pattern = r'REASONING:\s*(.+?)(?=REASONING:|KEY_INSIGHTS:|ANSWER:|$)'
    reasoning_matches = re.findall(reasoning_pattern, response, re.IGNORECASE | re.DOTALL)
    reasoning_rounds = [r.strip() for r in reasoning_matches if r.strip()]
    
    # Extract ANSWER (last occurrence if multiple)
    answer = ""
    answer_matches = re.findall(
        r'ANSWER:\s*(.+?)(?:\n\n|\n(?=[A-Z][A-Z_]*:)|$)', 
        response, 
        re.IGNORECASE | re.DOTALL
    )
    if answer_matches:
        answer = answer_matches[-1].strip()
        # Clean up: remove trailing format instructions if any
        if '\n' in answer:
            # Take only the first line/paragraph for objective answers
            first_line = answer.split('\n')[0].strip()
            # If first line is short enough, use it as the answer
            if len(first_line) < 100:
                answer = first_line
    
    # Extract KEY_INSIGHTS (bullet points)
    key_insights = []
    insights_match = re.search(
        r'KEY_INSIGHTS:\s*(.+?)(?=ANSWER:|$)', 
        response, 
        re.IGNORECASE | re.DOTALL
    )
    if insights_match:
        insights_text = insights_match.group(1)
        key_insights = [
            line.strip().lstrip('- ').lstrip('* ').lstrip('• ')
            for line in insights_text.split('\n')
            if line.strip() and (
                line.strip().startswith('-') or 
                line.strip().startswith('*') or
                line.strip().startswith('•')
            )
        ]
    
    return ExtractedResponse(
        reasoning_rounds=reasoning_rounds,
        key_insights=key_insights,
        answer=answer,
        raw_response=response
    )


# ============================================================
# Unified Evaluation Result
# ============================================================


@dataclass
class UnifiedEvalResult:
    """Complete evaluation result"""
    task_type: str
    question_id: str
    
    # Scores (0-1)
    answer_score: float
    tool_score: float
    state_score: float
    total_score: float
    
    # Detailed results
    answer_details: Dict[str, Any]
    tool_details: Dict[str, Any]
    state_details: Dict[str, Any]
    
    # Metadata
    eval_weights: Dict[str, float]


class UnifiedEvaluator:
    """
    Unified evaluator for all task types.
    
    Supports:
    - clear_single: Objective questions with clear answers
    - open_ended_single: Subjective questions with open-ended answers
    """
    
    def __init__(self):
        """Initialize unified evaluator (always uses real LLM for subjective evaluation)."""
        self.objective_eval = ObjectiveEvaluator()
        self.tool_eval = ToolEvaluator()
        self.state_eval = StateEvaluator()
        self.subjective_eval = SubjectiveEvaluator()
    
    def evaluate_task(self,
                      task_config: Dict,
                      agent_result: Dict,
                      question_idx: int = 0) -> UnifiedEvalResult:
        """
        Evaluate agent result against task configuration.
        
        Args:
            task_config: Task configuration from benchmark JSON
            agent_result: Agent's result containing answer, tool calls, final spec
            question_idx: Index of question to evaluate (for multi-question tasks)
            
        Returns:
            UnifiedEvalResult with all scores and details
        """
        questions = task_config.get("questions", [])
        if isinstance(questions, list) and question_idx < len(questions):
            question = questions[question_idx]
        elif isinstance(questions, dict):
            question = questions
        else:
            question = {}
        
        ground_truth = question.get("ground_truth", {})
        task_type = ground_truth.get("task_type")
        if not task_type:
            raise ValueError(f"question {question_idx} missing ground_truth.task_type")
        
        eval_weights = get_eval_weights_for_task_type(task_type)
        qid = question.get("qid", f"q_{question_idx}")
        
        if "clear" in task_type:
            return self._evaluate_objective(
                task_type, qid, eval_weights, ground_truth, agent_result
            )
        if "vague" in task_type:
            return self._evaluate_subjective(
                task_type, qid, eval_weights, ground_truth, agent_result
            )
        return self._evaluate_objective(
            task_type, qid, eval_weights, ground_truth, agent_result
        )
    
    def _evaluate_objective(self,
                            task_type: str,
                            qid: str,
                            eval_weights: Dict,
                            ground_truth: Dict,
                            agent_result: Dict) -> UnifiedEvalResult:
        """Evaluate objective question"""
        
        # 1. Answer evaluation - extract structured response first
        answer_config = ground_truth.get("answer", {})
        
        # 兼容字符串格式：如果是字符串，包装成 dict
        if isinstance(answer_config, str):
            answer_config = {"type": "categorical", "value": answer_config}
        # 如果已经是 dict 但没有 type，默认 categorical
        elif isinstance(answer_config, dict) and "type" not in answer_config:
            answer_config = {**answer_config, "type": "categorical"}
        
        raw_answer = agent_result.get("answer", agent_result.get("final_answer", ""))
        
        # Extract structured response
        extracted = extract_structured_response(raw_answer)
        
        # Use extracted answer if available, otherwise use raw answer
        agent_answer = extracted.answer if extracted.answer else raw_answer
        
        answer_result = self.objective_eval.evaluate(agent_answer, answer_config)
        answer_score = answer_result.score
        
        # Also check numeric_answer if present
        if "numeric_answer" in ground_truth:
            numeric_config = ground_truth["numeric_answer"]
            numeric_config["type"] = "numeric"
            numeric_result = self.objective_eval.evaluate(agent_answer, numeric_config)
            # Use higher score
            if numeric_result.score > answer_score:
                answer_score = numeric_result.score
                answer_result = numeric_result
        
        # 2. Tool evaluation (tools format only)
        tool_eval_config = ground_truth.get("tool_eval", {})
        called_tools = agent_result.get("tool_calls", agent_result.get("explorations", []))
        if called_tools and isinstance(called_tools[0], dict):
            normalized_tools = []
            for t in called_tools:
                if "tool_execution" in t:
                    te = t["tool_execution"]
                    if te:
                        normalized_tools.append({
                            "name": te.get("tool_name", ""),
                            "params": te.get("parameters", {})
                        })
                else:
                    normalized_tools.append(t)
            called_tools = normalized_tools

        tools_list = tool_eval_config.get("tools")
        if not isinstance(tools_list, list):
            tools_list = []
        tool_result = self.tool_eval.evaluate_sequence(called_tools, tools_list)
        tool_score = tool_result.total_score
        
        # 3. State evaluation
        state_eval_config = ground_truth.get("state_eval")
        final_spec = agent_result.get("final_spec", agent_result.get("vega_spec", {}))
        
        # Determine spec type (vega-lite or vega)
        spec_type = "vega-lite"
        if "$schema" in final_spec:
            if "vega-lite" not in final_spec["$schema"]:
                spec_type = "vega"
        
        state_result = self.state_eval.evaluate(final_spec, state_eval_config, spec_type)
        state_score = state_result.score
        
        # 4. Calculate total score
        # For objective: answer is main score, tool/state are process scores
        tool_weight = eval_weights.get("tool_call", 0.6)
        state_weight = eval_weights.get("final_state", 0.4)
        
        process_score = tool_weight * tool_score + state_weight * state_score
        total_score = 0.6 * answer_score + 0.4 * process_score
        
        return UnifiedEvalResult(
            task_type=task_type,
            question_id=qid,
            answer_score=answer_score,
            tool_score=tool_score,
            state_score=state_score,
            total_score=total_score,
            answer_details=asdict(answer_result) if hasattr(answer_result, '__dataclass_fields__') else {"score": answer_score},
            tool_details=asdict(tool_result),
            state_details=asdict(state_result),
            eval_weights=eval_weights
        )
    
    def _evaluate_subjective(self,
                             task_type: str,
                             qid: str,
                             eval_weights: Dict,
                             ground_truth: Dict,
                             agent_result: Dict) -> UnifiedEvalResult:
        """Evaluate subjective question"""
        
        # 1. Extract agent output - try structured extraction first
        raw_answer = agent_result.get("answer", agent_result.get("final_answer", ""))
        extracted = extract_structured_response(raw_answer)
        
        # Use extracted key_insights if available
        if extracted.key_insights:
            agent_insights = extracted.key_insights
        else:
            agent_insights = agent_result.get("key_insights", agent_result.get("insights", []))
        if isinstance(agent_insights, str):
            agent_insights = [agent_insights]
        
        # Use agent_result.reasoning_rounds, extracted, or explorations
        if agent_result.get("reasoning_rounds"):
            reasoning_rounds = agent_result["reasoning_rounds"]
            if reasoning_rounds and isinstance(reasoning_rounds[0], dict) and "reasoning" not in reasoning_rounds[0]:
                reasoning_rounds = [{"reasoning": r.get("reasoning", "")} for r in reasoning_rounds]
            view_states = []
        elif extracted.reasoning_rounds:
            reasoning_rounds = [{"reasoning": r} for r in extracted.reasoning_rounds]
            view_states = []
        else:
            explorations = agent_result.get("explorations", [])
            reasoning_rounds = []
            view_states = []
            for exp in explorations:
                analysis = exp.get("analysis_summary", {})
                reasoning_rounds.append({"reasoning": analysis.get("reasoning", "")})
                te = exp.get("tool_execution", {})
                if te:
                    view_states.append(te.get("result", {}))
        
        # 2. Extract ground truth
        gt_answer = ground_truth.get("answer", {})
        gt_value = gt_answer.get("value", "") if isinstance(gt_answer, dict) else gt_answer
        gt_insights = [gt_value] if isinstance(gt_value, str) else (gt_value if isinstance(gt_value, list) else [])
        gt_reasoning = ground_truth.get("reasoning", "")
        if isinstance(gt_reasoning, list):
            gt_reasoning = " ".join(
                r.get("reasoning", "") if isinstance(r, dict) else str(r) for r in gt_reasoning
            )
        
        # 3. Subjective evaluation
        subjective_result = self.subjective_eval.evaluate(
            agent_insights=agent_insights,
            agent_reasoning_rounds=reasoning_rounds,
            gt_insights=gt_insights if isinstance(gt_insights, list) else [gt_insights],
            gt_reasoning=gt_reasoning,
            view_states=view_states,
            is_fully_subjective=True
        )
        answer_score = subjective_result.total_score
        
        # 4. Tool evaluation (tools format only)
        tool_eval_config = ground_truth.get("tool_eval", {})
        called_tools = []
        if agent_result.get("tool_calls"):
            for t in agent_result["tool_calls"]:
                called_tools.append({
                    "name": t.get("tool_name", t.get("name", "")),
                    "params": t.get("parameters", t.get("params", {}))
                })
        else:
            for exp in agent_result.get("explorations", []):
                te = exp.get("tool_execution")
                if te:
                    called_tools.append({
                        "name": te.get("tool_name", ""),
                        "params": te.get("parameters", {})
                    })

        tools_list = tool_eval_config.get("tools")
        if not isinstance(tools_list, list):
            tools_list = []
        tool_result = self.tool_eval.evaluate_sequence(called_tools, tools_list)
        tool_score = tool_result.total_score
        
        # 5. State evaluation
        state_eval_config = ground_truth.get("state_eval")
        final_spec = agent_result.get("final_spec", agent_result.get("vega_spec", {}))
        
        spec_type = "vega-lite"
        if "$schema" in final_spec:
            if "vega-lite" not in final_spec["$schema"]:
                spec_type = "vega"
        
        state_result = self.state_eval.evaluate(final_spec, state_eval_config, spec_type)
        state_score = state_result.score
        
        # 6. Calculate total score
        tool_weight = eval_weights.get("tool_call", 0.6)
        state_weight = eval_weights.get("final_state", 0.4)
        
        process_score = tool_weight * tool_score + state_weight * state_score
        total_score = 0.5 * answer_score + 0.5 * process_score
        
        # Convert subjective result to dict
        answer_details = {
            "insight_score": subjective_result.insight_score,
            "reasoning_score": subjective_result.reasoning_score,
            "total_score": subjective_result.total_score,
            "insight_details": asdict(subjective_result.insight_details),
            "reasoning_rounds": len(subjective_result.reasoning_details)
        }
        
        return UnifiedEvalResult(
            task_type=task_type,
            question_id=qid,
            answer_score=answer_score,
            tool_score=tool_score,
            state_score=state_score,
            total_score=total_score,
            answer_details=answer_details,
            tool_details=asdict(tool_result),
            state_details=asdict(state_result),
            eval_weights=eval_weights
        )
    
    def evaluate_batch(self,
                       task_config: Dict,
                       agent_results: List[Dict]) -> List[UnifiedEvalResult]:
        """
        Evaluate multiple agent results for a task.
        
        Args:
            task_config: Task configuration
            agent_results: List of agent results
            
        Returns:
            List of UnifiedEvalResult
        """
        results = []
        questions = task_config.get("questions", [])
        
        if isinstance(questions, list):
            for i, result in enumerate(agent_results):
                if i < len(questions):
                    eval_result = self.evaluate_task(task_config, result, i)
                    results.append(eval_result)
        else:
            for result in agent_results:
                eval_result = self.evaluate_task(task_config, result, 0)
                results.append(eval_result)
        
        return results


def load_task_config(task_path: str) -> Dict:
    """Load task configuration from JSON file"""
    with open(task_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_from_files(task_path: str, result_path: str) -> UnifiedEvalResult:
    """
    Evaluate from task config and result files.
    
    Args:
        task_path: Path to task configuration JSON
        result_path: Path to agent result JSON
        
    Returns:
        UnifiedEvalResult
    """
    task_config = load_task_config(task_path)
    
    with open(result_path, "r", encoding="utf-8") as f:
        agent_result = json.load(f)
    
    evaluator = UnifiedEvaluator()
    return evaluator.evaluate_task(task_config, agent_result)
