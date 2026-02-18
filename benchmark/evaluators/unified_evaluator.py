"""
Unified Evaluator with Agent-as-Judge

Combines all evaluators:
- Objective answer evaluation
- Subjective answer evaluation (LLM Judge)
- Tool call evaluation (F1 + params)
- State evaluation
- Structured reasoning evaluation (tool-aligned)
- Agent-as-Judge for ambiguous scores in [0.4, 0.7]

Score composition (3 dimensions):
  1. answer_score: answer correctness
  2. tool_reasoning_score: tool calls (F1) + structured reasoning quality
  3. state_score: final visualization state correctness
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from .objective_evaluator import ObjectiveEvaluator, EvalResult
from .tool_evaluator import ToolEvaluator, ToolEvalResult
from .state_evaluator import StateEvaluator, StateEvalResult
from .subjective_evaluator import SubjectiveEvaluator, SubjectiveEvalResult


# ============================================================
# Configuration
# ============================================================

AGENT_JUDGE_LOW_THRESHOLD = 0.4
AGENT_JUDGE_HIGH_THRESHOLD = 0.7

# Score weights for each task type
SCORE_WEIGHTS = {
    "objective": {"answer": 0.50, "tool_reasoning": 0.25, "state": 0.25},
    "subjective": {"answer": 0.30, "tool_reasoning": 0.35, "state": 0.35},
}

# Within tool_reasoning, how to split between tool and reasoning
TOOL_REASONING_SPLIT = {"tool": 0.5, "reasoning": 0.5}

AGENT_JUDGE_PROMPT = """You are an expert evaluator performing a second-round review of an AI agent's performance on a visualization analysis task.

## Task Context
Task type: {task_type}
Question: {question_text}

## Rule-Based Scores (first-round)
- Answer score: {answer_score:.3f}
- Tool+Reasoning score: {tool_reasoning_score:.3f}
- State score: {state_score:.3f}
- Combined score: {combined_score:.3f}

## Agent's Response
Answer: {agent_answer}

## Ground Truth
Expected answer: {gt_answer}
Expected tools: {gt_tools}

## Score Details
{score_details}

## Your Task
The combined score {combined_score:.3f} falls in the ambiguous range [{low_threshold}, {high_threshold}].
Review and determine:
1. PASS — agent substantially completed the task despite imperfect rule matching
2. FAIL — agent fundamentally failed despite partial rule matching
3. BORDERLINE — score is fair as-is

Consider: task intent understanding, tool call reasonableness (alternative valid approaches), state closeness, answer correctness.

Output JSON only:
{{
    "verdict": "PASS" or "FAIL" or "BORDERLINE",
    "adjusted_score": <0.0-1.0>,
    "reasoning": "<brief explanation>"
}}"""


# ============================================================
# Response Extractor
# ============================================================

@dataclass
class ExtractedResponse:
    reasoning_rounds: List[str]
    key_insights: List[str]
    answer: str


def extract_structured_response(response: str) -> ExtractedResponse:
    """Extract ANSWER, REASONING, KEY_INSIGHTS from raw response text."""
    if not response:
        return ExtractedResponse(reasoning_rounds=[], key_insights=[], answer="")

    # Extract reasoning blocks
    reasoning_matches = re.findall(
        r'REASONING:\s*(.+?)(?=REASONING:|KEY_INSIGHTS:|ANSWER:|$)',
        response, re.IGNORECASE | re.DOTALL
    )
    reasoning_rounds = [r.strip() for r in reasoning_matches if r.strip()]

    # Extract answer
    answer = ""
    answer_matches = re.findall(
        r'ANSWER:\s*(.+?)(?:\n\n|\n(?=[A-Z][A-Z_]*:)|$)',
        response, re.IGNORECASE | re.DOTALL
    )
    if answer_matches:
        answer = answer_matches[-1].strip()
        if '\n' in answer:
            first_line = answer.split('\n')[0].strip()
            if len(first_line) < 100:
                answer = first_line

    # Extract key insights
    key_insights = []
    insights_match = re.search(
        r'KEY_INSIGHTS:\s*(.+?)(?=ANSWER:|$)',
        response, re.IGNORECASE | re.DOTALL
    )
    if insights_match:
        key_insights = [
            line.strip().lstrip('- ').lstrip('* ').lstrip('• ')
            for line in insights_match.group(1).split('\n')
            if line.strip() and line.strip()[0] in ('-', '*', '•')
        ]

    return ExtractedResponse(reasoning_rounds=reasoning_rounds, key_insights=key_insights, answer=answer)


# ============================================================
# Unified Evaluation Result
# ============================================================

@dataclass
class UnifiedEvalResult:
    task_type: str
    question_id: str

    # Scores (0-1)
    answer_score: float
    tool_score: float
    reasoning_score: float
    tool_reasoning_score: float
    state_score: float
    total_score: float

    # Detailed results
    answer_details: Dict[str, Any]
    tool_details: Dict[str, Any]
    reasoning_details: Dict[str, Any]
    state_details: Dict[str, Any]

    # Agent-as-Judge
    agent_judge_triggered: bool
    agent_judge_result: Optional[Dict]

    # Metadata
    eval_weights: Dict[str, float]


# ============================================================
# Unified Evaluator
# ============================================================

class UnifiedEvaluator:
    """
    Unified evaluator.

    Flow:
    1. Answer evaluation (objective or subjective)
    2. Tool evaluation (F1 + params)
    3. Structured reasoning evaluation (tool-aligned)
    4. State evaluation
    5. Combined score = weighted(answer, tool_reasoning, state)
    6. Agent-as-Judge for scores in [0.4, 0.7]
    """

    def __init__(self,
                 agent_judge_enabled: bool = True,
                 low_threshold: float = AGENT_JUDGE_LOW_THRESHOLD,
                 high_threshold: float = AGENT_JUDGE_HIGH_THRESHOLD):
        self.objective_eval = ObjectiveEvaluator()
        self.tool_eval = ToolEvaluator()
        self.state_eval = StateEvaluator()
        self.subjective_eval = SubjectiveEvaluator()

        self.agent_judge_enabled = agent_judge_enabled
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def evaluate_task(self, task_config: Dict, agent_result: Dict, question_idx: int = 0) -> UnifiedEvalResult:
        """Evaluate single-question task."""
        question = task_config["questions"][0]
        ground_truth = question["ground_truth"]
        task_type = ground_truth["task_type"]
        question_id = question.get("qid", "q_0")

        if "clear" in task_type:
            return self._evaluate_objective(task_type, question_id, ground_truth, agent_result, question)
        else:
            return self._evaluate_subjective(task_type, question_id, ground_truth, agent_result, question)

    # ================================================================
    # Objective Evaluation
    # ================================================================

    def _evaluate_objective(self, task_type, question_id, ground_truth, agent_result, question):
        # 1. Answer
        answer_dict = ground_truth["answer"]
        raw_answer = agent_result.get("answer", "")
        extracted = extract_structured_response(raw_answer)
        agent_answer = extracted.answer if extracted.answer else raw_answer
        answer_result = self.objective_eval.evaluate(agent_answer, answer_dict)
        answer_score = answer_result.score

        # 2. Tool calls
        tool_score, tool_result = self._eval_tools(ground_truth, agent_result)

        # 3. Structured reasoning
        reasoning_score, reasoning_details = self._eval_structured_reasoning(ground_truth, agent_result)

        # 4. State
        state_score, state_result = self._eval_state(ground_truth, agent_result)

        # 5. Combined score
        tr_split = TOOL_REASONING_SPLIT
        tool_reasoning_score = tr_split["tool"] * tool_score + tr_split["reasoning"] * reasoning_score

        weights = SCORE_WEIGHTS["objective"]
        total_score = (weights["answer"] * answer_score
                       + weights["tool_reasoning"] * tool_reasoning_score
                       + weights["state"] * state_score)

        # 6. Agent-as-Judge
        agent_judge_triggered = False
        agent_judge_result = None
        if self.agent_judge_enabled:
            total_score, agent_judge_triggered, agent_judge_result = self._agent_judge_review(
                total_score, task_type, question, agent_answer,
                ground_truth, answer_score, tool_reasoning_score, state_score,
                tool_result, state_result
            )

        return UnifiedEvalResult(
            task_type=task_type, question_id=question_id,
            answer_score=answer_score, tool_score=tool_score,
            reasoning_score=reasoning_score, tool_reasoning_score=tool_reasoning_score,
            state_score=state_score, total_score=total_score,
            answer_details=asdict(answer_result),
            tool_details=asdict(tool_result),
            reasoning_details=reasoning_details,
            state_details=asdict(state_result),
            agent_judge_triggered=agent_judge_triggered,
            agent_judge_result=agent_judge_result,
            eval_weights=SCORE_WEIGHTS["objective"]
        )

    # ================================================================
    # Subjective Evaluation
    # ================================================================

    def _evaluate_subjective(self, task_type, question_id, ground_truth, agent_result, question):
        # 1. Extract agent output
        raw_answer = agent_result.get("answer", "")
        extracted = extract_structured_response(raw_answer)
        agent_insights = agent_result.get("key_insights", []) or extracted.key_insights

        # 2. Build structured reasoning steps
        agent_steps = self._build_structured_steps(agent_result)
        gt_steps = ground_truth.get("reasoning", [])

        # 3. GT insights
        gt_answer = ground_truth.get("answer", {})
        gt_value = gt_answer.get("value", "")
        gt_insights = [gt_value] if isinstance(gt_value, str) else gt_value

        # 4. Subjective eval (insights + reasoning)
        subjective_result = self.subjective_eval.evaluate(
            agent_insights=agent_insights,
            agent_reasoning_steps=agent_steps,
            gt_insights=gt_insights,
            gt_reasoning_steps=gt_steps,
            is_fully_subjective=True
        )
        answer_score = subjective_result.insight_score
        reasoning_score = subjective_result.reasoning_score

        # 5. Tool calls
        tool_score, tool_result = self._eval_tools(ground_truth, agent_result)

        # 6. State
        state_score, state_result = self._eval_state(ground_truth, agent_result)

        # 7. Combined score
        tr_split = TOOL_REASONING_SPLIT
        tool_reasoning_score = tr_split["tool"] * tool_score + tr_split["reasoning"] * reasoning_score

        weights = SCORE_WEIGHTS["subjective"]
        total_score = (weights["answer"] * answer_score
                       + weights["tool_reasoning"] * tool_reasoning_score
                       + weights["state"] * state_score)

        # 8. Agent-as-Judge
        agent_judge_triggered = False
        agent_judge_result = None
        if self.agent_judge_enabled:
            total_score, agent_judge_triggered, agent_judge_result = self._agent_judge_review(
                total_score, task_type, question, raw_answer[:500],
                ground_truth, answer_score, tool_reasoning_score, state_score,
                tool_result, state_result
            )

        return UnifiedEvalResult(
            task_type=task_type, question_id=question_id,
            answer_score=answer_score, tool_score=tool_score,
            reasoning_score=reasoning_score, tool_reasoning_score=tool_reasoning_score,
            state_score=state_score, total_score=total_score,
            answer_details={
                "insight_score": subjective_result.insight_score,
                "reasoning_score": subjective_result.reasoning_score,
                "total_score": subjective_result.total_score,
                "insight_details": asdict(subjective_result.insight_details),
            },
            tool_details=asdict(tool_result),
            reasoning_details=asdict(subjective_result.reasoning_details),
            state_details=asdict(state_result),
            agent_judge_triggered=agent_judge_triggered,
            agent_judge_result=agent_judge_result,
            eval_weights=SCORE_WEIGHTS["subjective"]
        )

    # ================================================================
    # Shared Helpers
    # ================================================================

    def _build_structured_steps(self, agent_result: Dict) -> List[Dict]:
        """
        Pair reasoning_rounds with tool_calls to create structured steps.
        Each step: {iteration, tool, reasoning}
        """
        reasoning_rounds = agent_result.get("reasoning_rounds", [])
        tool_calls = agent_result.get("tool_calls", [])

        steps = []
        for i, rr in enumerate(reasoning_rounds):
            tool_name = "N/A"
            if i < len(tool_calls):
                tc = tool_calls[i]
                tool_name = tc.get("tool_name", tc.get("name", "N/A"))
            steps.append({
                "iteration": rr.get("iteration", i + 1),
                "tool": tool_name,
                "reasoning": rr.get("reasoning", "")
            })
        return steps

    def _eval_tools(self, ground_truth: Dict, agent_result: Dict) -> Tuple[float, ToolEvalResult]:
        """Extract and evaluate tool calls."""
        tool_eval_config = ground_truth.get("tool_eval", {})
        called_tools = agent_result.get("tool_calls", [])

        # Normalize tool call format
        normalized = []
        for t in called_tools:
            if "tool_execution" in t:
                te = t["tool_execution"]
                normalized.append({"name": te.get("tool_name", ""), "params": te.get("parameters", {})})
            else:
                normalized.append({
                    "name": t.get("name", t.get("tool_name", "")),
                    "params": t.get("params", t.get("parameters", {}))
                })

        tools_list = tool_eval_config.get("tools", [])
        result = self.tool_eval.evaluate_sequence(normalized, tools_list)
        return result.total_score, result

    def _eval_state(self, ground_truth: Dict, agent_result: Dict) -> Tuple[float, StateEvalResult]:
        """Extract and evaluate visualization state."""
        state_eval_config = ground_truth.get("state_eval")
        final_spec = agent_result.get("final_spec", {})

        spec_type = "vega-lite"
        schema = final_spec.get("$schema", "")
        if schema and "vega-lite" not in schema:
            spec_type = "vega"

        result = self.state_eval.evaluate(final_spec, state_eval_config, spec_type)
        return result.score, result

    def _eval_structured_reasoning(self, ground_truth: Dict, agent_result: Dict) -> Tuple[float, Dict]:
        """
        Evaluate reasoning with structure (each step paired with its tool).
        Reuses SubjectiveEvaluator for LLM-based evaluation.
        """
        agent_steps = self._build_structured_steps(agent_result)
        gt_steps = ground_truth.get("reasoning", [])

        if not agent_steps:
            return 0.0, {"score": 0.0, "note": "no reasoning found"}

        if not gt_steps:
            return 1.0, {"score": 1.0, "note": "no GT reasoning available"}

        result = self.subjective_eval._eval_reasoning(
            agent_steps=agent_steps,
            gt_steps=gt_steps,
            is_fully_subjective=False
        )
        return result.average_score, asdict(result)

    # ================================================================
    # Agent-as-Judge
    # ================================================================

    def _agent_judge_review(self, total_score, task_type, question, agent_answer,
                            ground_truth, answer_score, tool_reasoning_score, state_score,
                            tool_result, state_result) -> Tuple[float, bool, Optional[Dict]]:
        """
        Conditionally trigger LLM review for ambiguous scores.
        score < 0.4 -> auto-fail, score > 0.7 -> auto-pass
        """
        if total_score < self.low_threshold or total_score > self.high_threshold:
            return total_score, False, None

        # Build context for judge
        gt_answer = ground_truth.get("answer", {})
        gt_answer_str = gt_answer.get("value", str(gt_answer)) if isinstance(gt_answer, dict) else str(gt_answer)
        gt_tools = ground_truth.get("tool_eval", {}).get("tools", [])
        gt_tools_str = ", ".join(t["tool"] for t in gt_tools) if gt_tools else "N/A"
        question_text = question.get("question", "N/A")

        score_details = json.dumps({
            "answer": answer_score,
            "tool_reasoning": tool_reasoning_score,
            "tool_f1": tool_result.tool_match_score,
            "tool_params": tool_result.param_score,
            "state": state_score,
            "state_passed": state_result.passed,
        }, indent=2, ensure_ascii=False)

        prompt = AGENT_JUDGE_PROMPT.format(
            task_type=task_type,
            question_text=str(question_text)[:300],
            answer_score=answer_score,
            tool_reasoning_score=tool_reasoning_score,
            state_score=state_score,
            combined_score=total_score,
            agent_answer=str(agent_answer)[:500],
            gt_answer=str(gt_answer_str)[:300],
            gt_tools=gt_tools_str,
            score_details=score_details,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
        )

        judge_result = self.subjective_eval._call_llm(prompt)

        verdict = str(judge_result.get("verdict", "BORDERLINE")).upper()
        adjusted = max(0.0, min(1.0, float(judge_result.get("adjusted_score", total_score))))

        if verdict == "PASS":
            final_score = max(total_score, adjusted)
        elif verdict == "FAIL":
            final_score = min(total_score, adjusted)
        else:
            final_score = total_score

        judge_result["original_score"] = total_score
        judge_result["final_score"] = final_score

        return final_score, True, judge_result

    # ================================================================
    # Batch & File Helpers
    # ================================================================

    def evaluate_batch(self, task_config: Dict, agent_results: List[Dict]) -> List[UnifiedEvalResult]:
        """Evaluate multiple agent results for a task."""
        return [self.evaluate_task(task_config, result) for result in agent_results]


def load_task_config(task_path: str) -> Dict:
    with open(task_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_from_files(task_path: str, result_path: str,
                        agent_judge_enabled: bool = True) -> List[UnifiedEvalResult]:
    """
    Evaluate from task config and result files.
    result.json has {"results": [{...}, ...]} — evaluates each result item.
    """
    task_config = load_task_config(task_path)
    with open(result_path, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    agent_results = result_data["results"]
    evaluator = UnifiedEvaluator(agent_judge_enabled=agent_judge_enabled)
    return [evaluator.evaluate_task(task_config, r) for r in agent_results]