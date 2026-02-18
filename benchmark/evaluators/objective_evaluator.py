"""
Objective Answer Evaluator

Evaluates objective answers with 3 types:
- numeric: tolerance-based matching
- categorical: exact match + LLM-as-judge semantic matching
- boolean: normalized string match
"""

import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class EvalResult:
    """Evaluation result container"""
    score: float       # 0.0 to 1.0
    matched: bool
    details: Dict[str, Any]


class ObjectiveEvaluator:
    """Evaluator for objective (clear-answer) questions using LLM-as-judge"""

    def __init__(
        self,
        api_key_env: str = "OPENROUTER_API_KEY",
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "openai/gpt-5.2"
    ):
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy init OpenAI client — only created when LLM judge is actually needed."""
        if self._client is None:
            from openai import OpenAI
            api_key = os.getenv(self.api_key_env)
            if not api_key:
                raise RuntimeError(f"{self.api_key_env} environment variable not set")
            self._client = OpenAI(api_key=api_key, base_url=self.base_url)
        return self._client

    def evaluate(self, predicted: Any, ground_truth: Dict) -> EvalResult:
        answer_type = ground_truth.get("type", "categorical")

        if answer_type == "numeric":
            return self._eval_numeric(predicted, ground_truth)
        elif answer_type == "categorical":
            return self._eval_categorical(predicted, ground_truth)
        elif answer_type == "boolean":
            return self._eval_boolean(predicted, ground_truth)
        else:
            return self._eval_categorical(predicted, ground_truth)

    # ==================== Numeric ====================

    def _eval_numeric(self, predicted: Any, gt: Dict) -> EvalResult:
        target = float(gt["value"])
        tolerance = gt.get("tolerance", 0.05)
        pred_val = float(predicted)

        diff = abs(pred_val - target)
        score = 1.0 if diff <= tolerance else 0.0

        return EvalResult(
            score=score, matched=(score == 1.0),
            details={"predicted": pred_val, "target": target, "tolerance": tolerance, "difference": diff}
        )

    # ==================== Categorical ====================

    def _eval_categorical(self, predicted: Any, gt: Dict) -> EvalResult:
        target = gt["value"]
        alternatives = gt.get("alternatives", [])
        valid_answers = [str(target)] + [str(alt) for alt in alternatives]

        # Fast-path: exact match (skip LLM call)
        pred_norm = str(predicted).lower().strip()
        for valid in valid_answers:
            if pred_norm == valid.lower().strip():
                return EvalResult(
                    score=1.0, matched=True,
                    details={"predicted": predicted, "target": target, "judgment": "CORRECT", "method": "exact_match"}
                )

        # LLM judge for semantic matching
        valid_answers_str = "\n".join(f"  - {ans}" for ans in valid_answers)
        judge_prompt = f"""You are evaluating whether a predicted answer matches the expected answer(s) for a categorical question.

Expected answer(s):
{valid_answers_str}

Predicted answer: {predicted}

Task: Determine if the predicted answer is semantically equivalent to any of the expected answers. Consider:
- Exact matches (ignoring case and whitespace)
- Paraphrases with the same meaning (e.g., "sub-clusters" vs "show subclusters")
- Different wordings that convey the same categorical choice

Respond with ONLY "CORRECT" or "INCORRECT" on a single line."""

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=10
        )

        judgment = response.choices[0].message.content.strip().upper()
        is_correct = judgment == "CORRECT"
        score = 1.0 if is_correct else 0.0

        return EvalResult(
            score=score, matched=is_correct,
            details={"predicted": predicted, "target": target, "judgment": judgment, "method": "llm_judge"}
        )

    # ==================== Boolean ====================

    def _eval_boolean(self, predicted: Any, gt: Dict) -> EvalResult:
        target = gt["value"]

        pred_bool = self._to_boolean(predicted)
        target_bool = self._to_boolean(target)

        score = 1.0 if pred_bool == target_bool else 0.0
        return EvalResult(
            score=score, matched=(score == 1.0),
            details={"predicted": pred_bool, "target": target_bool}
        )

    @staticmethod
    def _to_boolean(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        s = str(value).strip().rstrip('.!').lower()
        if s in ('true', 'yes'):
            return True
        if s in ('false', 'no'):
            return False
        if s.startswith('yes'):
            return True
        if s.startswith('no'):
            return False
        return bool(value)