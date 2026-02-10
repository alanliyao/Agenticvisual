"""
Objective Answer Evaluator

Evaluates objective answers with 4 types:
- numeric: tolerance-based matching
- categorical: exact match with alternatives + BERTScore semantic matching
- boolean: exact match
- region: IoU-based matching
"""

from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass
import re


@dataclass
class EvalResult:
    """Evaluation result container"""
    score: float  # 0.0 to 1.0
    matched: bool
    details: Dict[str, Any]


class ObjectiveEvaluator:
    """Evaluator for objective (clear-answer) questions"""
    
    def __init__(
        self,
        bertscore_threshold: float = 0.80
    ):
        """
        Initialize the evaluator.
        
        Args:
            bertscore_threshold: Minimum BERTScore (0-1) to consider a semantic match for categorical answers
        """
        self.bertscore_threshold = bertscore_threshold
        self._bertscore_available = False
        self._bertscore_score = None
        try:
            from bert_score import score as bertscore_score
            self._bertscore_score = bertscore_score
            self._bertscore_available = True
        except ImportError:
            # BERTScore not available - will raise error when needed (hard fail)
            pass
    
    def evaluate(self, predicted: Any, ground_truth: Dict) -> EvalResult:
        """
        Evaluate predicted answer against ground truth.
        
        Args:
            predicted: The predicted answer value
            ground_truth: Dict containing 'type', 'value', and optionally 'alternatives', 'tolerance'
            
        Returns:
            EvalResult with score, matched flag, and details
        """
        answer_type = ground_truth.get("type", "categorical")
        
        if answer_type == "numeric":
            return self._eval_numeric(predicted, ground_truth)
        elif answer_type == "categorical":
            return self._eval_categorical(predicted, ground_truth)
        elif answer_type == "boolean":
            return self._eval_boolean(predicted, ground_truth)
        elif answer_type == "region":
            return self._eval_region(predicted, ground_truth)
        elif answer_type == "open_ended":
            # open_ended should use subjective evaluation (vague task_type)
            # Return 0 score with warning
            return EvalResult(
                score=0.0,
                matched=False,
                details={
                    "predicted": predicted,
                    "target": ground_truth.get("value"),
                    "warning": "open_ended answer type should use vague task_type for LLM judge evaluation"
                }
            )
        else:
            # Default to categorical
            return self._eval_categorical(predicted, ground_truth)
    
    def _eval_numeric(self, predicted: Any, gt: Dict) -> EvalResult:
        """
        Evaluate numeric answer with tolerance.
        
        Score = 1.0 if within tolerance, else max(0, 1 - diff/tolerance)
        """
        target = gt.get("value")
        tolerance = gt.get("tolerance", 0.05)  # Default 5% tolerance
        
        try:
            pred_val = float(predicted)
            target_val = float(target)
        except (ValueError, TypeError):
            return EvalResult(
                score=0.0,
                matched=False,
                details={"error": "Cannot convert to numeric", "predicted": predicted, "target": target}
            )
        
        # Handle percentage tolerance (e.g., 0.05 means 5%)
        if tolerance < 1:
            # Relative tolerance
            if target_val != 0:
                diff_ratio = abs(pred_val - target_val) / abs(target_val)
            else:
                diff_ratio = abs(pred_val - target_val)
            
            if diff_ratio <= tolerance:
                score = 1.0
                matched = True
            else:
                score = max(0.0, 1.0 - diff_ratio / tolerance)
                matched = False
        else:
            # Absolute tolerance
            diff = abs(pred_val - target_val)
            if diff <= tolerance:
                score = 1.0
                matched = True
            else:
                score = max(0.0, 1.0 - diff / tolerance)
                matched = False
        
        return EvalResult(
            score=score,
            matched=matched,
            details={
                "predicted": pred_val,
                "target": target_val,
                "tolerance": tolerance,
                "difference": abs(pred_val - target_val)
            }
        )
    
    def _eval_categorical(self, predicted: Any, gt: Dict) -> EvalResult:
        """
        Evaluate categorical answer with alternatives and BERTScore semantic matching.
        
        Matching strategy (in order):
        1. Exact match after normalization
        2. BERTScore semantic matching (handles paraphrases like "sub-clusters" vs "show subclusters")
        
        Score = 1.0 if matches value or alternatives, else BERTScore if above threshold, else 0.0
        """
        target = gt.get("value")
        alternatives = gt.get("alternatives", [])
        
        # Normalize for comparison
        pred_normalized = self._normalize_string(str(predicted))
        valid_answers = [self._normalize_string(str(target))]
        valid_answers.extend([self._normalize_string(str(alt)) for alt in alternatives])
        
        # Step 1: Exact match after normalization
        if pred_normalized in valid_answers:
            return EvalResult(
                score=1.0,
                matched=True,
                details={
                    "predicted": predicted,
                    "target": target,
                    "alternatives": alternatives,
                    "normalized_predicted": pred_normalized,
                    "match_method": "exact"
                }
            )
        
        # Step 2: BERTScore semantic matching (if available)
        if self._bertscore_available:
            best_score = 0.0
            best_match = None
            
            try:
                # Compute BERTScore for predicted vs all valid answers
                for ans in valid_answers:
                    # BERTScore expects lists of strings
                    P, R, F1 = self._bertscore_score(
                        [str(predicted)],
                        [ans],
                        lang='en',
                        verbose=False
                    )
                    # Use F1 score (harmonic mean of precision and recall)
                    # Handle both torch.Tensor and numpy array
                    f1_val = F1[0]
                    if hasattr(f1_val, 'item'):
                        bertscore = float(f1_val.item())
                    else:
                        bertscore = float(f1_val)
                    
                    if bertscore > best_score:
                        best_score = bertscore
                        best_match = ans
                
                if best_score >= self.bertscore_threshold:
                    return EvalResult(
                        score=1.0,
                        matched=True,
                        details={
                            "predicted": predicted,
                            "target": target,
                            "alternatives": alternatives,
                            "normalized_predicted": pred_normalized,
                            "match_method": "bertscore",
                            "bertscore": best_score,
                            "bertscore_matched_with": best_match,
                            "bertscore_threshold": self.bertscore_threshold
                        }
                    )
                else:
                    # Return BERTScore even if not matched (for debugging)
                    return EvalResult(
                        score=0.0,
                        matched=False,
                        details={
                            "predicted": predicted,
                            "target": target,
                            "alternatives": alternatives,
                            "normalized_predicted": pred_normalized,
                            "match_method": "none",
                            "best_bertscore": best_score,
                            "best_bertscore_match": best_match,
                            "bertscore_threshold": self.bertscore_threshold
                        }
                    )
            except Exception as e:
                # Fallback to no match if BERTScore fails
                return EvalResult(
                    score=0.0,
                    matched=False,
                    details={
                        "predicted": predicted,
                        "target": target,
                        "alternatives": alternatives,
                        "normalized_predicted": pred_normalized,
                        "match_method": "none",
                        "error": f"BERTScore failed: {str(e)}"
                    }
                )
        else:
            # BERTScore not available - hard fail
            raise RuntimeError(
                "BERTScore is required for semantic matching but not available. "
                "Please install it: pip install bert-score torch transformers"
            )
    
    def _eval_boolean(self, predicted: Any, gt: Dict) -> EvalResult:
        """
        Evaluate boolean answer.
        
        Score = 1.0 if exact match, else 0.0
        """
        target = gt.get("value")
        
        # Convert to boolean
        pred_bool = self._to_boolean(predicted)
        target_bool = self._to_boolean(target)
        
        if pred_bool is None:
            return EvalResult(
                score=0.0,
                matched=False,
                details={"error": "Cannot convert predicted to boolean", "predicted": predicted}
            )
        
        matched = pred_bool == target_bool
        score = 1.0 if matched else 0.0
        
        return EvalResult(
            score=score,
            matched=matched,
            details={
                "predicted": pred_bool,
                "target": target_bool
            }
        )
    
    def _eval_region(self, predicted: Dict, gt: Dict) -> EvalResult:
        """
        Evaluate region answer using IoU.
        
        Score = IoU value (partial credit)
        Matched = IoU >= threshold
        """
        target = gt.get("value") or gt.get("target")
        threshold = gt.get("threshold", 0.5)
        metric = gt.get("metric", "iou")
        
        if not isinstance(predicted, dict) or not isinstance(target, dict):
            return EvalResult(
                score=0.0,
                matched=False,
                details={"error": "Region must be dict with x and y ranges"}
            )
        
        if metric == "iou":
            iou = self._compute_iou(predicted, target)
            score = iou
            matched = iou >= threshold
        elif metric == "containment":
            containment = self._compute_containment(predicted, target)
            score = containment
            matched = containment >= threshold
        else:
            iou = self._compute_iou(predicted, target)
            score = iou
            matched = iou >= threshold
        
        return EvalResult(
            score=score,
            matched=matched,
            details={
                "predicted": predicted,
                "target": target,
                "metric": metric,
                "threshold": threshold,
                f"{metric}_value": score
            }
        )
    
    def _compute_iou(self, region1: Dict, region2: Dict) -> float:
        """Compute Intersection over Union for 2D regions"""
        x1 = region1.get("x", [0, 0])
        y1 = region1.get("y", [0, 0])
        x2 = region2.get("x", [0, 0])
        y2 = region2.get("y", [0, 0])
        
        # Compute intersection
        x_inter = max(0, min(x1[1], x2[1]) - max(x1[0], x2[0]))
        y_inter = max(0, min(y1[1], y2[1]) - max(y1[0], y2[0]))
        intersection = x_inter * y_inter
        
        # Compute union
        area1 = (x1[1] - x1[0]) * (y1[1] - y1[0])
        area2 = (x2[1] - x2[0]) * (y2[1] - y2[0])
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _compute_containment(self, region1: Dict, region2: Dict) -> float:
        """Compute how much of region2 is contained in region1"""
        x1 = region1.get("x", [0, 0])
        y1 = region1.get("y", [0, 0])
        x2 = region2.get("x", [0, 0])
        y2 = region2.get("y", [0, 0])
        
        # Compute intersection
        x_inter = max(0, min(x1[1], x2[1]) - max(x1[0], x2[0]))
        y_inter = max(0, min(y1[1], y2[1]) - max(y1[0], y2[0]))
        intersection = x_inter * y_inter
        
        # Compute area of region2
        area2 = (x2[1] - x2[0]) * (y2[1] - y2[0])
        
        if area2 == 0:
            return 0.0
        
        return intersection / area2
    
    def _normalize_string(self, s: str) -> str:
        """Normalize string for comparison"""
        return s.lower().strip()
    
    def _to_boolean(self, value: Any) -> Optional[bool]:
        """Convert value to boolean"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lower = value.lower().strip()
            if lower in ('true', 'yes', '1', 'y'):
                return True
            elif lower in ('false', 'no', '0', 'n'):
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return None


# Convenience function
def evaluate_objective(predicted: Any, ground_truth: Dict) -> EvalResult:
    """Evaluate objective answer"""
    evaluator = ObjectiveEvaluator()
    return evaluator.evaluate(predicted, ground_truth)

