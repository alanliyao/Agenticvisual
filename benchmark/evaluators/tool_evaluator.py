"""
Tool Call Evaluator

Evaluates tool calls with:
- Tool match: whether the correct tool was called
- Parameter evaluation: whether parameters are correct (supports partial credit)

Parameter types (6 types):
- region: IoU-based matching
- categorical: exact match with alternatives
- numeric: tolerance-based matching  
- list: Jaccard similarity for set matching
- date: date parsing with tolerance (treated as 1D region)
- order: position similarity for ordered sequences
"""

from typing import Dict, Any, List, Union, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import re


@dataclass
class ToolEvalResult:
    """Tool evaluation result"""
    tool_match_score: float  # 0.0 or 1.0
    param_score: float  # 0.0 to 1.0 (supports partial credit)
    total_score: float  # weighted average
    details: Dict[str, Any]


class ToolEvaluator:
    """Evaluator for tool calls"""
    
    def __init__(self, tool_weight: float = 0.5, param_weight: float = 0.5):
        """
        Initialize with weights for tool match and parameter evaluation.
        
        Args:
            tool_weight: Weight for tool match score (default 0.5)
            param_weight: Weight for parameter score (default 0.5)
        """
        self.tool_weight = tool_weight
        self.param_weight = param_weight

    def evaluate_sequence(
        self,
        called_tools: List[Dict],
        tools_config: List[Dict]
    ) -> ToolEvalResult:
        """
        Evaluate tool calls against expected tools (match by name, like original evaluate).
        tools_config: [ { "tool": str, "param_eval": dict }, ... ]
        Storage format only; matching is by tool name, not order.
        """
        expected = tools_config or []
        called = called_tools or []
        norm_called = [
            {"name": t.get("name", t.get("tool_name", "")), "params": t.get("params", t.get("parameters", {}))}
            for t in called
        ]
        if not expected:
            tool_match_score = 1.0 if not norm_called else 0.0
            param_score = 1.0
            return ToolEvalResult(
                tool_match_score=tool_match_score,
                param_score=param_score,
                total_score=(self.tool_weight * tool_match_score + self.param_weight * param_score),
                details={"expected_tools": [], "called_tools": [t["name"] for t in norm_called], "tool_matched": bool(tool_match_score), "param_eval": {}}
            )
        param_scores = []
        param_details_list = []
        used = set()
        matched = []
        for exp in expected:
            exp_tool = exp.get("tool", "")
            pe = exp.get("param_eval")
            match = None
            for j, c in enumerate(norm_called):
                if j not in used and c["name"] == exp_tool:
                    match = c
                    used.add(j)
                    break
            if match is None:
                matched.append(False)
                param_scores.append(0.0)
                param_details_list.append({"note": "tool not called"})
                continue
            matched.append(True)
            if pe:
                sc, det = self._eval_params(match["params"], pe)
                param_scores.append(sc)
                param_details_list.append(det)
            else:
                param_scores.append(1.0)
                param_details_list.append({"note": "no param eval"})
        all_matched = all(matched)
        tool_match_score = 1.0 if all_matched else 0.0
        param_score = sum(param_scores) / len(param_scores) if param_scores else 0.0
        total = self.tool_weight * tool_match_score + self.param_weight * param_score
        return ToolEvalResult(
            tool_match_score=tool_match_score,
            param_score=param_score,
            total_score=total,
            details={
                "expected_tools": [e.get("tool", "") for e in expected],
                "called_tools": [t["name"] for t in norm_called],
                "tool_matched": bool(all_matched),
                "param_eval": param_details_list
            }
        )

    def _eval_params(self, called_params: Dict, param_eval: Dict) -> Tuple[float, Dict]:
        """
        Evaluate parameters based on param_eval config.
        
        Returns:
            (score, details) tuple
        """
        eval_type = param_eval.get("type", "categorical")
        
        if eval_type == "region":
            return self._eval_region_param(called_params, param_eval)
        elif eval_type == "categorical" or eval_type == "field":
            return self._eval_categorical_param(called_params, param_eval)
        elif eval_type == "numeric":
            return self._eval_numeric_param(called_params, param_eval)
        elif eval_type == "list":
            return self._eval_list_param(called_params, param_eval)
        elif eval_type == "date":
            return self._eval_date_param(called_params, param_eval)
        elif eval_type == "order":
            return self._eval_order_param(called_params, param_eval)
        elif eval_type == "selection":
            return self._eval_selection_param(called_params, param_eval)
        else:
            return self._eval_categorical_param(called_params, param_eval)
    
    def _eval_region_param(self, called_params: Dict, param_eval: Dict) -> Tuple[float, Dict]:
        """Evaluate region parameters using IoU"""
        target = param_eval.get("target", {})
        threshold = param_eval.get("threshold", 0.5)
        metric = param_eval.get("metric", "iou")
        
        # Extract called region
        x_range = called_params.get("x_range", called_params.get("x", [0, 0]))
        y_range = called_params.get("y_range", called_params.get("y", [0, 0]))
        called_region = {"x": x_range, "y": y_range}
        
        if metric == "iou":
            score = self._compute_iou(called_region, target)
        elif metric == "containment":
            score = self._compute_containment(called_region, target)
        else:
            score = self._compute_iou(called_region, target)
        
        return score, {
            "type": "region",
            "called": called_region,
            "target": target,
            "metric": metric,
            f"{metric}_score": score,
            "threshold": threshold,
            "passed": score >= threshold
        }
    
    def _eval_categorical_param(self, called_params: Dict, param_eval: Dict) -> Tuple[float, Dict]:
        """Evaluate categorical/field parameters"""
        param_name = param_eval.get("param", param_eval.get("param_field", None))
        target = param_eval.get("target")
        alternatives = param_eval.get("alternatives", [])
        
        # Get called value
        if param_name:
            called_value = called_params.get(param_name, "")
        else:
            # Try common parameter names
            for key in ["field", "field_name", "category", "value", "method"]:
                if key in called_params:
                    called_value = called_params[key]
                    break
            else:
                called_value = ""
        
        # Normalize and compare
        called_norm = str(called_value).lower().strip()
        valid_answers = [str(target).lower().strip()]
        valid_answers.extend([str(alt).lower().strip() for alt in alternatives])
        
        matched = called_norm in valid_answers
        score = 1.0 if matched else 0.0
        
        return score, {
            "type": "categorical",
            "param": param_name,
            "called": called_value,
            "target": target,
            "alternatives": alternatives,
            "matched": matched
        }
    
    def _eval_numeric_param(self, called_params: Dict, param_eval: Dict) -> Tuple[float, Dict]:
        """Evaluate numeric parameters with tolerance"""
        param_name = param_eval.get("param", "")
        target = param_eval.get("target")
        tolerance = param_eval.get("tolerance", 0.05)
        
        called_value = called_params.get(param_name, 0)
        
        try:
            called_val = float(called_value)
            target_val = float(target)
        except (ValueError, TypeError):
            return 0.0, {"type": "numeric", "error": "Cannot convert to numeric"}
        
        # Handle tolerance list (e.g., [2, 3, 4] means any of these is acceptable)
        if isinstance(tolerance, list):
            if called_val in [float(t) for t in tolerance] or called_val == target_val:
                return 1.0, {
                    "type": "numeric",
                    "called": called_val,
                    "target": target_val,
                    "tolerance_list": tolerance,
                    "matched": True
                }
            else:
                return 0.0, {
                    "type": "numeric",
                    "called": called_val,
                    "target": target_val,
                    "tolerance_list": tolerance,
                    "matched": False
                }
        
        # Numeric tolerance
        if tolerance < 1:
            # Relative tolerance
            if target_val != 0:
                diff_ratio = abs(called_val - target_val) / abs(target_val)
            else:
                diff_ratio = abs(called_val - target_val)
            score = max(0.0, 1.0 - diff_ratio / tolerance) if diff_ratio > tolerance else 1.0
        else:
            # Absolute tolerance
            diff = abs(called_val - target_val)
            score = max(0.0, 1.0 - diff / tolerance) if diff > tolerance else 1.0
        
        return score, {
            "type": "numeric",
            "param": param_name,
            "called": called_val,
            "target": target_val,
            "tolerance": tolerance,
            "difference": abs(called_val - target_val),
            "score": score
        }
    
    def _eval_list_param(self, called_params: Dict, param_eval: Dict) -> Tuple[float, Dict]:
        """Evaluate list parameters using Jaccard similarity"""
        param_name = param_eval.get("param", "")
        target = param_eval.get("target", [])
        
        called_value = called_params.get(param_name, [])
        
        if not isinstance(called_value, list):
            called_value = [called_value]
        if not isinstance(target, list):
            target = [target]
        
        # Normalize to sets
        called_set = set(str(v).lower().strip() for v in called_value)
        target_set = set(str(v).lower().strip() for v in target)
        
        # Compute Jaccard similarity
        intersection = len(called_set & target_set)
        union = len(called_set | target_set)
        
        if union == 0:
            jaccard = 1.0  # Both empty
        else:
            jaccard = intersection / union
        
        return jaccard, {
            "type": "list",
            "param": param_name,
            "called": list(called_set),
            "target": list(target_set),
            "intersection": intersection,
            "union": union,
            "jaccard": jaccard
        }
    
    def _eval_date_param(self, called_params: Dict, param_eval: Dict) -> Tuple[float, Dict]:
        """Evaluate date parameters (treated as 1D region)"""
        target = param_eval.get("target", {})
        tolerance_days = param_eval.get("tolerance", 7)  # Default 7 days tolerance
        
        # Get called date range
        start = called_params.get("start", called_params.get("start_date", ""))
        end = called_params.get("end", called_params.get("end_date", ""))
        
        target_start = target.get("start", "")
        target_end = target.get("end", "")
        
        try:
            called_start = self._parse_date(start)
            called_end = self._parse_date(end)
            target_start_dt = self._parse_date(target_start)
            target_end_dt = self._parse_date(target_end)
        except:
            return 0.0, {"type": "date", "error": "Cannot parse dates"}
        
        # Compute overlap as 1D IoU
        overlap_start = max(called_start, target_start_dt)
        overlap_end = min(called_end, target_end_dt)
        
        if overlap_end <= overlap_start:
            iou = 0.0
        else:
            overlap = (overlap_end - overlap_start).days
            total = (max(called_end, target_end_dt) - min(called_start, target_start_dt)).days
            iou = overlap / total if total > 0 else 0.0
        
        return iou, {
            "type": "date",
            "called": {"start": str(start), "end": str(end)},
            "target": {"start": str(target_start), "end": str(target_end)},
            "iou": iou
        }
    
    def _eval_order_param(self, called_params: Dict, param_eval: Dict) -> Tuple[float, Dict]:
        """Evaluate order/sequence parameters"""
        param_name = param_eval.get("param", "order")
        target = param_eval.get("target", [])
        
        called_value = called_params.get(param_name, [])
        
        if not isinstance(called_value, list) or not isinstance(target, list):
            return 0.0, {"type": "order", "error": "Order must be a list"}
        
        # Normalize
        called_norm = [str(v).lower().strip() for v in called_value]
        target_norm = [str(v).lower().strip() for v in target]
        
        # Compute position similarity (simple approach)
        if len(called_norm) == 0 or len(target_norm) == 0:
            return 0.0, {"type": "order", "called": called_norm, "target": target_norm}
        
        # Count items in correct position
        max_len = max(len(called_norm), len(target_norm))
        correct = 0
        for i in range(min(len(called_norm), len(target_norm))):
            if called_norm[i] == target_norm[i]:
                correct += 1
        
        score = correct / max_len
        
        return score, {
            "type": "order",
            "param": param_name,
            "called": called_norm,
            "target": target_norm,
            "correct_positions": correct,
            "score": score
        }
    
    def _eval_selection_param(self, called_params: Dict, param_eval: Dict) -> Tuple[float, Dict]:
        """Evaluate selection parameters (field + operator + value)"""
        target_field = param_eval.get("param_field", "")
        target_op = param_eval.get("param_op", "==")
        target_value = param_eval.get("target", "")
        alternatives = param_eval.get("alternatives", [])
        
        called_field = called_params.get("field", "")
        called_op = called_params.get("op", "==")
        called_value = called_params.get("value", called_params.get("values", ""))
        
        # Check each component
        field_match = str(called_field).lower() == str(target_field).lower()
        op_match = str(called_op) == str(target_op)
        
        # Value matching with alternatives
        called_val_norm = str(called_value).lower().strip()
        valid_values = [str(target_value).lower().strip()]
        valid_values.extend([str(alt).lower().strip() for alt in alternatives])
        value_match = called_val_norm in valid_values
        
        # Average of three components
        score = (field_match + op_match + value_match) / 3
        
        return score, {
            "type": "selection",
            "called": {"field": called_field, "op": called_op, "value": called_value},
            "target": {"field": target_field, "op": target_op, "value": target_value},
            "field_match": field_match,
            "op_match": op_match,
            "value_match": value_match,
            "score": score
        }
    
    def _compute_iou(self, region1: Dict, region2: Dict) -> float:
        """Compute 2D IoU"""
        x1 = region1.get("x", [0, 0])
        y1 = region1.get("y", [0, 0])
        x2 = region2.get("x", [0, 0])
        y2 = region2.get("y", [0, 0])
        
        x_inter = max(0, min(x1[1], x2[1]) - max(x1[0], x2[0]))
        y_inter = max(0, min(y1[1], y2[1]) - max(y1[0], y2[0]))
        intersection = x_inter * y_inter
        
        area1 = (x1[1] - x1[0]) * (y1[1] - y1[0])
        area2 = (x2[1] - x2[0]) * (y2[1] - y2[0])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_containment(self, region1: Dict, region2: Dict) -> float:
        """Compute containment of region2 in region1"""
        x1 = region1.get("x", [0, 0])
        y1 = region1.get("y", [0, 0])
        x2 = region2.get("x", [0, 0])
        y2 = region2.get("y", [0, 0])
        
        x_inter = max(0, min(x1[1], x2[1]) - max(x1[0], x2[0]))
        y_inter = max(0, min(y1[1], y2[1]) - max(y1[0], y2[0]))
        intersection = x_inter * y_inter
        
        area2 = (x2[1] - x2[0]) * (y2[1] - y2[0])
        
        return intersection / area2 if area2 > 0 else 0.0
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime"""
        formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%Y-%m-%dT%H:%M:%S",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(str(date_str), fmt)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse date: {date_str}")
