"""
Tool Call Evaluator

Evaluates tool calls with:
- Tool match: F1-based scoring (precision × recall) to penalize tool spamming
- Parameter evaluation: 8 types (region, categorical, numeric, list, date, order, selection, object)
"""

from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ToolEvalResult:
    """Tool evaluation result"""
    tool_match_score: float  # 0.0 to 1.0 (F1)
    param_score: float       # 0.0 to 1.0
    total_score: float       # weighted average
    details: Dict[str, Any]


class ToolEvaluator:

    def __init__(self, tool_weight: float = 0.5, param_weight: float = 0.5):
        self.tool_weight = tool_weight
        self.param_weight = param_weight

    def evaluate_sequence(self, called_tools: List[Dict], tools_config: List[Dict]) -> ToolEvalResult:
        """
        Evaluate tool calls against expected tools.
        Match by tool name (not order). Score with F1 to penalize redundant calls.
        """
        norm_called = [
            {"name": t.get("name", t.get("tool_name", "")),
             "params": t.get("params", t.get("parameters", {}))}
            for t in called_tools
        ]

        if not tools_config:
            return ToolEvalResult(
                tool_match_score=1.0, param_score=1.0, total_score=1.0,
                details={"expected_tools": [], "called_tools": [t["name"] for t in norm_called]}
            )

        # Greedy match: for each expected tool, find first unused called tool with same name
        used = set()
        matched_flags = []
        param_scores = []
        param_details = []

        for exp in tools_config:
            exp_tool = exp["tool"]
            pe = exp.get("param_eval")
            match = None
            for j, c in enumerate(norm_called):
                if j not in used and c["name"] == exp_tool:
                    match = c
                    used.add(j)
                    break

            if match is None:
                matched_flags.append(False)
                param_scores.append(0.0)
                param_details.append({"note": "tool not called"})
                continue

            matched_flags.append(True)
            if pe:
                sc, det = self._eval_params(match["params"], pe)
                param_scores.append(sc)
                param_details.append(det)
            else:
                param_scores.append(1.0)
                param_details.append({"note": "no param eval"})

        # F1-based tool match scoring
        num_matched = sum(matched_flags)
        num_expected = len(tools_config)
        num_called = len(norm_called)

        recall = num_matched / num_expected
        precision = num_matched / num_called if num_called > 0 else 1.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        param_score = sum(param_scores) / len(param_scores)
        total = self.tool_weight * f1 + self.param_weight * param_score

        return ToolEvalResult(
            tool_match_score=f1,
            param_score=param_score,
            total_score=total,
            details={
                "expected_tools": [e["tool"] for e in tools_config],
                "called_tools": [t["name"] for t in norm_called],
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "param_eval": param_details
            }
        )

    # ==================== Parameter Dispatch ====================

    def _eval_params(self, called_params: Dict, param_eval: Dict) -> Tuple[float, Dict]:
        eval_type = param_eval.get("type", "categorical")

        if eval_type == "region":
            return self._eval_region_param(called_params, param_eval)
        elif eval_type in ("categorical", "field"):
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
        elif eval_type == "object":
            return self._eval_object_param(called_params, param_eval)
        else:
            return self._eval_categorical_param(called_params, param_eval)

    # ==================== Object (NEW) ====================

    def _eval_object_param(self, called_params: Dict, param_eval: Dict) -> Tuple[float, Dict]:
        """
        Evaluate object parameters by matching target dict keys against called params.
        Supports 'all_params' to compare against the entire called_params dict.
        Handles both scalar and list values within the target.
        """
        target = param_eval.get("target", {})
        param_name = param_eval.get("param", "")

        source = called_params if param_name == "all_params" else called_params.get(param_name, {})

        if not target:
            return 1.0, {"type": "object", "note": "empty target"}

        key_scores = {}
        for key, expected_val in target.items():
            actual_val = source.get(key)

            if isinstance(expected_val, list) and isinstance(actual_val, list):
                # Set comparison (Jaccard) for list values
                expected_set = {str(v).lower() for v in expected_val}
                actual_set = {str(v).lower() for v in actual_val}
                intersection = len(expected_set & actual_set)
                union = len(expected_set | actual_set)
                key_score = intersection / union if union > 0 else 1.0
                key_scores[key] = {
                    "expected": expected_val, "actual": actual_val,
                    "jaccard": key_score
                }
            else:
                matched = str(actual_val).lower().strip() == str(expected_val).lower().strip()
                key_score = 1.0 if matched else 0.0
                key_scores[key] = {
                    "expected": expected_val, "actual": actual_val,
                    "matched": matched
                }

        score = sum(d.get("jaccard", 1.0 if d.get("matched") else 0.0) for d in key_scores.values()) / len(target)
        return score, {"type": "object", "keys": key_scores, "score": score}

    # ==================== Region ====================

    def _eval_region_param(self, called_params: Dict, param_eval: Dict) -> Tuple[float, Dict]:
        target = param_eval.get("target", {})
        metric = param_eval.get("metric", "iou")

        x_range = called_params.get("x_range", called_params.get("x", [0, 0]))
        y_range = called_params.get("y_range", called_params.get("y", [0, 0]))
        called_region = {"x": x_range, "y": y_range}

        if metric == "containment":
            score = self._compute_containment(called_region, target)
        else:
            score = self._compute_iou(called_region, target)

        return score, {
            "type": "region", "called": called_region, "target": target,
            "metric": metric, "score": score
        }

    # ==================== Categorical ====================

    def _eval_categorical_param(self, called_params: Dict, param_eval: Dict) -> Tuple[float, Dict]:
        param_name = param_eval.get("param", param_eval.get("param_field"))
        target = param_eval.get("target")
        alternatives = param_eval.get("alternatives", [])

        if param_name:
            called_value = called_params.get(param_name, "")
        else:
            # Try common parameter names
            called_value = ""
            for key in ["field", "field_name", "category", "value", "method"]:
                if key in called_params:
                    called_value = called_params[key]
                    break

        called_norm = str(called_value).lower().strip()
        valid_answers = [str(target).lower().strip()] + [str(alt).lower().strip() for alt in alternatives]
        matched = called_norm in valid_answers
        score = 1.0 if matched else 0.0

        return score, {
            "type": "categorical", "param": param_name,
            "called": called_value, "target": target, "matched": matched
        }

    # ==================== Numeric ====================

    def _eval_numeric_param(self, called_params: Dict, param_eval: Dict) -> Tuple[float, Dict]:
        param_name = param_eval.get("param", "")
        target = param_eval.get("target")
        tolerance = param_eval.get("tolerance", 0.05)

        called_val = float(called_params.get(param_name, 0))
        target_val = float(target)

        # Tolerance can be a list of acceptable values
        if isinstance(tolerance, list):
            acceptable = {float(t) for t in tolerance} | {target_val}
            matched = called_val in acceptable
            score = 1.0 if matched else 0.0
            return score, {
                "type": "numeric", "called": called_val, "target": target_val,
                "tolerance_list": tolerance, "matched": matched
            }

        diff = abs(called_val - target_val)
        score = 1.0 if diff <= tolerance else 0.0
        return score, {
            "type": "numeric", "param": param_name,
            "called": called_val, "target": target_val,
            "tolerance": tolerance, "difference": diff
        }

    # ==================== List ====================

    def _eval_list_param(self, called_params: Dict, param_eval: Dict) -> Tuple[float, Dict]:
        param_name = param_eval.get("param", "")
        target = param_eval.get("target", [])
        called_value = called_params.get(param_name, [])

        if not isinstance(called_value, list):
            called_value = [called_value]
        if not isinstance(target, list):
            target = [target]

        called_set = {str(v).lower().strip() for v in called_value}
        target_set = {str(v).lower().strip() for v in target}

        intersection = len(called_set & target_set)
        union = len(called_set | target_set)
        jaccard = intersection / union if union > 0 else 1.0

        return jaccard, {
            "type": "list", "param": param_name,
            "called": list(called_set), "target": list(target_set),
            "jaccard": jaccard
        }

    # ==================== Date ====================

    def _eval_date_param(self, called_params: Dict, param_eval: Dict) -> Tuple[float, Dict]:
        target = param_eval.get("target", {})
        start = called_params.get("start", called_params.get("start_date", ""))
        end = called_params.get("end", called_params.get("end_date", ""))

        called_start = self._parse_date(start)
        called_end = self._parse_date(end)
        target_start = self._parse_date(target["start"])
        target_end = self._parse_date(target["end"])

        overlap_start = max(called_start, target_start)
        overlap_end = min(called_end, target_end)

        if overlap_end <= overlap_start:
            iou = 0.0
        else:
            overlap = (overlap_end - overlap_start).days
            total = (max(called_end, target_end) - min(called_start, target_start)).days
            iou = overlap / total if total > 0 else 0.0

        return iou, {
            "type": "date",
            "called": {"start": str(start), "end": str(end)},
            "target": {"start": target["start"], "end": target["end"]},
            "iou": iou
        }

    # ==================== Order ====================

    def _eval_order_param(self, called_params: Dict, param_eval: Dict) -> Tuple[float, Dict]:
        param_name = param_eval.get("param", "order")
        target = param_eval.get("target", [])
        called_value = called_params.get(param_name, [])

        called_norm = [str(v).lower().strip() for v in called_value]
        target_norm = [str(v).lower().strip() for v in target]

        if not called_norm or not target_norm:
            return 0.0, {"type": "order", "called": called_norm, "target": target_norm}

        max_len = max(len(called_norm), len(target_norm))
        correct = sum(
            1 for i in range(min(len(called_norm), len(target_norm)))
            if called_norm[i] == target_norm[i]
        )
        score = correct / max_len

        return score, {
            "type": "order", "param": param_name,
            "called": called_norm, "target": target_norm,
            "correct_positions": correct, "score": score
        }

    # ==================== Selection ====================

    def _eval_selection_param(self, called_params: Dict, param_eval: Dict) -> Tuple[float, Dict]:
        target_field = param_eval.get("param_field", "")
        target_op = param_eval.get("param_op", "==")
        target_value = param_eval.get("target", "")
        alternatives = param_eval.get("alternatives", [])

        called_field = called_params.get("field", "")
        called_op = called_params.get("op", "==")
        called_value = called_params.get("value", called_params.get("values", ""))

        field_match = str(called_field).lower() == str(target_field).lower()
        op_match = str(called_op) == str(target_op)

        valid_values = [str(target_value).lower().strip()] + [str(alt).lower().strip() for alt in alternatives]
        value_match = str(called_value).lower().strip() in valid_values

        score = (field_match + op_match + value_match) / 3

        return score, {
            "type": "selection",
            "called": {"field": called_field, "op": called_op, "value": called_value},
            "target": {"field": target_field, "op": target_op, "value": target_value},
            "field_match": field_match, "op_match": op_match, "value_match": value_match,
            "score": score
        }

    # ==================== Geometry helpers ====================

    def _compute_iou(self, region1: Dict, region2: Dict) -> float:
        x1, y1 = region1.get("x", [0, 0]), region1.get("y", [0, 0])
        x2, y2 = region2.get("x", [0, 0]), region2.get("y", [0, 0])

        x_inter = max(0, min(x1[1], x2[1]) - max(x1[0], x2[0]))
        y_inter = max(0, min(y1[1], y2[1]) - max(y1[0], y2[0]))
        intersection = x_inter * y_inter

        area1 = (x1[1] - x1[0]) * (y1[1] - y1[0])
        area2 = (x2[1] - x2[0]) * (y2[1] - y2[0])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _compute_containment(self, region1: Dict, region2: Dict) -> float:
        x1, y1 = region1.get("x", [0, 0]), region1.get("y", [0, 0])
        x2, y2 = region2.get("x", [0, 0]), region2.get("y", [0, 0])

        x_inter = max(0, min(x1[1], x2[1]) - max(x1[0], x2[0]))
        y_inter = max(0, min(y1[1], y2[1]) - max(y1[0], y2[0]))
        intersection = x_inter * y_inter
        area2 = (x2[1] - x2[0]) * (y2[1] - y2[0])

        return intersection / area2 if area2 > 0 else 0.0

    def _parse_date(self, date_str: str) -> datetime:
        for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S"]:
            try:
                return datetime.strptime(str(date_str), fmt)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse date: {date_str}")