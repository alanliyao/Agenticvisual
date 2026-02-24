"""
State Evaluator

Evaluates visualization state based on Vega-Lite spec fields.
For Vega specs (e.g., Sankey), separate handling is required.

State types:
- visible_domain: IoU-based domain matching
- encoding: field matching
- selections: selection existence and region overlap
- data_filtered: filter category matching
- layers: layer existence checking
- dimensions: field count/list for parallel coordinates
- axis_order: sort order checking
- node_visibility: Sankey node visibility
- node_order: Sankey node ordering
- color_scale: color scale domain checking
- transpose: x/y field swap checking
- clustering_order: clustering sort order
- anomaly_markers: anomaly point markers in layers
- opacity: opacity value/field checking
- time_unit: time resampling checking
- sorting: axis sort checking
- grouping: grouped bar encoding (xOffset, color)
- mark_mode: mark type and stacking configuration
- null: returns full score (1.0)
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class StateEvalResult:
    """State evaluation result"""
    score: float  # 0.0 to 1.0
    passed: bool
    details: Dict[str, Any]


class StateEvaluator:
    """Evaluator for visualization state"""
    
    def get_vega_lite_state(self, vega_spec: Dict) -> Dict:
        """
        Extract state fields from Vega-Lite spec.
        
        Args:
            vega_spec: Vega-Lite specification
            
        Returns:
            Dict with extracted state fields
        """
        # Handle layer specs: extract encoding from first layer if top-level encoding is empty
        layers = vega_spec.get("layer", [])
        if layers and not vega_spec.get("encoding"):
            # Use first layer's encoding and mark
            encoding = layers[0].get("encoding", {})
            mark = layers[0].get("mark", {})
        else:
            encoding = vega_spec.get("encoding", {})
            mark = vega_spec.get("mark", {})
        
        # Extract mark info (handle both string and dict format)
        if isinstance(mark, dict):
            mark_type = mark.get("type", "")
            mark_config = mark
        else:
            mark_type = mark
            mark_config = {}
        
        # Extract all encoding fields for dimension counting
        encoding_fields = []
        for channel, config in encoding.items():
            if isinstance(config, dict) and config.get("field"):
                encoding_fields.append(config.get("field"))
        
        # Extract layer information with more details
        layers = vega_spec.get("layer", [])
        layer_details = []
        for layer in layers:
            layer_mark = layer.get("mark", {})
            if isinstance(layer_mark, dict):
                layer_details.append({
                    "type": layer_mark.get("type", ""),
                    "color": layer_mark.get("color", ""),
                    "point": layer_mark.get("point", False)
                })
            else:
                layer_details.append({"type": layer_mark})
        
        # Extract transform details
        transforms = vega_spec.get("transform", [])
        time_unit_transforms = [t for t in transforms if "timeUnit" in t]
        aggregate_transforms = [t for t in transforms if "aggregate" in t]
        
        return {
            # Visible domain
            "x_domain": encoding.get("x", {}).get("scale", {}).get("domain"),
            "y_domain": encoding.get("y", {}).get("scale", {}).get("domain"),
            
            # Encoding fields
            "color_field": encoding.get("color", {}).get("field"),
            "color_scale_domain": encoding.get("color", {}).get("scale", {}).get("domain"),
            "color_scale_range": encoding.get("color", {}).get("scale", {}).get("range"),
            "size_field": encoding.get("size", {}).get("field"),
            "x_field": encoding.get("x", {}).get("field"),
            "y_field": encoding.get("y", {}).get("field"),
            "opacity_field": encoding.get("opacity", {}).get("field"),
            "opacity_value": encoding.get("opacity", {}).get("value"),
            "opacity_condition": encoding.get("opacity", {}).get("condition"),
            
            # xOffset for grouped bars
            "x_offset_field": encoding.get("xOffset", {}).get("field"),
            "y_offset_field": encoding.get("yOffset", {}).get("field"),
            
            # Sorting
            "x_sort": encoding.get("x", {}).get("sort"),
            "y_sort": encoding.get("y", {}).get("sort"),
            
            # Resolve scale (for faceted/layered charts)
            "resolve_scale": vega_spec.get("resolve", {}).get("scale", {}),
            
            # Transform/filter
            "transforms": transforms,
            "has_filter": any(t.get("filter") for t in transforms),
            "filter_conditions": [t.get("filter") for t in transforms if t.get("filter")],
            "time_unit_transforms": time_unit_transforms,
            "aggregate_transforms": aggregate_transforms,
            
            # Layers
            "layer_count": len(layers),
            "has_layer": len(layers) > 0,
            "layer_marks": [
                (l.get("mark", {}).get("type", l.get("mark")) if isinstance(l.get("mark"), dict) else l.get("mark"))
                for l in layers
            ],
            "layer_details": layer_details,
            
            # Selection/params
            "params": vega_spec.get("params", []),
            "has_selection": len(vega_spec.get("params", [])) > 0,
            "selection_names": [p.get("name") for p in vega_spec.get("params", [])],
            
            # Data
            "data_count": len(vega_spec.get("data", {}).get("values", [])),
            "data_values": vega_spec.get("data", {}).get("values", []),
            
            # Mark
            "mark_type": mark_type,
            "mark_config": mark_config,
            
            # Dimensions (for parallel coordinates)
            "encoding_fields": encoding_fields,
            "encoding_field_count": len(encoding_fields),
            
            # Full encoding for detailed inspection
            "encoding": encoding,
        }
    
    def get_vega_state(self, vega_spec: Dict) -> Dict:
        """
        Extract state fields from Vega spec (for Sankey, etc.).
        
        Args:
            vega_spec: Vega specification
            
        Returns:
            Dict with extracted state fields
        """
        # Vega specs have different structure
        data = vega_spec.get("data", [])
        signals = vega_spec.get("signals", [])
        marks = vega_spec.get("marks", [])
        
        # Extract data names, counts, and values
        data_info = {}
        data_values = {}
        for d in data:
            if isinstance(d, dict):
                name = d.get("name", "")
                values = d.get("values", d.get("source", []))
                if isinstance(values, list):
                    data_info[name] = len(values)
                    data_values[name] = values
        
        # Extract signal values
        signal_values = {}
        for s in signals:
            if isinstance(s, dict):
                signal_values[s.get("name", "")] = s.get("value")
        
        # Extract node visibility from data (for Sankey expand/fold)
        node_visibility = {}
        nodes_data = None
        for d in data:
            if isinstance(d, dict) and "node" in d.get("name", "").lower():
                nodes_data = d.get("values", [])
                break
        
        if nodes_data:
            for node in nodes_data:
                if isinstance(node, dict):
                    node_id = node.get("id", node.get("name", ""))
                    # Check for visibility flag or collapsed state
                    visible = node.get("visible", node.get("expanded", True))
                    node_visibility[node_id] = visible
        
        # Extract node order
        node_order = []
        if nodes_data:
            for node in nodes_data:
                if isinstance(node, dict):
                    node_order.append(node.get("id", node.get("name", "")))
        
        # Extract mark colors/encodings
        mark_colors = []
        for m in marks:
            if isinstance(m, dict):
                encode = m.get("encode", {})
                update = encode.get("update", {})
                fill = update.get("fill", {})
                if fill:
                    mark_colors.append(fill)
        
        return {
            "data_sources": list(data_info.keys()),
            "data_counts": data_info,
            "data_values": data_values,
            "signals": signal_values,
            "marks": [m.get("type") for m in marks if isinstance(m, dict)],
            "mark_colors": mark_colors,
            "node_visibility": node_visibility,
            "node_order": node_order,
        }
    
    def evaluate(self, 
                 actual_spec: Dict, 
                 expected_state: Optional[Dict],
                 spec_type: str = "vega-lite") -> StateEvalResult:
        """
        Evaluate actual spec state against expected state.
        
        Args:
            actual_spec: Actual Vega/Vega-Lite specification
            expected_state: Expected state configuration from ground truth
            spec_type: "vega-lite" or "vega"
            
        Returns:
            StateEvalResult with score, passed flag, and details
        """
        # Handle null expected_state - return full score
        if expected_state is None:
            return StateEvalResult(
                score=1.0,
                passed=True,
                details={"note": "No state evaluation required (null)"}
            )
        
        # Extract actual state
        if spec_type == "vega-lite":
            actual_state = self.get_vega_lite_state(actual_spec)
        else:
            actual_state = self.get_vega_state(actual_spec)
        
        # Evaluate each state requirement
        scores = []
        details = {}
        
        # ==================== Existing evaluations ====================
        
        # visible_domain - for Scatter zoom, Line zoom_time_range/drilldown
        if "visible_domain" in expected_state:
            score, result = self._eval_visible_domain(actual_state, expected_state["visible_domain"])
            scores.append(score)
            details["visible_domain"] = result
        
        # encoding - for Scatter change_encoding
        if "encoding" in expected_state:
            score, result = self._eval_encoding(actual_state, expected_state["encoding"])
            scores.append(score)
            details["encoding"] = result
        
        # selections - for highlight/filter, highlight_region
        if "selections" in expected_state:
            score, result = self._eval_selections(actual_state, actual_spec, expected_state["selections"])
            scores.append(score)
            details["selections"] = result
        
        # data_filtered - for filter operations
        if "data_filtered" in expected_state:
            df = expected_state["data_filtered"]
            # GT format is a list of filter dicts: [{"filter": "..."}, ...]
            expected_filters = df if isinstance(df, list) else [df]
            score, result = self._eval_data_filtered(actual_state, expected_filters)
            scores.append(score)
            details["data_filtered"] = result
        
        # layers - for regression_line, marginal_bars
        if "layers" in expected_state:
            score, result = self._eval_layers(actual_state, expected_state["layers"])
            scores.append(score)
            details["layers"] = result
        
        # ==================== New evaluations ====================
        
        # dimensions - for Parallel hide_dimensions
        if "dimensions" in expected_state:
            score, result = self._eval_dimensions(actual_state, expected_state["dimensions"])
            scores.append(score)
            details["dimensions"] = result
        
        # axis_order - for Parallel change_axis_order
        if "axis_order" in expected_state:
            score, result = self._eval_axis_order(actual_state, expected_state["axis_order"])
            scores.append(score)
            details["axis_order"] = result
        
        # node_visibility - for Sankey expand/fold_node
        if "node_visibility" in expected_state:
            score, result = self._eval_node_visibility(actual_state, expected_state["node_visibility"])
            scores.append(score)
            details["node_visibility"] = result
        
        # node_order - for Sankey reorder_nodes
        if "node_order" in expected_state:
            score, result = self._eval_node_order(actual_state, expected_state["node_order"])
            scores.append(score)
            details["node_order"] = result
        
        # color_scale - for Heatmap adjust_color_scale
        if "color_scale" in expected_state:
            score, result = self._eval_color_scale(actual_state, expected_state["color_scale"])
            scores.append(score)
            details["color_scale"] = result
        
        # transpose - for Heatmap transpose (x/y field swap)
        if "transpose" in expected_state:
            score, result = self._eval_transpose(actual_state, expected_state["transpose"])
            scores.append(score)
            details["transpose"] = result
        
        # clustering_order - for Heatmap cluster_rows_cols
        if "clustering_order" in expected_state:
            score, result = self._eval_clustering_order(actual_state, expected_state["clustering_order"])
            scores.append(score)
            details["clustering_order"] = result
        
        # anomaly_markers - for Line detect_anomalies
        if "anomaly_markers" in expected_state:
            score, result = self._eval_anomaly_markers(actual_state, expected_state["anomaly_markers"])
            scores.append(score)
            details["anomaly_markers"] = result
        
        # opacity - for Line bold/filter/focus_lines, Bar highlight_top_n
        if "opacity" in expected_state:
            score, result = self._eval_opacity(actual_state, expected_state["opacity"])
            scores.append(score)
            details["opacity"] = result
        
        # time_unit - for Line resample_time
        if "time_unit" in expected_state:
            score, result = self._eval_time_unit(actual_state, expected_state["time_unit"])
            scores.append(score)
            details["time_unit"] = result
        
        # sorting - for Bar sort
        if "sorting" in expected_state:
            score, result = self._eval_sorting(actual_state, expected_state["sorting"])
            scores.append(score)
            details["sorting"] = result
        
        # grouping - for Bar compare_group
        if "grouping" in expected_state:
            score, result = self._eval_grouping(actual_state, expected_state["grouping"])
            scores.append(score)
            details["grouping"] = result
        
        # mark_mode - for Bar toggle_mode (stacking)
        if "mark_mode" in expected_state:
            score, result = self._eval_mark_mode(actual_state, actual_spec, expected_state["mark_mode"])
            scores.append(score)
            details["mark_mode"] = result
        
        # cluster_field - for Scatter identify_clusters
        if "cluster_field" in expected_state:
            score, result = self._eval_cluster_field(actual_state, expected_state["cluster_field"])
            scores.append(score)
            details["cluster_field"] = result
        
        # Calculate overall score
        if scores:
            total_score = sum(scores) / len(scores)
            passed = all(s >= 0.5 for s in scores)  # Passed if all components >= 0.5
        else:
            total_score = 1.0
            passed = True
        
        return StateEvalResult(
            score=total_score,
            passed=passed,
            details={
                "components": details,
                "component_scores": scores,
                "actual_state": {k: v for k, v in actual_state.items() 
                                if v is not None and k not in ["data_values", "encoding"]}
            }
        )
    
    # ==================== Existing evaluation methods ====================
    
    def _eval_visible_domain(self, actual_state: Dict, expected: Dict) -> Tuple[float, Dict]:
        """
        Evaluate visible domain using IoU.
        Used for: Scatter zoom/select_region, Line zoom_time_range/drilldown_line_time
        """
        scores = []
        results = {}
        
        # X domain
        if "x" in expected:
            actual_x = actual_state.get("x_domain")
            expected_x = expected["x"]
            if actual_x and expected_x:
                iou = self._compute_1d_iou(actual_x, expected_x)
                scores.append(iou)
                results["x_domain"] = {
                    "actual": actual_x,
                    "expected": expected_x,
                    "iou": iou
                }
            else:
                scores.append(0.0)
                results["x_domain"] = {"error": "Missing domain", "actual": actual_x, "expected": expected_x}
        
        # Y domain
        if "y" in expected:
            actual_y = actual_state.get("y_domain")
            expected_y = expected["y"]
            if actual_y and expected_y:
                iou = self._compute_1d_iou(actual_y, expected_y)
                scores.append(iou)
                results["y_domain"] = {
                    "actual": actual_y,
                    "expected": expected_y,
                    "iou": iou
                }
            else:
                scores.append(0.0)
                results["y_domain"] = {"error": "Missing domain", "actual": actual_y, "expected": expected_y}
        
        total_score = sum(scores) / len(scores) if scores else 1.0
        
        return total_score, results
    
    def _eval_encoding(self, actual_state: Dict, expected: Dict) -> Tuple[float, Dict]:
        """
        Evaluate encoding fields.
        Used for: Scatter change_encoding, Sankey highlight_path/color_nodes
        """
        scores = []
        results = {}
        
        for channel in ["color", "size", "x", "y", "opacity"]:
            if channel in expected:
                expected_field = expected[channel]
                actual_field = actual_state.get(f"{channel}_field", "")
                
                if expected_field is None:
                    # Expecting no field
                    matched = actual_field is None or actual_field == ""
                else:
                    matched = str(actual_field).lower() == str(expected_field).lower()
                
                scores.append(1.0 if matched else 0.0)
                results[f"{channel}_field"] = {
                    "actual": actual_field,
                    "expected": expected_field,
                    "matched": matched
                }
            
            # Check for pattern matching (e.g., contains "cluster")
            if f"{channel}_pattern" in expected:
                pattern = expected[f"{channel}_pattern"].lower()
                actual_field = actual_state.get(f"{channel}_field", "") or ""
                matched = pattern in actual_field.lower()
                scores.append(1.0 if matched else 0.0)
                results[f"{channel}_field_pattern"] = {
                    "actual": actual_field,
                    "pattern": pattern,
                    "matched": matched
                }
        
        total_score = sum(scores) / len(scores) if scores else 1.0
        
        return total_score, results
    
    def _eval_selections(self, actual_state: Dict, actual_spec: Dict, expected: Dict) -> Tuple[float, Dict]:
        """
        Evaluate selections.
        Used for: Scatter highlight/filter, Heatmap highlight_region, Bar highlight_top_n
        """
        scores = []
        results = {}
        
        # Check existence
        if "exists" in expected:
            has_selection = actual_state.get("has_selection", False)
            matched = has_selection == expected["exists"]
            scores.append(1.0 if matched else 0.0)
            results["exists"] = {"actual": has_selection, "expected": expected["exists"], "matched": matched}
        
        # Check region overlap
        if "region_overlap" in expected:
            region_config = expected["region_overlap"]
            target = region_config.get("target", {})
            threshold = region_config.get("threshold", 0.5)
            metric = region_config.get("metric", "iou")
            
            # Try to extract selection region from params
            selection_region = self._extract_selection_region(actual_spec)
            
            if selection_region:
                if metric == "iou":
                    score = self._compute_iou(selection_region, target)
                else:
                    score = self._compute_containment(selection_region, target)
                scores.append(score)
                results["region_overlap"] = {
                    "actual": selection_region,
                    "target": target,
                    f"{metric}_value": score,
                    "passed": score >= threshold
                }
            else:
                scores.append(0.0)
                results["region_overlap"] = {"error": "No selection region found"}
        
        # Check field/value
        if "field" in expected:
            expected_field = expected["field"]
            expected_value = expected.get("value", "")
            
            has_selection = actual_state.get("has_selection", False)
            scores.append(1.0 if has_selection else 0.0)
            results["field_value"] = {
                "expected_field": expected_field,
                "expected_value": expected_value,
                "has_selection": has_selection
            }
        
        total_score = sum(scores) / len(scores) if scores else 1.0
        
        return total_score, results
    
    def _eval_data_filtered(self, actual_state: Dict, expected_filters: List[Dict]) -> Tuple[float, Dict]:
        """
        Evaluate data filtering by comparing filter expressions.
        
        For each GT filter expression, check if the actual filter conditions
        contain the same content (simple string containment on the full
        GT expression against the concatenated actual conditions).
        
        Args:
            actual_state: extracted state from actual spec
            expected_filters: [{"filter": "<expression>"}, ...]
        """
        if not actual_state.get("has_filter", False):
            return 0.0, {"error": "No filter transform found"}

        actual_conditions = actual_state.get("filter_conditions", [])
        # Concatenate all actual filter conditions into one searchable string
        actual_combined = " ".join(str(c).lower() for c in actual_conditions)

        # For each GT filter, extract the category values mentioned and check coverage
        gt_categories = []
        for f in expected_filters:
            expr = f.get("filter", "")
            # Extract double-quoted values from GT expression (category names)
            values = re.findall(r'"([^"]+)"', expr.replace('\\"', '"'))
            gt_categories.extend(v.lower() for v in values)

        if not gt_categories:
            return 1.0, {"has_filter": True, "note": "no GT categories to check"}

        # Check how many GT categories appear in the actual filter
        found = [cat for cat in gt_categories if cat in actual_combined]
        missing = [cat for cat in gt_categories if cat not in actual_combined]
        score = len(found) / len(gt_categories)

        return score, {
            "has_filter": True,
            "gt_categories": gt_categories,
            "found": found,
            "missing": missing,
            "coverage": score
        }
    
    def _eval_layers(self, actual_state: Dict, expected: Dict) -> Tuple[float, Dict]:
        """
        Evaluate layers.
        Used for: Line highlight_trend, Heatmap add_marginal_bars
        """
        scores = []
        results = {}
        
        # Check regression line
        if "regression_line" in expected:
            expected_has = expected["regression_line"]
            layer_marks = actual_state.get("layer_marks", [])
            has_regression = any("line" in str(m).lower() or "rule" in str(m).lower() 
                                for m in layer_marks)
            transforms = actual_state.get("transforms", [])
            has_regression_transform = any("regression" in str(t).lower() for t in transforms)
            
            actual_has = has_regression or has_regression_transform or actual_state.get("layer_count", 0) > 1
            matched = actual_has == expected_has
            scores.append(1.0 if matched else 0.0)
            results["regression_line"] = {
                "expected": expected_has,
                "actual": actual_has,
                "matched": matched
            }
        
        # Check layer count (for marginal bars, etc.)
        if "count" in expected:
            actual_count = actual_state.get("layer_count", 0)
            expected_count = expected["count"]
            matched = actual_count >= expected_count
            scores.append(1.0 if matched else 0.0)
            results["layer_count"] = {
                "expected": expected_count,
                "actual": actual_count,
                "matched": matched
            }
        
        # Check for specific layer types
        if "has_type" in expected:
            expected_type = expected["has_type"].lower()
            layer_marks = actual_state.get("layer_marks", [])
            has_type = any(expected_type in str(m).lower() for m in layer_marks)
            scores.append(1.0 if has_type else 0.0)
            results["has_type"] = {
                "expected": expected_type,
                "layer_marks": layer_marks,
                "matched": has_type
            }
        
        total_score = sum(scores) / len(scores) if scores else 1.0
        
        return total_score, results
    
    # ==================== New evaluation methods ====================
    
    def _eval_dimensions(self, actual_state: Dict, expected: Dict) -> Tuple[float, Dict]:
        """
        Evaluate dimensions for parallel coordinates.
        Used for: Parallel hide_dimensions
        
        Checks:
        - field_count: number of encoding fields
        - visible_fields: list of fields that should be visible
        - hidden_fields: list of fields that should be hidden
        """
        scores = []
        results = {}
        
        actual_fields = actual_state.get("encoding_fields", [])
        actual_count = len(actual_fields)
        
        # Check field count
        if "field_count" in expected:
            expected_count = expected["field_count"]
            matched = actual_count == expected_count
            scores.append(1.0 if matched else 0.0)
            results["field_count"] = {
                "actual": actual_count,
                "expected": expected_count,
                "matched": matched
            }
        
        # Check visible fields
        if "visible_fields" in expected:
            expected_visible = set(f.lower() for f in expected["visible_fields"])
            actual_visible = set(f.lower() for f in actual_fields)
            intersection = len(expected_visible & actual_visible)
            union = len(expected_visible | actual_visible)
            jaccard = intersection / union if union > 0 else 1.0
            scores.append(jaccard)
            results["visible_fields"] = {
                "actual": list(actual_visible),
                "expected": list(expected_visible),
                "jaccard": jaccard
            }
        
        # Check hidden fields (should NOT be in actual)
        if "hidden_fields" in expected:
            expected_hidden = set(f.lower() for f in expected["hidden_fields"])
            actual_visible = set(f.lower() for f in actual_fields)
            # Fields that should be hidden but are still visible
            incorrectly_visible = expected_hidden & actual_visible
            if len(expected_hidden) > 0:
                score = 1.0 - (len(incorrectly_visible) / len(expected_hidden))
            else:
                score = 1.0
            scores.append(score)
            results["hidden_fields"] = {
                "expected_hidden": list(expected_hidden),
                "incorrectly_visible": list(incorrectly_visible),
                "score": score
            }
        
        total_score = sum(scores) / len(scores) if scores else 1.0
        return total_score, results
    
    def _eval_axis_order(self, actual_state: Dict, expected: Dict) -> Tuple[float, Dict]:
        """
        Evaluate axis order/sort.
        Used for: Parallel change_axis_order
        
        Checks:
        - x_sort: sort configuration for x axis
        - y_sort: sort configuration for y axis
        - resolve_scale: scale resolution for multi-view
        """
        scores = []
        results = {}
        
        # Check x sort
        if "x_sort" in expected:
            actual_sort = actual_state.get("x_sort")
            expected_sort = expected["x_sort"]
            matched = self._compare_sort(actual_sort, expected_sort)
            scores.append(1.0 if matched else 0.0)
            results["x_sort"] = {
                "actual": actual_sort,
                "expected": expected_sort,
                "matched": matched
            }
        
        # Check y sort
        if "y_sort" in expected:
            actual_sort = actual_state.get("y_sort")
            expected_sort = expected["y_sort"]
            matched = self._compare_sort(actual_sort, expected_sort)
            scores.append(1.0 if matched else 0.0)
            results["y_sort"] = {
                "actual": actual_sort,
                "expected": expected_sort,
                "matched": matched
            }
        
        # Check resolve.scale
        if "resolve_scale" in expected:
            actual_resolve = actual_state.get("resolve_scale", {})
            expected_resolve = expected["resolve_scale"]
            matched = actual_resolve == expected_resolve
            scores.append(1.0 if matched else 0.0)
            results["resolve_scale"] = {
                "actual": actual_resolve,
                "expected": expected_resolve,
                "matched": matched
            }
        
        # Check dimension order (list of field names in order)
        if "dimension_order" in expected:
            actual_fields = actual_state.get("encoding_fields", [])
            expected_order = expected["dimension_order"]
            score = self._compute_order_similarity(actual_fields, expected_order)
            scores.append(score)
            results["dimension_order"] = {
                "actual": actual_fields,
                "expected": expected_order,
                "score": score
            }
        
        total_score = sum(scores) / len(scores) if scores else 1.0
        return total_score, results
    
    def _eval_node_visibility(self, actual_state: Dict, expected: Dict) -> Tuple[float, Dict]:
        """
        Evaluate node visibility for Sankey diagrams.
        Used for: Sankey expand/fold_node
        
        Checks:
        - visible_nodes: list of nodes that should be visible
        - hidden_nodes: list of nodes that should be hidden/collapsed
        """
        scores = []
        results = {}
        
        actual_visibility = actual_state.get("node_visibility", {})
        
        # Check visible nodes
        if "visible_nodes" in expected:
            expected_visible = expected["visible_nodes"]
            correct = 0
            for node in expected_visible:
                if actual_visibility.get(node, True):  # Default to visible
                    correct += 1
            score = correct / len(expected_visible) if expected_visible else 1.0
            scores.append(score)
            results["visible_nodes"] = {
                "expected": expected_visible,
                "actual_visibility": actual_visibility,
                "score": score
            }
        
        # Check hidden/collapsed nodes
        if "hidden_nodes" in expected:
            expected_hidden = expected["hidden_nodes"]
            correct = 0
            for node in expected_hidden:
                if not actual_visibility.get(node, True):  # Should be hidden
                    correct += 1
            score = correct / len(expected_hidden) if expected_hidden else 1.0
            scores.append(score)
            results["hidden_nodes"] = {
                "expected": expected_hidden,
                "actual_visibility": actual_visibility,
                "score": score
            }
        
        total_score = sum(scores) / len(scores) if scores else 1.0
        return total_score, results
    
    def _eval_node_order(self, actual_state: Dict, expected: Dict) -> Tuple[float, Dict]:
        """
        Evaluate node order for Sankey diagrams.
        Used for: Sankey reorder_nodes
        
        Checks:
        - node_order: expected order of nodes
        """
        actual_order = actual_state.get("node_order", [])
        expected_order = expected.get("order", [])
        
        if not expected_order:
            return 1.0, {"note": "No expected order specified"}
        
        # Compute order similarity
        score = self._compute_order_similarity(actual_order, expected_order)
        
        return score, {
            "actual": actual_order,
            "expected": expected_order,
            "score": score
        }
    
    def _eval_color_scale(self, actual_state: Dict, expected: Dict) -> Tuple[float, Dict]:
        """
        Evaluate color scale configuration.
        Used for: Heatmap adjust_color_scale
        
        Checks:
        - domain: color scale domain values
        - range: color scale range values
        """
        scores = []
        results = {}
        
        # Check domain
        if "domain" in expected:
            actual_domain = actual_state.get("color_scale_domain")
            expected_domain = expected["domain"]
            
            if actual_domain and expected_domain:
                # Compare as lists
                if isinstance(actual_domain, list) and isinstance(expected_domain, list):
                    iou = self._compute_1d_iou(
                        [min(actual_domain), max(actual_domain)],
                        [min(expected_domain), max(expected_domain)]
                    )
                    scores.append(iou)
                    results["domain"] = {
                        "actual": actual_domain,
                        "expected": expected_domain,
                        "iou": iou
                    }
                else:
                    matched = actual_domain == expected_domain
                    scores.append(1.0 if matched else 0.0)
                    results["domain"] = {
                        "actual": actual_domain,
                        "expected": expected_domain,
                        "matched": matched
                    }
            else:
                scores.append(0.0)
                results["domain"] = {"error": "Missing domain"}
        
        # Check range
        if "range" in expected:
            actual_range = actual_state.get("color_scale_range")
            expected_range = expected["range"]
            matched = actual_range == expected_range
            scores.append(1.0 if matched else 0.0)
            results["range"] = {
                "actual": actual_range,
                "expected": expected_range,
                "matched": matched
            }
        
        total_score = sum(scores) / len(scores) if scores else 1.0
        return total_score, results
    
    def _eval_transpose(self, actual_state: Dict, expected: Dict) -> Tuple[float, Dict]:
        """
        Evaluate transpose (x/y field swap) for heatmaps.
        Used for: Heatmap transpose
        
        Checks:
        - transposed: boolean indicating if x and y should be swapped
        - x_field: expected x field after transpose
        - y_field: expected y field after transpose
        """
        scores = []
        results = {}
        
        actual_x = actual_state.get("x_field", "")
        actual_y = actual_state.get("y_field", "")
        
        # Check if transposed (fields swapped)
        if "transposed" in expected:
            original_x = expected.get("original_x", "")
            original_y = expected.get("original_y", "")
            
            if expected["transposed"]:
                # Should be swapped: actual_x == original_y and actual_y == original_x
                swapped = (str(actual_x).lower() == str(original_y).lower() and 
                          str(actual_y).lower() == str(original_x).lower())
            else:
                # Should NOT be swapped
                swapped = (str(actual_x).lower() == str(original_x).lower() and 
                          str(actual_y).lower() == str(original_y).lower())
            
            scores.append(1.0 if swapped else 0.0)
            results["transposed"] = {
                "expected_transposed": expected["transposed"],
                "actual_x": actual_x,
                "actual_y": actual_y,
                "original_x": original_x,
                "original_y": original_y,
                "matched": swapped
            }
        
        # Direct field check
        if "x_field" in expected:
            matched = str(actual_x).lower() == str(expected["x_field"]).lower()
            scores.append(1.0 if matched else 0.0)
            results["x_field"] = {"actual": actual_x, "expected": expected["x_field"], "matched": matched}
        
        if "y_field" in expected:
            matched = str(actual_y).lower() == str(expected["y_field"]).lower()
            scores.append(1.0 if matched else 0.0)
            results["y_field"] = {"actual": actual_y, "expected": expected["y_field"], "matched": matched}
        
        total_score = sum(scores) / len(scores) if scores else 1.0
        return total_score, results
    
    def _eval_clustering_order(self, actual_state: Dict, expected: Dict) -> Tuple[float, Dict]:
        """
        Evaluate clustering order for heatmaps.
        Used for: Heatmap cluster_rows_cols
        
        Checks:
        - y_sort: sort order indicating clustering
        - clustered: boolean indicating if clustering is applied
        """
        scores = []
        results = {}
        
        # Check if clustering is applied (usually indicated by sort)
        if "clustered" in expected:
            y_sort = actual_state.get("y_sort")
            # Clustering usually sets a custom sort order
            has_clustering = y_sort is not None and y_sort != "ascending" and y_sort != "descending"
            matched = has_clustering == expected["clustered"]
            scores.append(1.0 if matched else 0.0)
            results["clustered"] = {
                "expected": expected["clustered"],
                "actual_y_sort": y_sort,
                "has_clustering": has_clustering,
                "matched": matched
            }
        
        # Check specific sort order
        if "y_sort_order" in expected:
            actual_sort = actual_state.get("y_sort")
            expected_sort = expected["y_sort_order"]
            
            if isinstance(actual_sort, list) and isinstance(expected_sort, list):
                score = self._compute_order_similarity(actual_sort, expected_sort)
            else:
                score = 1.0 if actual_sort == expected_sort else 0.0
            
            scores.append(score)
            results["y_sort_order"] = {
                "actual": actual_sort,
                "expected": expected_sort,
                "score": score
            }
        
        total_score = sum(scores) / len(scores) if scores else 1.0
        return total_score, results
    
    def _eval_anomaly_markers(self, actual_state: Dict, expected: Dict) -> Tuple[float, Dict]:
        """
        Evaluate anomaly markers in layers.
        Used for: Line detect_anomalies
        
        Checks:
        - has_markers: whether anomaly point markers exist
        - marker_type: type of marker (point, circle, etc.)
        """
        scores = []
        results = {}
        
        layer_marks = actual_state.get("layer_marks", [])
        layer_details = actual_state.get("layer_details", [])
        
        # Check for anomaly markers (usually point or circle marks)
        if "has_markers" in expected:
            expected_has = expected["has_markers"]
            has_point_marks = any(
                "point" in str(m).lower() or "circle" in str(m).lower()
                for m in layer_marks
            )
            matched = has_point_marks == expected_has
            scores.append(1.0 if matched else 0.0)
            results["has_markers"] = {
                "expected": expected_has,
                "actual": has_point_marks,
                "layer_marks": layer_marks,
                "matched": matched
            }
        
        # Check marker type
        if "marker_type" in expected:
            expected_type = expected["marker_type"].lower()
            has_type = any(expected_type in str(m).lower() for m in layer_marks)
            scores.append(1.0 if has_type else 0.0)
            results["marker_type"] = {
                "expected": expected_type,
                "layer_marks": layer_marks,
                "matched": has_type
            }
        
        # Check for specific color (anomaly markers often have distinct color)
        if "marker_color" in expected:
            expected_color = expected["marker_color"].lower()
            has_color = any(
                expected_color in str(d.get("color", "")).lower()
                for d in layer_details
            )
            scores.append(1.0 if has_color else 0.0)
            results["marker_color"] = {
                "expected": expected_color,
                "matched": has_color
            }
        
        total_score = sum(scores) / len(scores) if scores else 1.0
        return total_score, results
    
    def _eval_opacity(self, actual_state: Dict, expected: Dict) -> Tuple[float, Dict]:
        """
        Evaluate opacity configuration.
        Used for: Line bold/filter/focus_lines, Bar highlight_top_n
        
        Checks:
        - has_opacity: whether opacity encoding exists
        - value: specific opacity value
        - field: field used for opacity encoding
        - has_condition: whether conditional opacity is used
        """
        scores = []
        results = {}
        
        actual_field = actual_state.get("opacity_field")
        actual_value = actual_state.get("opacity_value")
        actual_condition = actual_state.get("opacity_condition")
        
        # Check if opacity is used
        if "has_opacity" in expected:
            has_opacity = (actual_field is not None or 
                         actual_value is not None or 
                         actual_condition is not None)
            matched = has_opacity == expected["has_opacity"]
            scores.append(1.0 if matched else 0.0)
            results["has_opacity"] = {
                "expected": expected["has_opacity"],
                "actual": has_opacity,
                "matched": matched
            }
        
        # Check opacity value
        if "value" in expected:
            expected_value = expected["value"]
            if actual_value is not None:
                # Allow some tolerance
                diff = abs(float(actual_value) - float(expected_value))
                matched = diff <= 0.1
            else:
                matched = False
            scores.append(1.0 if matched else 0.0)
            results["value"] = {
                "expected": expected_value,
                "actual": actual_value,
                "matched": matched
            }
        
        # Check opacity field
        if "field" in expected:
            matched = str(actual_field or "").lower() == str(expected["field"]).lower()
            scores.append(1.0 if matched else 0.0)
            results["field"] = {
                "expected": expected["field"],
                "actual": actual_field,
                "matched": matched
            }
        
        # Check for conditional opacity
        if "has_condition" in expected:
            has_cond = actual_condition is not None
            matched = has_cond == expected["has_condition"]
            scores.append(1.0 if matched else 0.0)
            results["has_condition"] = {
                "expected": expected["has_condition"],
                "actual": has_cond,
                "matched": matched
            }
        
        total_score = sum(scores) / len(scores) if scores else 1.0
        return total_score, results
    
    def _eval_time_unit(self, actual_state: Dict, expected: Dict) -> Tuple[float, Dict]:
        """
        Evaluate time unit / resampling configuration.
        Used for: Line resample_time
        
        Checks:
        - has_time_unit: whether timeUnit transform exists
        - unit: specific time unit (year, month, day, etc.)
        """
        scores = []
        results = {}
        
        time_unit_transforms = actual_state.get("time_unit_transforms", [])
        
        # Check if time unit transform exists
        if "has_time_unit" in expected:
            has_time_unit = len(time_unit_transforms) > 0
            matched = has_time_unit == expected["has_time_unit"]
            scores.append(1.0 if matched else 0.0)
            results["has_time_unit"] = {
                "expected": expected["has_time_unit"],
                "actual": has_time_unit,
                "matched": matched
            }
        
        # Check specific time unit
        if "unit" in expected:
            expected_unit = expected["unit"].lower()
            found_unit = False
            for t in time_unit_transforms:
                if isinstance(t, dict):
                    unit = str(t.get("timeUnit", "")).lower()
                    if expected_unit in unit:
                        found_unit = True
                        break
            scores.append(1.0 if found_unit else 0.0)
            results["unit"] = {
                "expected": expected_unit,
                "transforms": time_unit_transforms,
                "matched": found_unit
            }
        
        total_score = sum(scores) / len(scores) if scores else 1.0
        return total_score, results
    
    def _eval_sorting(self, actual_state: Dict, expected: Dict) -> Tuple[float, Dict]:
        """
        Evaluate sorting configuration.
        Used for: Bar sort
        
        Checks:
        - x_sort: x axis sort (ascending, descending, or field-based)
        - y_sort: y axis sort
        """
        scores = []
        results = {}
        
        # Check x sort
        if "x_sort" in expected:
            actual_sort = actual_state.get("x_sort")
            expected_sort = expected["x_sort"]
            matched = self._compare_sort(actual_sort, expected_sort)
            scores.append(1.0 if matched else 0.0)
            results["x_sort"] = {
                "expected": expected_sort,
                "actual": actual_sort,
                "matched": matched
            }
        
        # Check y sort
        if "y_sort" in expected:
            actual_sort = actual_state.get("y_sort")
            expected_sort = expected["y_sort"]
            matched = self._compare_sort(actual_sort, expected_sort)
            scores.append(1.0 if matched else 0.0)
            results["y_sort"] = {
                "expected": expected_sort,
                "actual": actual_sort,
                "matched": matched
            }
        
        # Check if any sort is applied
        if "has_sort" in expected:
            has_sort = (actual_state.get("x_sort") is not None or 
                       actual_state.get("y_sort") is not None)
            matched = has_sort == expected["has_sort"]
            scores.append(1.0 if matched else 0.0)
            results["has_sort"] = {
                "expected": expected["has_sort"],
                "actual": has_sort,
                "matched": matched
            }
        
        total_score = sum(scores) / len(scores) if scores else 1.0
        return total_score, results
    
    def _eval_grouping(self, actual_state: Dict, expected: Dict) -> Tuple[float, Dict]:
        """
        Evaluate grouping configuration for grouped bar charts.
        Used for: Bar compare_group
        
        Checks:
        - x_offset_field: field used for xOffset encoding
        - color_field: field used for color encoding in grouped bars
        """
        scores = []
        results = {}
        
        # Check xOffset field
        if "x_offset_field" in expected:
            actual_field = actual_state.get("x_offset_field")
            expected_field = expected["x_offset_field"]
            matched = str(actual_field or "").lower() == str(expected_field).lower()
            scores.append(1.0 if matched else 0.0)
            results["x_offset_field"] = {
                "expected": expected_field,
                "actual": actual_field,
                "matched": matched
            }
        
        # Check color field (often used with grouping)
        if "color_field" in expected:
            actual_field = actual_state.get("color_field")
            expected_field = expected["color_field"]
            matched = str(actual_field or "").lower() == str(expected_field).lower()
            scores.append(1.0 if matched else 0.0)
            results["color_field"] = {
                "expected": expected_field,
                "actual": actual_field,
                "matched": matched
            }
        
        # Check if grouped (has either xOffset or specific color encoding)
        if "is_grouped" in expected:
            is_grouped = (actual_state.get("x_offset_field") is not None or
                         actual_state.get("y_offset_field") is not None)
            matched = is_grouped == expected["is_grouped"]
            scores.append(1.0 if matched else 0.0)
            results["is_grouped"] = {
                "expected": expected["is_grouped"],
                "actual": is_grouped,
                "matched": matched
            }
        
        total_score = sum(scores) / len(scores) if scores else 1.0
        return total_score, results
    
    def _eval_mark_mode(self, actual_state: Dict, actual_spec: Dict, expected: Dict) -> Tuple[float, Dict]:
        """
        Evaluate mark mode and stacking configuration.
        Used for: Bar toggle_mode
        
        Checks:
        - mark_type: type of mark (bar, area, line, etc.)
        - stacking: stack configuration (zero, normalize, center, null)
        """
        scores = []
        results = {}
        
        actual_mark_type = actual_state.get("mark_type", "")
        
        # Check mark type
        if "mark_type" in expected:
            expected_type = expected["mark_type"].lower()
            matched = str(actual_mark_type).lower() == expected_type
            scores.append(1.0 if matched else 0.0)
            results["mark_type"] = {
                "expected": expected_type,
                "actual": actual_mark_type,
                "matched": matched
            }
        
        # Check stacking configuration
        if "stacking" in expected:
            # Extract stack config from encoding or mark
            encoding = actual_spec.get("encoding", {})
            y_stack = encoding.get("y", {}).get("stack")
            x_stack = encoding.get("x", {}).get("stack")
            actual_stack = y_stack or x_stack
            
            expected_stack = expected["stacking"]
            matched = actual_stack == expected_stack
            scores.append(1.0 if matched else 0.0)
            results["stacking"] = {
                "expected": expected_stack,
                "actual": actual_stack,
                "matched": matched
            }
        
        # Check if stacked at all
        if "is_stacked" in expected:
            encoding = actual_spec.get("encoding", {})
            y_stack = encoding.get("y", {}).get("stack")
            x_stack = encoding.get("x", {}).get("stack")
            is_stacked = (y_stack is not None and y_stack != False) or \
                        (x_stack is not None and x_stack != False)
            matched = is_stacked == expected["is_stacked"]
            scores.append(1.0 if matched else 0.0)
            results["is_stacked"] = {
                "expected": expected["is_stacked"],
                "actual": is_stacked,
                "matched": matched
            }
        
        total_score = sum(scores) / len(scores) if scores else 1.0
        return total_score, results
    
    def _eval_cluster_field(self, actual_state: Dict, expected: Dict) -> Tuple[float, Dict]:
        """
        Evaluate cluster field for scatter plots.
        Used for: Scatter identify_clusters
        
        Checks:
        - color_field_contains: pattern that should be in color field name
        - has_cluster_encoding: whether clustering is reflected in encoding
        """
        scores = []
        results = {}
        
        actual_color_field = actual_state.get("color_field", "") or ""
        
        # Check if color field contains pattern (e.g., "cluster")
        if "color_field_contains" in expected:
            pattern = expected["color_field_contains"].lower()
            matched = pattern in actual_color_field.lower()
            scores.append(1.0 if matched else 0.0)
            results["color_field_contains"] = {
                "pattern": pattern,
                "actual_color_field": actual_color_field,
                "matched": matched
            }
        
        # Check specific color field
        if "color_field" in expected:
            expected_field = expected["color_field"]
            matched = actual_color_field.lower() == expected_field.lower()
            scores.append(1.0 if matched else 0.0)
            results["color_field"] = {
                "expected": expected_field,
                "actual": actual_color_field,
                "matched": matched
            }
        
        total_score = sum(scores) / len(scores) if scores else 1.0
        return total_score, results
    
    # ==================== Helper methods ====================
    
    def _extract_selection_region(self, spec: Dict) -> Optional[Dict]:
        """
        Extract selection region from spec params.
        
        FIX: Actually implemented instead of returning None.
        Looks for interval selection params with domain values.
        """
        params = spec.get("params", [])
        
        for param in params:
            if isinstance(param, dict):
                select = param.get("select", {})
                if isinstance(select, dict) and select.get("type") == "interval":
                    # Try to get region from param value
                    value = param.get("value", {})
                    if isinstance(value, dict):
                        region = {}
                        if "x" in value:
                            region["x"] = value["x"]
                        if "y" in value:
                            region["y"] = value["y"]
                        if region:
                            return region
                    
                    # Try to get from encodings + scale domain
                    encodings = select.get("encodings", [])
                    encoding = spec.get("encoding", {})
                    region = {}
                    for enc in encodings:
                        if enc in encoding:
                            domain = encoding[enc].get("scale", {}).get("domain")
                            if domain and isinstance(domain, list) and len(domain) >= 2:
                                region[enc] = domain
                    if region:
                        return region
        
        # Fallback: check if spec has selection store in data
        for layer in spec.get("layer", []):
            if isinstance(layer, dict):
                layer_params = layer.get("params", [])
                for param in layer_params:
                    if isinstance(param, dict):
                        select = param.get("select", {})
                        if isinstance(select, dict) and select.get("type") == "interval":
                            value = param.get("value", {})
                            if isinstance(value, dict) and ("x" in value or "y" in value):
                                return value
        
        return None
    
    def _compute_1d_iou(self, range1: List, range2: List) -> float:
        """Compute 1D IoU for ranges"""
        if not range1 or not range2 or len(range1) < 2 or len(range2) < 2:
            return 0.0
        
        min1, max1 = min(range1), max(range1)
        min2, max2 = min(range2), max(range2)
        
        intersection = max(0, min(max1, max2) - max(min1, min2))
        union = max(max1, max2) - min(min1, min2)
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_iou(self, region1: Dict, region2: Dict) -> float:
        """Compute 2D IoU"""
        x1 = region1.get("x", [0, 0]) or [0, 0]
        y1 = region1.get("y", [0, 0]) or [0, 0]
        x2 = region2.get("x", [0, 0]) or [0, 0]
        y2 = region2.get("y", [0, 0]) or [0, 0]
        
        if len(x1) < 2: x1 = [0, 0]
        if len(y1) < 2: y1 = [0, 0]
        if len(x2) < 2: x2 = [0, 0]
        if len(y2) < 2: y2 = [0, 0]
        
        x_inter = max(0, min(x1[1], x2[1]) - max(x1[0], x2[0]))
        y_inter = max(0, min(y1[1], y2[1]) - max(y1[0], y2[0]))
        intersection = x_inter * y_inter
        
        area1 = (x1[1] - x1[0]) * (y1[1] - y1[0])
        area2 = (x2[1] - x2[0]) * (y2[1] - y2[0])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_containment(self, region1: Dict, region2: Dict) -> float:
        """Compute containment of region2 in region1"""
        x1 = region1.get("x", [0, 0]) or [0, 0]
        y1 = region1.get("y", [0, 0]) or [0, 0]
        x2 = region2.get("x", [0, 0]) or [0, 0]
        y2 = region2.get("y", [0, 0]) or [0, 0]
        
        if len(x1) < 2: x1 = [0, 0]
        if len(y1) < 2: y1 = [0, 0]
        if len(x2) < 2: x2 = [0, 0]
        if len(y2) < 2: y2 = [0, 0]
        
        x_inter = max(0, min(x1[1], x2[1]) - max(x1[0], x2[0]))
        y_inter = max(0, min(y1[1], y2[1]) - max(y1[0], y2[0]))
        intersection = x_inter * y_inter
        
        area2 = (x2[1] - x2[0]) * (y2[1] - y2[0])
        
        return intersection / area2 if area2 > 0 else 0.0
    
    def _compare_sort(self, actual: Any, expected: Any) -> bool:
        """Compare sort configurations"""
        if actual is None and expected is None:
            return True
        if actual is None or expected is None:
            return False
        
        # Handle string comparison
        if isinstance(actual, str) and isinstance(expected, str):
            return actual.lower() == expected.lower()
        
        # Handle dict comparison
        if isinstance(actual, dict) and isinstance(expected, dict):
            return actual == expected
        
        # Handle list comparison (custom sort order)
        if isinstance(actual, list) and isinstance(expected, list):
            return actual == expected
        
        return str(actual).lower() == str(expected).lower()
    
    def _compute_order_similarity(self, actual: List, expected: List) -> float:
        """Compute similarity between two ordered lists"""
        if not actual or not expected:
            return 0.0 if actual or expected else 1.0
        
        # Normalize to lowercase strings
        actual_norm = [str(x).lower() for x in actual]
        expected_norm = [str(x).lower() for x in expected]
        
        # Count items in correct position
        correct = 0
        for i, item in enumerate(expected_norm):
            if i < len(actual_norm) and actual_norm[i] == item:
                correct += 1
        
        return correct / len(expected_norm)


# Convenience function
def evaluate_state(actual_spec: Dict, 
                   expected_state: Optional[Dict],
                   spec_type: str = "vega-lite") -> StateEvalResult:
    """Evaluate visualization state"""
    evaluator = StateEvaluator()
    return evaluator.evaluate(actual_spec, expected_state, spec_type)