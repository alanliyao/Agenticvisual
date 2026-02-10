import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from core.utils import app_logger, get_spec_data_values


class LargeDatasetManager:
    """
    manage sampling and incremental loading of large datasets (view limit default 500).
    - internally use displayed_ids (set) to record the full index of the displayed points, not exposed to the outside.
    - return the vega_spec.data.values only contains the points that need to be rendered, without the id list.
    """

    def __init__(
        self,
        full_values: List[Dict],
        x_field: Optional[str],
        y_field: Optional[str],
        view_limit: int = 500,
    ):
        self.full_values = full_values or []
        self.x_field = x_field
        self.y_field = y_field
        self.view_limit = max(1, int(view_limit or 500))
        self.displayed_ids = set()  # 全量数据索引集合

    @classmethod
    def from_spec(cls, spec: Dict, base_dir: Optional[Path] = None) -> "LargeDatasetManager":
        """from spec read full_data_path/view_limit/x_field/y_field, construct manager."""
        meta = spec.get("_metadata") or {}
        view_limit = meta.get("view_limit", 500)
        x_field = spec.get("encoding", {}).get("x", {}).get("field")
        y_field = spec.get("encoding", {}).get("y", {}).get("field")

        full_values = get_spec_data_values(spec) or []
        full_data_path = meta.get("full_data_path")

        if full_data_path:
            base_dir = base_dir or Path(__file__).resolve().parent.parent
            data_path = (base_dir / full_data_path).resolve()
            try:
                if data_path.exists():
                    loaded = json.loads(data_path.read_text(encoding="utf-8")).get("values", [])
                    if loaded:
                        full_values = loaded
                else:
                    app_logger.warning(f"full_data_path not found: {data_path}")
            except Exception as exc:  # noqa: BLE001
                app_logger.error(f"failed to load full_data_path {data_path}: {exc}")

        return cls(full_values=full_values, x_field=x_field, y_field=y_field, view_limit=view_limit)

    def _point_in_region(self, rec: Dict, region: Dict) -> bool:
        """determine if the point is in the region; region can be omitted any dimension."""
        if not region:
            return True

        def _in_range(val, lower, upper):
            if lower is not None and val < lower:
                return False
            if upper is not None and val > upper:
                return False
            return True

        # x
        if self.x_field and (region.get("x_min") is not None or region.get("x_max") is not None):
            x_val = rec.get(self.x_field)
            if not isinstance(x_val, (int, float)):
                return False
            if not _in_range(x_val, region.get("x_min"), region.get("x_max")):
                return False

        # y
        if self.y_field and (region.get("y_min") is not None or region.get("y_max") is not None):
            y_val = rec.get(self.y_field)
            if not isinstance(y_val, (int, float)):
                return False
            if not _in_range(y_val, region.get("y_min"), region.get("y_max")):
                return False

        return True

    def _current_displayed_values(self) -> List[Dict]:
        """return the points corresponding to the current displayed set (不超过 view_limit)."""
        values = []
        for idx in sorted(self.displayed_ids):
            if 0 <= idx < len(self.full_values):
                values.append(copy.deepcopy(self.full_values[idx]))
            if len(values) >= self.view_limit:
                break
        return values

    def init_sample(self) -> List[Dict]:
        """initial sampling: if the full amount <= the limit, return the full amount, otherwise randomly sample the limit amount."""
        if not self.full_values:
            return []

        if len(self.full_values) <= self.view_limit:
            sample_indices = list(range(len(self.full_values)))
        else:
            sample_indices = random.sample(range(len(self.full_values)), self.view_limit)

        self.displayed_ids.update(sample_indices)
        return [copy.deepcopy(self.full_values[i]) for i in sample_indices]

    def load_region(self, region: Optional[Dict]) -> List[Dict]:
        """
        incremental loading of the region: retain the displayed points in the region; if there are undisplayed points in the region and the view is not full, fill up to view_limit.
        if all the undisplayed points in the region have been displayed, do not forcefully fill the points (可能 < view_limit).
        """
        if not self.full_values:
            return []

        # when there is no region information, return the current displayed (or reinitialize)
        if not region:
            return self._current_displayed_values() or self.init_sample()

        candidates = []
        for idx, rec in enumerate(self.full_values):
            if self._point_in_region(rec, region):
                candidates.append((idx, rec))

        if not candidates:
            return []

        retained = [(idx, rec) for idx, rec in candidates if idx in self.displayed_ids]
        unseen = [(idx, rec) for idx, rec in candidates if idx not in self.displayed_ids]

        result: List[Dict] = [copy.deepcopy(rec) for _, rec in retained]

        if unseen and len(result) < self.view_limit:
            need = self.view_limit - len(result)
            chosen = unseen if len(unseen) <= need else random.sample(unseen, need)
            for idx, rec in chosen:
                self.displayed_ids.add(idx)
                result.append(copy.deepcopy(rec))

        return result
