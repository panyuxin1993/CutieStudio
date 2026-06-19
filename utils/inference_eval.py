"""Inline forward/backward robustness IoU evaluation during bidirectional inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


def compute_binary_mask_iou(forward_bin: np.ndarray, backward_bin: np.ndarray) -> float:
    """Binary IoU between two boolean/binary masks."""
    forward_bin = forward_bin > 0 if forward_bin.dtype != bool else forward_bin
    backward_bin = backward_bin > 0 if backward_bin.dtype != bool else backward_bin
    intersection = np.count_nonzero(forward_bin & backward_bin)
    union = np.count_nonzero(forward_bin | backward_bin)
    if union > 0:
        return float(intersection / union)
    return 0.0


def compute_frame_robustness_single_channel(
    forward_mask: np.ndarray,
    forward_object_ids: List[int],
    backward_mask: np.ndarray,
    backward_object_ids: List[int],
    object_ids: List[int],
) -> Tuple[Dict[str, Dict[str, float]], Dict[int, float]]:
    """Compute forward/backward overlap (IoU) for one frame."""
    forward_id_set = set(forward_object_ids)
    backward_id_set = set(backward_object_ids)

    frame_metrics = {
        'overlap_ratios': {},
        'present_in_forward': {},
        'present_in_backward': {},
    }
    overlap_ratios: Dict[int, float] = {}

    for obj_id in object_ids:
        present_forward = obj_id in forward_id_set
        present_backward = obj_id in backward_id_set
        frame_metrics['present_in_forward'][str(obj_id)] = present_forward
        frame_metrics['present_in_backward'][str(obj_id)] = present_backward

        if present_forward and present_backward:
            forward_bin = forward_mask == obj_id
            backward_bin = backward_mask == obj_id
            ratio = compute_binary_mask_iou(forward_bin, backward_bin)
            overlap_ratios[obj_id] = ratio
            frame_metrics['overlap_ratios'][str(obj_id)] = ratio
        else:
            frame_metrics['overlap_ratios'][str(obj_id)] = 0.0

    return frame_metrics, overlap_ratios


def select_mask_by_seed_proximity(
    frame_idx: int,
    seed_frames: List[int],
    forward_mask: np.ndarray,
    backward_mask: np.ndarray,
) -> np.ndarray:
    """Pick forward or backward single-channel mask by distance to bracketing seeds."""
    seeds = sorted(seed_frames)
    if not seeds:
        return forward_mask

    left = max((s for s in seeds if s <= frame_idx), default=seeds[0])
    right = min((s for s in seeds if s >= frame_idx), default=seeds[-1])

    if left == right:
        return forward_mask

    if frame_idx <= (left + right) // 2:
        return forward_mask
    return backward_mask


class RobustnessAccumulator:
    """Caches forward masks; computes per-object IoU when backward masks arrive."""

    def __init__(
        self,
        object_ids: List[int],
        seed_frames: Optional[List[int]] = None,
        eval_frame_indices: Optional[List[int]] = None,
    ):
        self.object_ids = sorted(object_ids)
        self.seed_frames = sorted(seed_frames or [])
        self._eval_frame_set: Optional[Set[int]] = (
            set(eval_frame_indices) if eval_frame_indices is not None else None
        )
        self._forward: Dict[int, np.ndarray] = {}
        self._backward: Dict[int, np.ndarray] = {}
        self._per_object_frame_iou: Dict[int, Dict[int, float]] = {
            obj_id: {} for obj_id in self.object_ids
        }

    def _should_eval_frame(self, frame_idx: int) -> bool:
        if self._eval_frame_set is None:
            return True
        return frame_idx in self._eval_frame_set

    def on_forward(self, frame_idx: int, mask_np: np.ndarray, object_ids: List[int]) -> None:
        if not self._should_eval_frame(frame_idx):
            return
        self._forward[frame_idx] = mask_np.copy()

    def on_backward(self, frame_idx: int, mask_np: np.ndarray, object_ids: List[int]) -> None:
        if not self._should_eval_frame(frame_idx):
            return
        self._backward[frame_idx] = mask_np.copy()

        forward_mask = self._forward.get(frame_idx)
        if forward_mask is None:
            return

        _, ratios = compute_frame_robustness_single_channel(
            forward_mask,
            object_ids,
            mask_np,
            object_ids,
            self.object_ids,
        )
        for obj_id, iou in ratios.items():
            self._per_object_frame_iou[obj_id][frame_idx] = iou

    def finalize_seed_frames(self, seed_object_ids: Dict[int, List[int]]) -> None:
        """Set IoU=1.0 on seed frames where both passes use the same annotation."""
        for frame_idx in self.seed_frames:
            if not self._should_eval_frame(frame_idx):
                continue
            obj_ids = seed_object_ids.get(frame_idx, self.object_ids)
            for obj_id in self.object_ids:
                if obj_id in obj_ids:
                    self._per_object_frame_iou[obj_id][frame_idx] = 1.0

    def get_forward_mask(self, frame_idx: int) -> Optional[np.ndarray]:
        return self._forward.get(frame_idx)

    def get_backward_mask(self, frame_idx: int) -> Optional[np.ndarray]:
        return self._backward.get(frame_idx)

    def get_per_object_frame_iou(self) -> Dict[int, Dict[int, float]]:
        return self._per_object_frame_iou

    def summary_mean_iou_per_object(self) -> Dict[int, float]:
        summary = {}
        for obj_id in self.object_ids:
            values = list(self._per_object_frame_iou[obj_id].values())
            summary[obj_id] = float(np.mean(values)) if values else 0.0
        return summary


def build_iou_table(
    object_ids: List[int],
    frame_indices: List[int],
    per_object_frame_iou: Dict[int, Dict[int, float]],
    object_names: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """Build objects × frames IoU matrix."""
    object_names = object_names or {}
    rows = []
    for obj_id in sorted(object_ids):
        row_name = object_names.get(obj_id, f'object_{obj_id}')
        row = {'object_id': obj_id, 'object_name': row_name}
        for frame_idx in frame_indices:
            row[str(frame_idx)] = per_object_frame_iou.get(obj_id, {}).get(frame_idx, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def save_iou_table(
    df: pd.DataFrame,
    workspace: Path,
    csv_name: str = 'iou_table.csv',
    save_json: bool = True,
) -> Path:
    """Write IoU table CSV (and optional long-format JSON) to workspace."""
    workspace = Path(workspace)
    csv_path = workspace / csv_name
    df.to_csv(csv_path, index=False)

    if save_json:
        long_rows = []
        for _, row in df.iterrows():
            obj_id = int(row['object_id'])
            obj_name = row.get('object_name', f'object_{obj_id}')
            for col in df.columns:
                if col in ('object_id', 'object_name'):
                    continue
                val = row[col]
                if pd.isna(val):
                    continue
                long_rows.append({
                    'object_id': obj_id,
                    'object_name': obj_name,
                    'frame': int(col),
                    'iou': float(val),
                })
        json_path = workspace / csv_name.replace('.csv', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(long_rows, f, indent=2)

    return csv_path


def load_iou_table(csv_path: Path) -> Optional[Dict]:
    """Load wide-format IoU table CSV written by save_iou_table / build_iou_table."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if df.empty or 'object_id' not in df.columns:
        return None

    frame_cols = [c for c in df.columns if c not in ('object_id', 'object_name')]
    frame_indices = sorted(int(c) for c in frame_cols)

    object_ids: List[int] = []
    object_names: Dict[int, str] = {}
    values: Dict[int, Dict[int, float]] = {}

    for _, row in df.iterrows():
        obj_id = int(row['object_id'])
        object_ids.append(obj_id)
        if 'object_name' in df.columns and not pd.isna(row.get('object_name')):
            object_names[obj_id] = str(row['object_name'])
        obj_values: Dict[int, float] = {}
        for frame_idx in frame_indices:
            val = row.get(str(frame_idx), np.nan)
            if pd.isna(val):
                continue
            obj_values[frame_idx] = float(val)
        values[obj_id] = obj_values

    if not object_ids or not frame_indices:
        return None

    return {
        'object_ids': sorted(object_ids),
        'object_names': object_names,
        'frame_indices': frame_indices,
        'values': values,
    }


def frames_below_iou_threshold(data: Dict, threshold: float) -> List[int]:
    """Return frame indices where at least one object has IoU strictly below threshold."""
    if not data:
        return []
    below: Set[int] = set()
    for frame_vals in data.get('values', {}).values():
        for frame_idx, iou in frame_vals.items():
            if iou < threshold:
                below.add(int(frame_idx))
    return sorted(below)


def merge_iou_values(
    base: Dict[int, Dict[int, float]],
    updates: Dict[int, Dict[int, float]],
) -> Dict[int, Dict[int, float]]:
    """Merge per-object frame IoU maps, with updates overwriting base."""
    merged = {obj_id: dict(frames) for obj_id, frames in base.items()}
    for obj_id, frame_vals in updates.items():
        if obj_id not in merged:
            merged[obj_id] = {}
        merged[obj_id].update(frame_vals)
    return merged
