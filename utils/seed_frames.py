"""Explicit seed frames for bidirectional propagation (workspace/seed_frames/)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from utils.mask_storage import _binary_channel_to_mask, load_all_masks_frame


def seed_frames_dir(workspace: Path, subdir: str = 'seed_frames') -> Path:
    return Path(workspace) / subdir


def list_submitted_seed_frames(workspace: Path, subdir: str = 'seed_frames') -> List[int]:
    """Return sorted frame indices with submitted seed NPZ files."""
    directory = seed_frames_dir(workspace, subdir)
    if not directory.exists():
        return []

    indices = set()
    for pattern in ('*.npz', '*.npy'):
        for path in directory.glob(pattern):
            try:
                indices.add(int(path.stem))
            except ValueError:
                pass
    return sorted(indices)


def _object_masks_to_multi_channel(
    object_masks: Dict[int, np.ndarray],
    num_objects: int,
    height: int,
    width: int,
) -> np.ndarray:
    multi = np.zeros((num_objects, height, width), dtype=np.uint8)
    for obj_id, binary in object_masks.items():
        if 1 <= obj_id <= num_objects:
            channel = _binary_channel_to_mask(binary)
            if channel.shape[:2] != (height, width):
                raise ValueError(
                    f'Object {obj_id} mask shape {channel.shape[:2]} '
                    f'does not match ({height}, {width})'
                )
            multi[obj_id - 1] = channel
    return multi


def submit_seed_frame(
    workspace: Path,
    frame_idx: int,
    object_masks: Dict[int, np.ndarray],
    num_objects: int,
    subdir: str = 'seed_frames',
) -> Path:
    """Save tracked object masks for a frame as seed_frames/{frame_idx:07d}.npz."""
    if not object_masks:
        raise ValueError('No object masks to submit')

    sample = next(iter(object_masks.values()))
    height, width = sample.shape[:2]
    multi = _object_masks_to_multi_channel(object_masks, num_objects, height, width)

    directory = seed_frames_dir(workspace, subdir)
    directory.mkdir(parents=True, exist_ok=True)
    out_path = directory / f'{frame_idx:07d}.npz'
    np.savez_compressed(str(out_path), mask=multi)
    return out_path


def load_submitted_seed_object_masks(
    workspace: Path,
    frame_idx: int,
    num_objects: int,
    tracked_object_ids: Optional[Set[int]] = None,
    subdir: str = 'seed_frames',
) -> Dict[int, np.ndarray]:
    """Load per-object binary masks from a submitted seed frame NPZ."""
    directory = seed_frames_dir(workspace, subdir)
    multi = load_all_masks_frame(directory, frame_idx)
    if multi is None:
        return {}

    result: Dict[int, np.ndarray] = {}
    n_channels = min(multi.shape[0], num_objects)
    for obj_id in range(1, n_channels + 1):
        if tracked_object_ids is not None and obj_id not in tracked_object_ids:
            continue
        binary = _binary_channel_to_mask(multi[obj_id - 1])
        if np.any(binary):
            result[obj_id] = binary
    return result
