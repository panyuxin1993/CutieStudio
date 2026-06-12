"""Load per-object masks from workspace all_masks (preferred), soft_masks, or inference masks/."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

_inference_mask_path_cache: Dict[str, Dict[int, Path]] = {}


def resolve_mask_workspace(mask_dir: Path) -> Dict[str, Path]:
    mask_dir = Path(mask_dir)
    workspace = mask_dir.parent
    return {
        'workspace': workspace,
        'mask_dir': mask_dir,
        'all_masks_dir': workspace / 'all_masks',
        'soft_mask_dir': workspace / 'soft_masks',
    }


def load_all_masks_frame(all_masks_dir: Path, frame_idx: int) -> Optional[np.ndarray]:
    """Load (num_objects, H, W) uint8 mask array from .npz or legacy .npy."""
    all_masks_dir = Path(all_masks_dir)
    npz_path = all_masks_dir / f'{frame_idx:07d}.npz'
    npy_path = all_masks_dir / f'{frame_idx:07d}.npy'

    if npz_path.exists():
        try:
            return np.load(npz_path)['mask']
        except Exception:
            return None
    if npy_path.exists():
        try:
            return np.load(npy_path)
        except Exception:
            return None
    return None


def _get_inference_mask_index(mask_dir: Path) -> Dict[int, Path]:
    """Map frame index -> masks/*.png path (cached per directory)."""
    key = str(Path(mask_dir).resolve())
    if key not in _inference_mask_path_cache:
        index: Dict[int, Path] = {}
        for f in Path(mask_dir).glob('*.png'):
            try:
                index[int(f.stem)] = f
            except ValueError:
                pass
        _inference_mask_path_cache[key] = index
    return _inference_mask_path_cache[key]


def resolve_inference_mask_path(mask_dir: Path, frame_idx: int) -> Optional[Path]:
    """Resolve inference mask PNG path for a frame index."""
    return _get_inference_mask_index(mask_dir).get(frame_idx)


def list_frame_indices(all_masks_dir: Path, soft_mask_dir: Path,
                       mask_dir: Optional[Path] = None) -> List[int]:
    """List frame indices from all_masks, soft_masks, and inference masks/."""
    indices = set()
    all_masks_dir = Path(all_masks_dir)
    soft_mask_dir = Path(soft_mask_dir)

    if all_masks_dir.exists():
        for pattern in ('*.npz', '*.npy'):
            for f in all_masks_dir.glob(pattern):
                try:
                    indices.add(int(f.stem))
                except ValueError:
                    pass

    if soft_mask_dir.exists():
        for obj_dir in soft_mask_dir.iterdir():
            if obj_dir.is_dir():
                for f in obj_dir.glob('*.png'):
                    try:
                        indices.add(int(f.stem))
                    except ValueError:
                        pass

    if mask_dir is not None:
        indices.update(_get_inference_mask_index(mask_dir).keys())

    return sorted(indices)


def _binary_channel_to_mask(channel: np.ndarray) -> np.ndarray:
    if channel.dtype == np.bool_:
        return channel.astype(np.uint8)
    return (channel > 0).astype(np.uint8)


def load_object_masks_from_inference_mask(mask_dir: Path, frame_idx: int,
                                          num_objects: int) -> Dict[int, np.ndarray]:
    """Extract per-object binary masks directly from inference masks/ PNG (palette IDs).

    Faster than building an all_masks array: one file read plus per-object equality tests.
    """
    mask_path = resolve_inference_mask_path(mask_dir, frame_idx)
    if mask_path is None:
        return {}

    id_mask = np.array(Image.open(mask_path))
    result: Dict[int, np.ndarray] = {}
    for obj_id in np.unique(id_mask):
        obj_id = int(obj_id)
        if obj_id <= 0 or obj_id > num_objects:
            continue
        binary = (id_mask == obj_id).astype(np.uint8)
        if np.any(binary):
            result[obj_id] = binary
    return result


def load_object_mask_from_soft_masks(soft_mask_dir: Path, frame_idx: int,
                                     obj_id: int) -> Optional[np.ndarray]:
    mask_path = soft_mask_dir / str(obj_id) / f'{frame_idx:07d}.png'
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return (mask > 127).astype(np.uint8)


def load_frame_object_masks(workspace: Path, frame_idx: int,
                            num_objects: int) -> Dict[int, np.ndarray]:
    """Load binary masks for each object that has data in this frame."""
    workspace = Path(workspace)
    all_masks_dir = workspace / 'all_masks'
    soft_mask_dir = workspace / 'soft_masks'
    result: Dict[int, np.ndarray] = {}

    multi = load_all_masks_frame(all_masks_dir, frame_idx)
    if multi is not None:
        n_channels = min(multi.shape[0], num_objects)
        for obj_id in range(1, n_channels + 1):
            binary = _binary_channel_to_mask(multi[obj_id - 1])
            if np.any(binary):
                result[obj_id] = binary
        if result:
            return result

    if soft_mask_dir.exists():
        for obj_id in range(1, num_objects + 1):
            mask = load_object_mask_from_soft_masks(soft_mask_dir, frame_idx, obj_id)
            if mask is not None and np.any(mask):
                result[obj_id] = mask
        if result:
            return result

    inference_mask_dir = workspace / 'masks'
    if inference_mask_dir.exists():
        result = load_object_masks_from_inference_mask(
            inference_mask_dir, frame_idx, num_objects)
        if result:
            return result

    return result


def mask_storage_available(mask_dir: Path) -> bool:
    dirs = resolve_mask_workspace(mask_dir)
    return bool(list_frame_indices(
        dirs['all_masks_dir'], dirs['soft_mask_dir'], dirs['mask_dir']))


def get_frame_indices_for_workspace(mask_dir: Path, num_objects: int) -> Tuple[int, ...]:
    dirs = resolve_mask_workspace(mask_dir)
    return tuple(list_frame_indices(
        dirs['all_masks_dir'], dirs['soft_mask_dir'], dirs['mask_dir']))
