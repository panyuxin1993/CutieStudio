"""Bidirectional gap filling from annotated seed frames using CUTIE inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image

from cutie.inference.inference_core import InferenceCore
from cutie.model.cutie import CUTIE
from gui.interactive_utils import image_to_torch
from utils.inference_eval import (
    RobustnessAccumulator,
    build_iou_table,
    load_iou_table,
    merge_iou_values,
    save_iou_table,
    select_mask_by_seed_proximity,
)
from utils.mask_transformer import MaskTransformer
from utils.seed_frames import list_submitted_seed_frames, load_submitted_seed_object_masks

log = logging.getLogger(__name__)

SeedMaskSource = Union[str, Path, np.ndarray]


def index_numpy_to_one_hot_torch(mask: np.ndarray, num_classes: int):
    mask = torch.from_numpy(mask).long()
    return F.one_hot(mask, num_classes=num_classes).permute(2, 0, 1).float()


def check_to_clear_non_permanent_cuda_memory(
    processor: InferenceCore, device, mem_cleanup_ratio: float
) -> None:
    if 'cuda' not in str(device):
        return
    if mem_cleanup_ratio <= 0 or mem_cleanup_ratio > 1:
        return
    info = torch.cuda.mem_get_info()
    global_free, global_total = info
    global_used = (global_total - global_free) / global_total
    if global_used > mem_cleanup_ratio:
        processor.clear_non_permanent_memory()
        torch.cuda.empty_cache()


class WorkspaceFrameReader:
    """Read frames from workspace/images/{frame_idx:07d}.jpg|.png."""

    def __init__(self, image_dir: Path, total_frames: int):
        self.image_dir = Path(image_dir)
        self.total_frames = total_frames
        self.current_frame_index = 0

    def set_frame(self, frame_idx: int) -> bool:
        if 0 <= frame_idx < self.total_frames:
            self.current_frame_index = frame_idx
            return True
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        stem = f'{self.current_frame_index:07d}'
        for ext in ('.jpg', '.png', '.jpeg'):
            path = self.image_dir / f'{stem}{ext}'
            if path.exists():
                frame = cv2.imread(str(path))
                if frame is not None:
                    self.current_frame_index += 1
                    return True, frame
        return False, None

    def release(self) -> None:
        pass


def object_masks_to_indexed(
    object_masks: Dict[int, np.ndarray],
    height: int,
    width: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for obj_id, binary in sorted(object_masks.items()):
        mask[binary > 0] = obj_id
    return mask


def discover_seed_frames(workspace: Path, num_objects: int) -> List[int]:
    """List submitted seed frames (legacy name kept for callers)."""
    return list_submitted_seed_frames(workspace)


def build_seed_masks(
    workspace: Path,
    seed_frames: List[int],
    num_objects: int,
    tracked_object_ids: Optional[Set[int]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    seed_frames_subdir: str = 'seed_frames',
) -> Tuple[Dict[int, np.ndarray], Dict[int, List[int]]]:
    """Build {frame_idx: single-channel mask} from submitted seed_frames/."""
    workspace = Path(workspace)
    seed_sources: Dict[int, np.ndarray] = {}
    seed_object_ids: Dict[int, List[int]] = {}

    for frame_idx in seed_frames:
        object_masks = load_submitted_seed_object_masks(
            workspace,
            frame_idx,
            num_objects,
            tracked_object_ids=tracked_object_ids,
            subdir=seed_frames_subdir,
        )
        if not object_masks:
            continue

        if height is None or width is None:
            sample = next(iter(object_masks.values()))
            height, width = sample.shape[:2]

        indexed = object_masks_to_indexed(object_masks, height, width)
        seed_sources[frame_idx] = indexed
        seed_object_ids[frame_idx] = sorted(object_masks.keys())

    return seed_sources, seed_object_ids


@dataclass
class BidirectionalFillResult:
    iou_df: pd.DataFrame
    iou_table_path: Path
    forward_dir: Path
    backward_dir: Path
    seed_frames: List[int]
    frame_indices: List[int]
    object_ids: List[int]
    mean_iou_per_object: Dict[int, float] = field(default_factory=dict)
    merged_frames_written: int = 0


class CutieGapFiller:
    """Segment-wise forward/backward propagation from multiple seed frames."""

    def __init__(
        self,
        mask_transformer: MaskTransformer,
        cfg: DictConfig,
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or log
        self.mask_transformer = mask_transformer
        self.cfg = cfg

        if device is None:
            if cfg.get('device') == 'cuda' and torch.cuda.is_available():
                device = 'cuda'
            elif cfg.get('device') == 'mps' and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = torch.device(device)

        self.model = CUTIE(cfg).to(self.device).eval()
        if cfg.weights is not None:
            weights = torch.load(cfg.weights, map_location=self.device)
            self.model.load_weights(weights)

        self.processor = InferenceCore(self.model, cfg=cfg)
        self.processor.max_internal_size = cfg.get('max_internal_size', 480)
        self.mem_cleanup_ratio = cfg.get('mem_cleanup_ratio', 0.8)
        torch.set_grad_enabled(False)

    def _load_seed_mask(self, source: SeedMaskSource) -> Tuple[np.ndarray, Optional[str]]:
        if isinstance(source, np.ndarray):
            return source, None
        img = Image.open(source)
        return np.array(img), img.mode

    def _resolve_mask_save_format(self, first_source: SeedMaskSource) -> str:
        if isinstance(first_source, np.ndarray):
            return 'davis'
        mode = Image.open(first_source).mode
        if mode in ('P', 'RGB'):
            return 'davis'
        if mode == 'L':
            return 'binary'
        raise ValueError(f'Unknown mode {mode} in {first_source}')

    @staticmethod
    def _build_target_frames(
        start_frame: int,
        end_frame: int,
        step: int,
        seed_frames: List[int],
    ) -> List[int]:
        if step == 1:
            return list(range(start_frame, end_frame + 1))
        targets = {f for f in range(0, end_frame + step, step) if start_frame <= f <= end_frame}
        targets.update((start_frame, end_frame))
        targets.update(f for f in seed_frames if start_frame <= f <= end_frame)
        return sorted(targets)

    def _process_direction(
        self,
        frame_reader: WorkspaceFrameReader,
        seed_sources: Dict[int, np.ndarray],
        output_dir: Path,
        inference_mode: str,
        start_frame: int,
        end_frame: int,
        step: int,
        accumulator: RobustnessAccumulator,
        mask_save_format: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        frames_done: int = 0,
        total_frames: int = 0,
        output_frames: Optional[Set[int]] = None,
    ) -> int:
        self.processor.clear_memory()

        seed_frames = sorted(seed_sources.keys())
        first_source = seed_sources[seed_frames[0]]

        target_frames = self._build_target_frames(start_frame, end_frame, step, seed_frames)
        if inference_mode == 'backward':
            frame_indices = sorted(target_frames, reverse=True)
            seed_frames_ordered = sorted(seed_frames, reverse=True)
        else:
            frame_indices = target_frames
            seed_frames_ordered = sorted(seed_frames)

        processed = 0
        use_amp = self.cfg.get('amp', True)

        with torch.inference_mode():
            with torch.amp.autocast(
                self.device.type,
                enabled=(use_amp and self.device.type == 'cuda'),
            ):
                for segment_idx, current_seed_frame in enumerate(seed_frames_ordered):
                    self.processor.clear_memory()
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()

                    if inference_mode == 'backward':
                        if segment_idx < len(seed_frames_ordered) - 1:
                            next_seed_frame = seed_frames_ordered[segment_idx + 1]
                            segment_targets = [
                                f for f in frame_indices
                                if next_seed_frame < f <= current_seed_frame
                            ]
                        else:
                            segment_targets = [
                                f for f in frame_indices if f <= current_seed_frame
                            ]
                    else:
                        if segment_idx < len(seed_frames_ordered) - 1:
                            next_seed_frame = seed_frames_ordered[segment_idx + 1]
                            segment_targets = [
                                f for f in frame_indices
                                if current_seed_frame <= f < next_seed_frame
                            ]
                        else:
                            segment_targets = [
                                f for f in frame_indices if f >= current_seed_frame
                            ]

                    if not segment_targets:
                        continue

                    seed_source = seed_sources.get(current_seed_frame)
                    if seed_source is None:
                        continue

                    frame_reader.set_frame(current_seed_frame)
                    ret, frame = frame_reader.read()
                    if not ret:
                        continue

                    frame_torch = image_to_torch(frame, device=str(self.device))
                    mask_np, _ = self._load_seed_mask(seed_source)
                    unique_ids = np.unique(mask_np)
                    object_ids = [oid for oid in unique_ids if oid > 0]
                    if not object_ids:
                        continue

                    id_map = {oid: idx + 1 for idx, oid in enumerate(object_ids)}
                    remapped_mask = np.zeros_like(mask_np)
                    for orig_id, new_id in id_map.items():
                        remapped_mask[mask_np == orig_id] = new_id
                    num_objects_segment = len(object_ids)
                    mask_torch = index_numpy_to_one_hot_torch(
                        remapped_mask, num_objects_segment + 1
                    ).to(self.device)

                    self.processor.step(
                        frame_torch, mask_torch[1:], idx_mask=False, force_permanent=True
                    )

                    for idx in segment_targets:
                        frame_reader.set_frame(idx)
                        ret, frame = frame_reader.read()
                        if not ret:
                            continue

                        frame_torch = image_to_torch(frame, device=str(self.device))
                        frame_object_ids = None
                        reverse_map = None

                        if idx in seed_sources:
                            mask_np, _ = self._load_seed_mask(seed_sources[idx])
                            unique_ids = np.unique(mask_np)
                            frame_object_ids = [oid for oid in unique_ids if oid > 0]
                            if frame_object_ids:
                                frame_id_map = {
                                    oid: i + 1 for i, oid in enumerate(frame_object_ids)
                                }
                                reverse_map = {v: k for k, v in frame_id_map.items()}
                                remapped = np.zeros_like(mask_np)
                                for orig_id, new_id in frame_id_map.items():
                                    remapped[mask_np == orig_id] = new_id
                                num_objects_frame = len(frame_object_ids)
                                mask_torch = index_numpy_to_one_hot_torch(
                                    remapped, num_objects_frame + 1
                                ).to(self.device)
                                prob = self.processor.step(
                                    frame_torch, mask_torch[1:], idx_mask=False
                                )
                            else:
                                prob = self.processor.step(frame_torch)
                        else:
                            prob = self.processor.step(frame_torch)

                        mask = torch.argmax(prob, dim=0)
                        mask_np = mask.cpu().numpy()

                        if reverse_map is not None:
                            original_mask = np.zeros_like(mask_np)
                            for new_id, orig_id in reverse_map.items():
                                original_mask[mask_np == new_id] = orig_id
                            mask_np = original_mask

                        save_object_ids = (
                            frame_object_ids if frame_object_ids is not None else object_ids
                        )
                        effective_ids = save_object_ids or list(range(1, num_objects_segment + 1))

                        record_frame = (
                            output_frames is None or idx in output_frames
                        )
                        if record_frame:
                            if inference_mode == 'forward':
                                accumulator.on_forward(idx, mask_np, effective_ids)
                            else:
                                accumulator.on_backward(idx, mask_np, effective_ids)

                            self.mask_transformer.save_masks(
                                mask_np,
                                output_dir,
                                idx,
                                format_type=mask_save_format,
                                object_ids=effective_ids,
                            )

                        check_to_clear_non_permanent_cuda_memory(
                            self.processor, self.device, self.mem_cleanup_ratio
                        )

                        processed += 1
                        if progress_callback is not None:
                            progress_callback(
                                frames_done + processed,
                                total_frames,
                                f'{inference_mode} frame {idx}',
                            )

        return processed

    def process_direction(
        self,
        workspace: Path,
        seed_sources: Dict[int, np.ndarray],
        output_dir: Path,
        inference_mode: str,
        total_frames: int,
        start_frame: int,
        end_frame: int,
        accumulator: RobustnessAccumulator,
        step: int = 1,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        frames_done: int = 0,
        total_work: int = 0,
        output_frames: Optional[Set[int]] = None,
    ) -> int:
        workspace = Path(workspace)
        image_dir = workspace / 'images'
        frame_reader = WorkspaceFrameReader(image_dir, total_frames)
        first_source = seed_sources[sorted(seed_sources.keys())[0]]
        mask_save_format = self._resolve_mask_save_format(first_source)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        return self._process_direction(
            frame_reader,
            seed_sources,
            output_dir,
            inference_mode,
            start_frame,
            end_frame,
            step,
            accumulator,
            mask_save_format,
            progress_callback,
            frames_done,
            total_work,
            output_frames=output_frames,
        )


def run_bidirectional_gap_fill(
    workspace: Path,
    cfg: DictConfig,
    *,
    seed_frames: Optional[List[int]] = None,
    frame_range: Optional[Tuple[int, int]] = None,
    tracked_object_ids: Optional[Set[int]] = None,
    object_names: Optional[Dict[int, str]] = None,
    num_objects: Optional[int] = None,
    forward_subdir: str = 'forward_masks',
    backward_subdir: str = 'backward_masks',
    iou_table_name: str = 'iou_table.csv',
    merge_strategy: str = 'seed_proximity',
    overwrite_inferred: bool = True,
    step: int = 1,
    output_frames: Optional[Set[int]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> BidirectionalFillResult:
    """Run forward + backward inference from annotated seeds; merge and save IoU table."""
    workspace = Path(workspace)
    num_objects = num_objects or cfg.get('num_objects', 12)

    if seed_frames is None:
        seed_frames = list_submitted_seed_frames(
            workspace,
            cfg.get('bidirectional', {}).get('seed_frames_subdir', 'seed_frames'),
        )
    seed_frames = sorted(set(seed_frames))

    if not seed_frames:
        raise ValueError(
            'No submitted seed frames found. Use Submit seed frame on key frames first.'
        )

    image_dir = workspace / 'images'
    image_files = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
    total_frames = len(image_files)
    if total_frames == 0:
        raise ValueError(f'No images found in {image_dir}')

    if frame_range is None:
        start_frame, end_frame = 0, total_frames - 1
    else:
        start_frame, end_frame = frame_range

    seed_sources, seed_object_ids = build_seed_masks(
        workspace,
        seed_frames,
        num_objects,
        tracked_object_ids=tracked_object_ids,
        seed_frames_subdir=cfg.get('bidirectional', {}).get('seed_frames_subdir', 'seed_frames'),
    )
    if not seed_sources:
        raise ValueError('Seed frames have no valid masks for tracked objects')

    all_object_ids: Set[int] = set()
    for obj_ids in seed_object_ids.values():
        all_object_ids.update(obj_ids)
    object_ids = sorted(all_object_ids)

    target_frames = CutieGapFiller._build_target_frames(
        start_frame, end_frame, step, list(seed_sources.keys())
    )
    if output_frames is not None:
        eval_frames = sorted(f for f in output_frames if start_frame <= f <= end_frame)
        if not eval_frames:
            raise ValueError('No output frames fall within the selected frame range')
    else:
        eval_frames = [f for f in target_frames if start_frame <= f <= end_frame]

    existing_iou = load_iou_table(workspace / iou_table_name)

    forward_dir = workspace / forward_subdir
    backward_dir = workspace / backward_subdir
    mask_dir = workspace / 'masks'

    mask_transformer = MaskTransformer(num_objects=num_objects)
    gap_filler = CutieGapFiller(mask_transformer, cfg)

    accumulator = RobustnessAccumulator(
        object_ids=object_ids,
        seed_frames=list(seed_sources.keys()),
        eval_frame_indices=eval_frames,
    )

    total_work = len(eval_frames) * 2
    frames_done = 0

    if progress_callback:
        progress_callback(0, total_work, 'Starting forward pass...')

    frames_done += gap_filler.process_direction(
        workspace,
        seed_sources,
        forward_dir,
        'forward',
        total_frames,
        start_frame,
        end_frame,
        accumulator,
        step=step,
        progress_callback=progress_callback,
        frames_done=frames_done,
        total_work=total_work,
        output_frames=output_frames,
    )

    if progress_callback:
        progress_callback(frames_done, total_work, 'Starting backward pass...')

    frames_done += gap_filler.process_direction(
        workspace,
        seed_sources,
        backward_dir,
        'backward',
        total_frames,
        start_frame,
        end_frame,
        accumulator,
        step=step,
        progress_callback=progress_callback,
        frames_done=frames_done,
        total_work=total_work,
        output_frames=output_frames,
    )

    accumulator.finalize_seed_frames(seed_object_ids)

    if merge_strategy != 'seed_proximity':
        raise ValueError(f'Unsupported merge_strategy: {merge_strategy}')

    seed_set = set(seed_sources.keys())
    merged_written = 0
    mask_save_format = 'davis'

    if overwrite_inferred:
        mask_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx in eval_frames:
            if frame_idx in seed_set:
                continue

            forward_mask = accumulator.get_forward_mask(frame_idx)
            backward_mask = accumulator.get_backward_mask(frame_idx)
            if forward_mask is None or backward_mask is None:
                continue

            merged = select_mask_by_seed_proximity(
                frame_idx,
                list(seed_sources.keys()),
                forward_mask,
                backward_mask,
            )

            present_ids = sorted(set(np.unique(merged)) - {0})
            if not present_ids:
                continue

            mask_transformer.save_masks(
                merged,
                mask_dir,
                frame_idx,
                format_type=mask_save_format,
                object_ids=present_ids,
            )
            merged_written += 1

    per_object_iou = accumulator.get_per_object_frame_iou()
    if existing_iou is not None:
        per_object_iou = merge_iou_values(existing_iou['values'], per_object_iou)
        frame_indices = sorted(
            set(existing_iou.get('frame_indices', [])) | set(eval_frames)
        )
    else:
        frame_indices = eval_frames

    iou_df = build_iou_table(
        object_ids,
        frame_indices,
        per_object_iou,
        object_names=object_names,
    )
    iou_path = save_iou_table(iou_df, workspace, csv_name=iou_table_name)

    if progress_callback:
        progress_callback(total_work, total_work, 'Done')

    return BidirectionalFillResult(
        iou_df=iou_df,
        iou_table_path=iou_path,
        forward_dir=forward_dir,
        backward_dir=backward_dir,
        seed_frames=list(seed_sources.keys()),
        frame_indices=eval_frames,
        object_ids=object_ids,
        mean_iou_per_object=accumulator.summary_mean_iou_per_object(),
        merged_frames_written=merged_written,
    )
