'''
https://github.com/hkchengrex/Cutie/blob/main/scripts/process_video.py
'''

from os import path, listdir
from omegaconf import DictConfig, open_dict
from hydra import compose, initialize

import torch

from cutie.model.cutie import CUTIE
from cutie.inference.inference_core import InferenceCore
from cutie.inference.utils.results_utils import ResultSaver

from tqdm import tqdm

from time import perf_counter
import cv2
from gui.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch
import numpy as np
from PIL import Image

from argparse import ArgumentParser
from pathlib import Path
import os

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from utils import MaskTransformer

class FrameReader:
    """Abstract base class for reading frames."""
    def __init__(self):
        self.current_frame_index = 0
        self.total_frames = 0

    def read(self):
        """Read next frame."""
        raise NotImplementedError

    def set_frame(self, frame_idx):
        """Set current frame index."""
        raise NotImplementedError

    def release(self):
        """Release resources."""
        pass

class VideoReader(FrameReader):
    """Read frames from video file."""
    def __init__(self, video_path):
        super().__init__()
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f'Unable to open video {video_path}!')
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_index += 1
        return ret, frame

    def set_frame(self, frame_idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self.current_frame_index = frame_idx

    def release(self):
        self.cap.release()

class DirectoryReader(FrameReader):
    """Read frames from directory."""
    def __init__(self, frames_dir):
        super().__init__()
        self.frames_dir = Path(frames_dir)
        # Get sorted frame paths
        self.frame_paths = sorted(list(self.frames_dir.glob('*.jpg')) + list(self.frames_dir.glob('*.png')))
        self.total_frames = len(self.frame_paths)
        if self.total_frames == 0:
            raise ValueError(f'No frames found in {frames_dir}!')

    def read(self):
        if self.current_frame_index >= self.total_frames:
            return False, None
        frame = cv2.imread(str(self.frame_paths[self.current_frame_index]))
        self.current_frame_index += 1
        return True, frame

    def set_frame(self, frame_idx):
        if 0 <= frame_idx < self.total_frames:
            self.current_frame_index = frame_idx
            return True
        return False

def extract_object_masks_with_transformer(mask_transformer, mask_dir, output_dir, object_id):
    """Extract binary masks for a single object using MaskTransformer."""
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for mask_file in mask_dir.glob('*.png'):
        mask, _, _ = mask_transformer.read_mask(str(mask_file))
        if mask is None:
            continue
        binary_mask = (mask == object_id).astype(np.uint8) * 255
        output_file = output_dir / mask_file.name
        cv2.imwrite(str(output_file), binary_mask)

def process_video(cfg: DictConfig):
    # general setup
    torch.set_grad_enabled(False)
    if cfg['device'] == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    elif cfg['device'] == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Using device: {device}')

    use_amp = cfg.amp

    # Load the network weights
    print(f'Loading Cutie and weights')
    cutie = CUTIE(cfg).to(device).eval()
    if cfg.weights is not None:
        model_weights = torch.load(cfg.weights, map_location=device)
        cutie.load_weights(model_weights)
    else:
        print('No model weights loaded. Are you sure about this?')

    # Initialize frame reader based on input type
    input_type = cfg.get('input_type', 'video')
    # input_type = 'frames'
    if input_type == 'video':
        video = cfg['video']
        if video is None:
            print('No video defined. Please specify!')
            exit()
        frame_reader = VideoReader(video)
        video_name = path.splitext(video)[0]
    else:  # frames
        frames_dir = cfg.get('frames_dir')
        if frames_dir is None:
            print('No frames directory defined. Please specify!')
            exit()
        frame_reader = DirectoryReader(frames_dir)
        video_name = Path(frames_dir).name

    total_frame_count = frame_reader.total_frames

    # Initial mask handling
    mask_dir = cfg['mask_dir']
    if mask_dir is None:
        print('No mask_dir defined. Please specify!')
        exit()

    # determine if the mask uses 3-channel long ID or 1-channel (0~255) short ID
    all_mask_frames = sorted(listdir(mask_dir))
    first_mask_frame = all_mask_frames[0]
    first_mask = Image.open(path.join(mask_dir, first_mask_frame))

    if first_mask.mode == 'P':
        save_format = 'davis'
        use_long_id = False
        palette = first_mask.getpalette()
    elif first_mask.mode == 'RGB':
        save_format = 'davis'
        use_long_id = True
        palette = None
    elif first_mask.mode == 'L':
        save_format = 'binary'
        use_long_id = False
        palette = None
    else:
        print(f'Unknown mode {first_mask.mode} in {first_mask_frame}.')
        exit()

    num_objects = cfg['num_objects']
    if num_objects is None or num_objects < 1:
        num_objects = len(np.unique(first_mask)) - 1

    processor = InferenceCore(cutie, cfg=cfg)
     # Initialize MaskTransformer
    mask_transformer = MaskTransformer(num_objects=cfg['num_objects'])

    # First commit mask input into permanent memory
    num_masks = len(all_mask_frames)
    if num_masks == 0:
        print(f'No mask frames found!')
        exit()

    mode = cfg.get('mode', 'simultaneous')
    if mode == 'sequential':
        # Get all object IDs from the first mask
        all_mask_frames = sorted(listdir(mask_dir))
        first_mask = np.array(Image.open(os.path.join(mask_dir, all_mask_frames[0])))
        object_ids = [id for id in np.unique(first_mask) if id > 0]
        soft_masks_dir = os.path.join(cfg['output_dir'], 'soft_masks')
        Path(soft_masks_dir).mkdir(exist_ok=True)
        # 1. Extract and process each object
        for obj_id in object_ids:
            obj_mask_dir = os.path.join(soft_masks_dir, str(obj_id))
            extract_object_masks_with_transformer(mask_transformer, Path(mask_dir), Path(obj_mask_dir), obj_id)
            # Now run inference for this object
            # Use obj_mask_dir as the mask_dir, output to obj_mask_dir
            # Use the same process as before, but only for this object
            # Set up a temporary config for this object
            temp_cfg = cfg.copy()
            temp_cfg['mask_dir'] = obj_mask_dir
            temp_cfg['output_dir'] = obj_mask_dir
            temp_cfg['mode'] = 'simultaneous'  # Use normal inference for each object
            temp_cfg['num_objects'] = 1
            process_video(temp_cfg)
        # 2. Merge per-object masks
        final_output_dir = os.path.join(cfg['output_dir'], 'sequential_merged')
        Path(final_output_dir).mkdir(exist_ok=True)
        # Determine number of frames
        all_obj_dirs = [Path(soft_masks_dir) / str(obj_id) for obj_id in object_ids]
        frame_indices = set()
        for obj_dir in all_obj_dirs:
            frame_indices.update([int(f.stem) for f in obj_dir.glob('*.png')])
        frame_indices = sorted(frame_indices)
        for frame_idx in frame_indices:
            masks = []
            present_ids = []
            for obj_id in object_ids:
                mask_path = Path(soft_masks_dir) / str(obj_id) / f"{frame_idx:07d}.png"
                if mask_path.exists():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        masks.append((mask > 127).astype(np.float32))
                        present_ids.append(obj_id)
                    else:
                        print(f"No mask found for object {obj_id} at frame {frame_idx}")
            if masks:
                multi_mask = np.stack(masks, axis=0)
                combined_mask = mask_transformer.multi_to_single_channel(multi_mask, present_ids)
                # Save using mask_transformer instead of ResultSaver
                mask_transformer.save_masks(
                    combined_mask,
                    final_output_dir,
                    frame_idx,
                    format_type=save_format,
                    object_ids=present_ids
                )
        print(f"Sequential mode complete. Merged masks saved to {final_output_dir}")
        return

    with torch.inference_mode():
        with torch.amp.autocast(device, enabled=(use_amp and device == 'cuda')):
            pbar = tqdm(total=num_masks)
            pbar.set_description('Commiting masks into permenent memory')
            for mask_name in all_mask_frames:
                mask = Image.open(path.join(mask_dir, mask_name))
                frame_number = int(mask_name[:-4])
                frame_reader.set_frame(frame_number)

                # load frame matching mask
                ret, frame = frame_reader.read()
                if not ret:
                    break

                # convert numpy array to pytorch tensor format
                frame_torch = image_to_torch(frame, device=device)
                
                mask_np = np.array(mask)
                unique_ids = np.unique(mask_np)
                object_ids = [id for id in unique_ids if id > 0]
                id_map = {id: idx+1 for idx, id in enumerate(object_ids)}  # 0 stays 0 (background)
                remapped_mask = np.zeros_like(mask_np)
                for orig_id, new_id in id_map.items():
                    remapped_mask[mask_np == orig_id] = new_id
                num_objects = len(object_ids)
                mask_torch = index_numpy_to_one_hot_torch(remapped_mask, num_objects + 1).to(device)

                # the background mask is fed into the model
                prob = processor.step(frame_torch,
                                      mask_torch[1:],
                                      idx_mask=False,
                                      force_permanent=True)

                pbar.update(1)

    # Next start inference on video
    frame_reader.set_frame(0)  # reset frame reading
    total_process_time = 0
    current_frame_index = 0
    mask_output_root = cfg['output_dir']
    mem_cleanup_ratio = cfg['mem_cleanup_ratio']

    if cfg['direction'] == 'backward':
        # Find the highest-indexed mask frame
        all_mask_frames = sorted(listdir(mask_dir))
        mask_indices = [int(f[:-4]) for f in all_mask_frames if f.endswith('.png')]
        if not mask_indices:
            raise ValueError("No mask frames found!")
        start_idx = max(mask_indices)
        frame_indices = range(start_idx, -1, -1)
    else:
        frame_indices = range(total_frame_count)

    with torch.inference_mode():
        with torch.amp.autocast(device, enabled=(use_amp and device == 'cuda')):
            pbar = tqdm(total=total_frame_count)
            pbar.set_description(f'Processing {cfg["direction"]} {"video" if input_type == "video" else "frames"} {video_name}')
            for idx in frame_indices:
                frame_reader.set_frame(idx)
                ret, frame = frame_reader.read()
                if not ret:
                    break
                # convert numpy array to pytorch tensor format
                frame_torch = image_to_torch(frame, device=device)
                # timing start
                if 'cuda' in device:
                    torch.cuda.synchronize(device)
                    start = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
                    end = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
                    start.record()
                else:
                    a = perf_counter()

                frame_name = f'{idx:07d}.png'
                mask = None
                mask_path = path.join(mask_dir, frame_name)
                if path.exists(mask_path):
                    mask = Image.open(mask_path)
                    mask_np = np.array(mask)
                    # Get unique IDs and remap them to sequential indices
                    unique_ids = np.unique(mask_np)
                    object_ids = [id for id in unique_ids if id > 0]
                    id_map = {id: idx+1 for idx, id in enumerate(object_ids)}  # 0 stays 0 (background)
                    remapped_mask = np.zeros_like(mask_np)
                    for orig_id, new_id in id_map.items():
                        remapped_mask[mask_np == orig_id] = new_id
                    num_objects = len(object_ids)
                    mask_torch = index_numpy_to_one_hot_torch(remapped_mask, num_objects + 1).to(device)
                    prob = processor.step(frame_torch, mask_torch[1:], idx_mask=False)
                else:
                    prob = processor.step(frame_torch)

                # Convert probability to mask and save using mask_transformer
                mask = torch.argmax(prob, dim=0)
                mask_np = mask.cpu().numpy()
                # Convert back to original object IDs if needed
                if 'id_map' in locals():
                    reverse_map = {v: k for k, v in id_map.items()}
                    original_mask = np.zeros_like(mask_np)
                    for new_id, orig_id in reverse_map.items():
                        original_mask[mask_np == new_id] = orig_id
                    mask_np = original_mask
                
                # Save using mask_transformer
                mask_transformer.save_masks(
                    mask_np,
                    mask_output_root,
                    idx,
                    format_type=save_format,
                    object_ids=object_ids if 'object_ids' in locals() else None
                )

                # timing end
                if 'cuda' in device:
                    end.record()
                    torch.cuda.synchronize(device)
                    total_process_time += (start.elapsed_time(end) / 1000)
                else:
                    b = perf_counter()
                    total_process_time += (b - a)

                check_to_clear_non_permanent_cuda_memory(
                    processor=processor,
                    device=device,
                    mem_cleanup_ratio=mem_cleanup_ratio
                )
                pbar.update(1)

    pbar.close()
    frame_reader.release()  # Release resources


    print(
        '------------------------------------------------------------------------------------------------------------------------------------------------'
    )
    print(f'Total processing time: {total_process_time}')
    print(f'Total processed frames: {current_frame_index}')
    print(f'FPS: {current_frame_index / total_process_time}')
    print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}'
          ) if device == 'cuda' else None
    print(
        '------------------------------------------------------------------------------------------------------------------------------------------------'
    )

def check_to_clear_non_permanent_cuda_memory(processor: InferenceCore, device, mem_cleanup_ratio):
    if 'cuda' in device:
        if mem_cleanup_ratio > 0 and mem_cleanup_ratio <= 1:
            info = torch.cuda.mem_get_info()

            global_free, global_total = info
            global_free /= (2**30)  # GB
            global_total /= (2**30)  # GB
            global_used = global_total - global_free
            #mem_ratio = round(global_used / global_total * 100)
            mem_ratio = global_used / global_total
            if mem_ratio > mem_cleanup_ratio:
                print(f'GPU cleanup triggered: {mem_ratio} > {mem_cleanup_ratio}')
                processor.clear_non_permanent_memory()
                torch.cuda.empty_cache()


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        '--frames_dir',
        help='Path to frames directory',
        type=str,
        default=None)
    parser.add_argument(
        '--video',
        help='Path to video file',
        type=str,
        default=None)
    parser.add_argument(
        '--output',
        help='Output directory',
        type=str,
        default=None)
    parser.add_argument(
        '--model',
        help='Path to model file',
        type=str,
        default=None)
    parser.add_argument(
        '--device',
        help='Device to use (cuda/cpu)',
        type=str,
        default='cuda')
    parser.add_argument(
        '--max_frames',
        help='Maximum number of frames to process',
        type=int,
        default=None)
    parser.add_argument(
        '--max_internal_size',
        help='Maximum internal size for processing',
        type=int,
        default=480)
    parser.add_argument(
        '--mem_cleanup_ratio',
        help='How often to clear non permanent GPU memory; when ratio of GPU memory used is above given mem_cleanup_ratio [0;1] then cleanup is triggered; only used when device=cuda.',
        type=float,
        default=-1)
    parser.add_argument(
        '--input_type',
        help='Input type (video/frames)',
        type=str,
        default='frames')
    parser.add_argument(
        '--direction',
        help='Inference direction (forward/backward)',
        type=str,
        default='forward'
    )
    parser.add_argument(
        '--mode',
        help='Tracking mode (simultaneous/sequential)',
        type=str,
        default='simultaneousß'
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # input arguments
    args = get_arguments()

    # getting hydra's config without using its decorator
    initialize(version_base='1.3.2', config_path="../../configs", job_name="process_video")
    cfg = compose(config_name="video_config")

    # merge arguments into config
    args_dict = vars(args)
    with open_dict(cfg):
        # Update top-level config
        for k, v in args_dict.items():
            if v is not None:  # Only update if value is provided
                cfg[k] = v
        
        # Update model config if model path is provided
        if args.model is not None:
            cfg.model.path = args.model
            cfg.model.device = args.device
            cfg.model.use_amp = (args.device == 'cuda')
            cfg.model.max_internal_size = args.max_internal_size
            cfg.model.mem_cleanup_ratio = args.mem_cleanup_ratio

    process_video(cfg)