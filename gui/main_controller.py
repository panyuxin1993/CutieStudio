import os
from os import path
import logging
from pathlib import Path
import time
import cv2
from PIL import Image

# fix conflicts between qt5 and cv2
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

import torch
try:
    from torch import mps
except:
    print('torch.MPS not available.')
from torch import autocast
from torchvision.transforms.functional import to_tensor
from omegaconf import DictConfig, open_dict
from PySide6.QtCore import Qt, QThread, Signal, QObject

from cutie.model.cutie import CUTIE
from cutie.inference.inference_core import InferenceCore

from gui.interaction import *
from gui.interactive_utils import *
from gui.resource_manager import ResourceManager
from gui.gui import GUI
from gui.click_controller import ClickController
from gui.reader import PropagationReader, get_data_loader
from gui.exporter import convert_frames_to_video, convert_mask_to_binary
from scripts.download_models import download_models_if_needed
from utils.mask_metrics import (
    calculate_mask_metrics_batch, 
    calculate_all_pairwise_metrics, 
    calculate_all_pairwise_metrics_optimized, 
    calculate_all_pairwise_metrics_batch_optimized,
    save_pairwise_metrics
)
from utils.performance_monitor import start_global_monitoring, stop_global_monitoring, update_global_frame_count, print_global_summary

import numpy as np
import pandas as pd
from tqdm import tqdm
from cutie.utils.palette import davis_palette_np
from typing import Dict, List, Tuple

log = logging.getLogger()


class MainController():

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.initialized = False

        # setting up the workspace
        if cfg["workspace"] is None:
            if cfg["images"] is not None:
                basename = path.basename(cfg["images"])
            elif cfg["video"] is not None:
                basename = path.basename(cfg["video"])[:-4]
            else:
                raise NotImplementedError('Either images, video, or workspace has to be specified')

            cfg["workspace"] = path.join(cfg['workspace_root'], basename)

        # reading arguments
        self.cfg = cfg
        self.num_objects = cfg['num_objects']
        self.name_objects = cfg['name_objects']
        self.device = cfg['device']
        self.amp = cfg['amp']

        # Initialize sets for visible and tracked objects
        self.visible_objects = set(range(1, self.num_objects + 1))  # All objects visible by default
        self.tracked_objects = set(range(1, self.num_objects + 1))  # All objects tracked by default

        # initializing the network(s)
        self.initialize_networks()

        # main components
        self.res_man = ResourceManager(cfg)
        if 'workspace_init_only' in cfg and cfg['workspace_init_only']:
            return
        self.processor = InferenceCore(self.cutie, self.cfg)
        
        # Initialize save_soft_mask flag
        self.save_soft_mask = True
        
        # Performance monitoring
        self.performance_stats = {
            'frames_processed': 0,
            'total_processing_time': 0.0,
            'avg_fps': 0.0,
            'last_frame_time': 0.0
        }
        
        # Get performance settings from config
        self.batch_save_soft_masks = cfg.get('performance', {}).get('batch_save_soft_masks', True)
        self.enable_mask_cache = cfg.get('performance', {}).get('enable_mask_cache', True)
        self.lazy_saving = cfg.get('performance', {}).get('lazy_saving', True)
        self.save_only_tracked = cfg.get('performance', {}).get('save_only_tracked', True)
        self.save_all_visible = cfg.get('performance', {}).get('save_all_visible', True)
        self.pairwise_metrics_batch_size = cfg.get('performance', {}).get('pairwise_metrics_batch_size', 50)
        self.pairwise_metrics_max_workers = cfg.get('performance', {}).get('pairwise_metrics_max_workers', 8)
        self.pairwise_metrics_optimization_level = cfg.get('performance', {}).get('pairwise_metrics_optimization_level', 'mega')
        
        print(f"Performance settings:")
        print(f"  - Batch save soft masks: {self.batch_save_soft_masks}")
        print(f"  - Enable mask cache: {self.enable_mask_cache}")
        print(f"  - Lazy saving: {self.lazy_saving}")
        print(f"  - Save all visible objects: {self.save_all_visible}")
        print(f"  - Save only tracked objects: {self.save_only_tracked}")
        print(f"  - Pairwise metrics batch size: {self.pairwise_metrics_batch_size}")
        print(f"  - Pairwise metrics max workers: {self.pairwise_metrics_max_workers}")
        print(f"  - Pairwise metrics optimization level: {self.pairwise_metrics_optimization_level}")
        
        # Create GUI after initializing other components
        self.gui = GUI(self, self.cfg)

        # initialize control info
        self.length: int = self.res_man.length
        self.interaction: Interaction = None
        self.interaction_type: str = 'Click'
        self.curr_ti: int = 0
        self.curr_object: int = 1
        self.propagating: bool = False
        self.propagate_direction: Literal['forward', 'backward', 'none'] = 'none'
        self.last_ex = self.last_ey = 0

        # current frame info
        self.curr_frame_dirty: bool = False
        self.curr_image_np: np.ndarray = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.curr_image_torch: torch.Tensor = None
        self.curr_mask: np.ndarray = np.zeros((self.h, self.w), dtype=np.uint8)
        self.curr_prob: torch.Tensor = torch.zeros((self.num_objects + 1, self.h, self.w),
                                                   dtype=torch.float).to(self.device)
        self.curr_prob[0] = 1

        # visualization info
        self.vis_mode: str = 'davis'
        self.vis_image: np.ndarray = None
        self.save_visualization_mode: str = 'Always'

        self.interacted_prob: torch.Tensor = None
        self.overlay_layer: np.ndarray = None
        self.overlay_layer_torch: torch.Tensor = None

        # the object id used for popup/layer overlay
        self.vis_target_objects = list(range(1, self.num_objects + 1))

        # Zoom and pan state
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.is_panning = False
        self.last_pan_pos = None

        print("Loading initial frame...")
        self.load_current_image_mask()
        self.convert_current_image_mask_torch()
        self.show_current_frame()

        # initialize stuff
        self.update_memory_gauges()
        self.update_gpu_gauges()
        self.gui.work_mem_min.setValue(self.processor.memory.min_mem_frames)
        self.gui.work_mem_max.setValue(self.processor.memory.max_mem_frames)
        self.gui.long_mem_max.setValue(self.processor.memory.max_long_tokens)
        self.gui.mem_every_box.setValue(self.processor.mem_every)

        # for exporting videos
        self.output_fps = cfg['output_fps']
        self.output_bitrate = cfg['output_bitrate']

        # set callbacks
        self.gui.on_mouse_motion_xy = self.on_mouse_motion_xy
        self.gui.click_fn = self.click_fn

        self.gui.show()
        self.gui.text('Initialized.')
        self.initialized = True

        # Update checkbox states to reflect logical relationship
        self.update_checkbox_states()

        # try to load the default overlay
        self._try_load_layer('./docs/uiuc.png')
        self.gui.set_object_color(self.curr_object)
        self.update_config()

    def initialize_networks(self) -> None:
        download_models_if_needed()
        print("\nInitializing CUTIE model...")
        print(f"Loading weights from: {self.cfg.weights}")
        print(f"Using device: {self.device}")
        print(f"Model configuration:")
        print(f"- Number of objects: {self.num_objects}")
        print(f"- Using AMP: {self.amp}")
        print(f"- Long term memory: {self.cfg.use_long_term}")
        if self.cfg.use_long_term:
            print(f"  - Max memory frames: {self.cfg.long_term.max_mem_frames}")
            print(f"  - Min memory frames: {self.cfg.long_term.min_mem_frames}")
            print(f"  - Number of prototypes: {self.cfg.long_term.num_prototypes}")
        print(f"- Memory update frequency: {self.cfg.mem_every}")
        print(f"- Top-k: {self.cfg.top_k}")
        
        self.cutie = CUTIE(self.cfg).eval().to(self.device)
        model_weights = torch.load(self.cfg.weights, map_location=self.device)
        self.cutie.load_weights(model_weights)
        print("Model loaded successfully!\n")

        print("Initializing RITM model...")
        print(f"Loading weights from: {self.cfg.ritm_weights}")
        self.click_ctrl = ClickController(self.cfg.ritm_weights, device=self.device)
        print("RITM model loaded successfully!\n")

    def hit_number_key(self, number: int):
        if number == self.curr_object:
            return
        self.curr_object = number
        self.gui.object_dial.setValue(number)
        if self.click_ctrl is not None:
            self.click_ctrl.unanchor()
        self.gui.text(f'Current object changed to {number}.')
        self.gui.set_object_color(number)
        self.show_current_frame()

    def click_fn(self, action: Literal['left', 'right', 'middle'], x: int, y: int):
        if self.propagating:
            return

        last_interaction = self.interaction
        new_interaction = None

        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            if action in ['left', 'right']:
                # left: positive click
                # right: negative click
                self.convert_current_image_mask_torch()
                image = self.curr_image_torch
                if (last_interaction is None or last_interaction.tar_obj != self.curr_object):
                    # create new interaction is needed
                    self.complete_interaction()
                    self.click_ctrl.unanchor()
                    new_interaction = ClickInteraction(image, self.curr_prob, (self.h, self.w),
                                                       self.click_ctrl, self.curr_object)
                    if new_interaction is not None:
                        self.interaction = new_interaction

                self.interaction.push_point(x, y, is_neg=(action == 'right'))
                self.interacted_prob = self.interaction.predict().to(self.device, non_blocking=True)
                self.update_interacted_mask()
                self.update_gpu_gauges()

            elif action == 'middle':
                # middle: select a new visualization object
                target_object = self.curr_mask[int(y), int(x)]
                if target_object in self.vis_target_objects:
                    self.vis_target_objects.remove(target_object)
                else:
                    self.vis_target_objects.append(target_object)
                self.gui.text(f'Overlay target(s) changed to {self.vis_target_objects}')
                self.show_current_frame()
                return
            else:
                raise NotImplementedError

    def load_current_image_mask(self, no_mask: bool = False, force_from_all_masks: bool = False):
        """Load the current frame's image and mask for inference
        
        First tries to load from masks folder. If not found, loads from all_masks
        and extracts tracked object channels to generate inference mask.
        
        When force_from_all_masks is True, skips masks folder and loads from all_masks
        (or soft masks) only. Use this when masks folder is incomplete (e.g. only
        some objects after modifying tracking).
        """
        print(f"Loading current image mask for frame {self.curr_ti}" + (" (force from all_masks)" if force_from_all_masks else ""))
        try:
            self.curr_image_np = self.res_man.get_image(self.curr_ti)
            print(f"Loaded image shape: {self.curr_image_np.shape}")
            self.curr_image_torch = None

            if not no_mask:
                # Skip masks folder when force_from_all_masks (masks folder may be incomplete)
                loaded_mask = None
                if not force_from_all_masks:
                    loaded_mask = self.res_man.get_mask(self.curr_ti, tracked_objects=self.tracked_objects)
                if loaded_mask is not None:
                    print("Loaded existing mask from masks folder (tracked objects only)")
                    # Filter to only tracked objects (should already be filtered, but double-check)
                    filtered_mask = np.zeros_like(loaded_mask)
                    for obj_id in self.tracked_objects:
                        filtered_mask[loaded_mask == obj_id] = obj_id
                    self.curr_mask = filtered_mask
                    self.curr_prob = None
                else:
                    # No mask in masks folder, try loading from all_masks and extract tracked objects
                    print("No mask found in masks folder, trying to load from all_masks...")
                    all_masks_data = self.res_man.get_all_masks(self.curr_ti)
                    if all_masks_data is not None:
                        multi_channel_mask = all_masks_data['mask']  # (num_objects, H, W)
                        print(f"Loaded all_masks for frame {self.curr_ti}, shape: {multi_channel_mask.shape}")
                        
                        # Extract tracked objects from multi-channel mask and convert to single-channel format
                        # Channel i-1 corresponds to object ID i
                        inference_mask = np.zeros((self.h, self.w), dtype=np.uint8)
                        found_objects = set()
                        
                        # Process tracked objects in sorted order (later objects overwrite earlier ones in overlaps)
                        for obj_id in sorted(self.tracked_objects):
                            if 1 <= obj_id <= self.num_objects:
                                channel_idx = obj_id - 1
                                if channel_idx < multi_channel_mask.shape[0]:
                                    # Get binary mask for this object
                                    binary_mask = (multi_channel_mask[channel_idx] > 0.5).astype(np.uint8)
                                    # Set object ID in inference mask (overwrites previous objects in overlaps)
                                    inference_mask[binary_mask > 0] = obj_id
                                    num_pixels = np.sum(binary_mask > 0)
                                    if num_pixels > 0:
                                        print(f"  Extracted object {obj_id} (channel {channel_idx}) from all_masks: {num_pixels} pixels")
                                        found_objects.add(obj_id)
                        
                        # Check for any tracked objects missing from all_masks - try loading from soft_masks
                        missing_objects = self.tracked_objects - found_objects
                        if missing_objects:
                            print(f"Some tracked objects missing from all_masks: {missing_objects}, checking soft_masks...")
                            for obj_id in sorted(missing_objects):
                                if 1 <= obj_id <= self.num_objects:
                                    p = os.path.join(self.res_man.soft_mask_dir, f'{obj_id}', f'{self.curr_ti:07d}.png')
                                    if os.path.exists(p):
                                        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                                        if m is not None:
                                            if m.shape != (self.h, self.w):
                                                m = cv2.resize(m, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
                                            binary = (m > 127).astype(np.uint8)
                                            inference_mask[binary > 0] = obj_id
                                            num_pixels = np.sum(binary > 0)
                                            if num_pixels > 0:
                                                print(f"  Loaded object {obj_id} from soft_masks: {num_pixels} pixels")
                        
                        self.curr_mask = inference_mask
                        self.curr_prob = None
                        print(f"Generated inference mask from all_masks (and soft_masks): {len(self.tracked_objects)} tracked objects, {np.sum(inference_mask > 0)} non-zero pixels")
                    else:
                        # all_masks missing; build from individual soft masks if force_from_all_masks
                        if force_from_all_masks:
                            inference_mask = np.zeros((self.h, self.w), dtype=np.uint8)
                            for obj_id in sorted(self.tracked_objects):
                                if 1 <= obj_id <= self.num_objects:
                                    p = os.path.join(self.res_man.soft_mask_dir, f'{obj_id}', f'{self.curr_ti:07d}.png')
                                    if os.path.exists(p):
                                        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                                        if m is not None:
                                            if m.shape != (self.h, self.w):
                                                m = cv2.resize(m, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
                                            binary = (m > 127).astype(np.uint8)
                                            inference_mask[binary > 0] = obj_id
                            self.curr_mask = inference_mask
                            self.curr_prob = None
                            print(f"Built inference mask from soft_masks: {len(self.tracked_objects)} tracked, {np.sum(inference_mask > 0)} non-zero pixels")
                        else:
                            print("No mask found in all_masks folder either, using empty mask")
                            self.curr_mask.fill(0)
                            self.curr_prob = None
        except Exception as e:
            print(f"Error loading frame {self.curr_ti}: {str(e)}")
            raise

    def convert_current_image_mask_torch(self, no_mask: bool = False):
        """Convert current frame to torch format"""

        try:
            if self.curr_image_torch is None:
                self.curr_image_torch = to_tensor(self.curr_image_np).to(self.device, non_blocking=True)


            if self.curr_prob is None and not no_mask:
                # Ensure mask values are within valid range
                if np.max(self.curr_mask) > self.num_objects:
                    print(f"Warning: Mask contains object ID {np.max(self.curr_mask)} > num_objects {self.num_objects}")
                    # Remap to valid range
                    unique_ids = np.unique(self.curr_mask)
                    object_ids = [id for id in unique_ids if id > 0]
                    id_map = {id: idx+1 for idx, id in enumerate(object_ids)}  # 0 stays 0 (background)
                    remapped_mask = np.zeros_like(self.curr_mask)
                    for orig_id, new_id in id_map.items():
                        remapped_mask[self.curr_mask == orig_id] = new_id
                    self.curr_mask = remapped_mask
                    print(f"Remapped object IDs: {id_map}")
                
                self.curr_prob = index_numpy_to_one_hot_torch(self.curr_mask, self.num_objects + 1).to(
                    self.device, non_blocking=True)

        except Exception as e:
            print(f"Error converting to torch format: {str(e)}")
            raise

    def compose_current_im(self):
        """Compose current image with masks from all_masks folder
        
        Loads multi-channel masks from all_masks folder and overlays each visible object
        one by one. Later objects overwrite earlier ones in overlapping areas.
        Also loads soft masks for visible objects that aren't in all_masks yet.
        """
        # Always load from all_masks folder (multi-channel format)
        loaded_mask_data = self.res_man.get_all_masks(self.curr_ti)
        
        # Get multi-channel mask data (fixed size: num_objects channels)
        if loaded_mask_data is None:
            # No mask data available in all_masks, start with empty fixed-size structure
            print(f"No mask data found in all_masks for frame {self.curr_ti}, will load from soft masks if available")
            # Fixed-size mask: channel i-1 = object ID i
            multi_channel_mask = np.zeros((self.num_objects, self.h, self.w), dtype=np.float32)
            object_ids = list(range(1, self.num_objects + 1))  # Fixed mapping
        else:
            # Extract multi-channel mask (fixed size: num_objects channels)
            multi_channel_mask = loaded_mask_data['mask']  # (num_objects, H, W)
            object_ids = loaded_mask_data['object_ids']  # [1, 2, ..., num_objects]
            print(f"Loaded all_masks for frame {self.curr_ti}: shape {multi_channel_mask.shape}, expected {self.num_objects} channels")
            
            # Verify fixed-size format
            if multi_channel_mask.shape[0] != self.num_objects:
                raise ValueError(f"Mask has {multi_channel_mask.shape[0]} channels, expected {self.num_objects}")
            
            # Ensure we have the correct dimensions
            _, h_mask, w_mask = multi_channel_mask.shape
            if h_mask != self.h or w_mask != self.w:
                print(f"Warning: Mask dimensions mismatch. Expected ({self.h}, {self.w}), got ({h_mask}, {w_mask}). Resizing...")
                # Resize all channels
                resized_channels = []
                for ch_idx in range(multi_channel_mask.shape[0]):
                    ch_resized = cv2.resize(multi_channel_mask[ch_idx], (self.w, self.h), interpolation=cv2.INTER_NEAREST)
                    resized_channels.append(ch_resized)
                multi_channel_mask = np.stack(resized_channels, axis=0)
        
        # Update with current probabilities for tracked objects if available
        # Fixed mapping: channel i-1 = object ID i
        if self.curr_prob is not None:
            prob_np = self.curr_prob.cpu().numpy() if hasattr(self.curr_prob, 'cpu') else self.curr_prob
            
            # For each tracked object, update its channel in the multi-channel mask
            for obj_id in range(1, min(prob_np.shape[0], self.num_objects + 1)):
                if obj_id in self.tracked_objects:
                    channel_idx = obj_id - 1  # Fixed mapping: channel i-1 = object ID i
                    # Update channel with current probability (overwrites existing mask)
                    multi_channel_mask[channel_idx] = prob_np[obj_id].astype(np.float32)
        
        # Load soft masks for visible objects that aren't already loaded or don't have current probabilities
        # Fixed mapping: channel i-1 = object ID i
        for obj_id in self.visible_objects:
            if 1 <= obj_id <= self.num_objects:
                channel_idx = obj_id - 1
                # Only load from soft mask if channel is empty (tracked objects with current prob take priority)
                if not np.any(multi_channel_mask[channel_idx] > 0.5):
                    # Check if soft mask exists for this object
                    soft_mask_path = os.path.join(self.res_man.soft_mask_dir, f'{obj_id}', f'{self.curr_ti:07d}.png')
                    print(f"Checking for soft mask: {soft_mask_path} (exists: {os.path.exists(soft_mask_path)})")
                    if os.path.exists(soft_mask_path):
                        soft_mask = cv2.imread(soft_mask_path, cv2.IMREAD_GRAYSCALE)
                        if soft_mask is not None:
                            print(f"Loaded soft mask for object {obj_id}, shape: {soft_mask.shape}, expected: ({self.h}, {self.w})")
                            # Resize if needed
                            if soft_mask.shape != (self.h, self.w):
                                print(f"Resizing soft mask from {soft_mask.shape} to ({self.h}, {self.w})")
                                soft_mask = cv2.resize(soft_mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
                            
                            # Convert binary mask to probability (0.0 or 1.0)
                            mask_prob = (soft_mask > 127).astype(np.float32)
                            num_pixels = np.sum(mask_prob > 0.5)
                            
                            # Update the fixed channel (channel i-1 = object ID i)
                            multi_channel_mask[channel_idx] = mask_prob
                            
                            print(f"Added object {obj_id} to visualization mask (channel {channel_idx}), {num_pixels} pixels")
                        else:
                            print(f"Failed to read soft mask for object {obj_id} - cv2.imread returned None")
                    else:
                        print(f"Soft mask not found for object {obj_id} at frame {self.curr_ti}: {soft_mask_path}")
                        # Also check if soft_mask_dir exists
                        if not os.path.exists(self.res_man.soft_mask_dir):
                            print(f"  ERROR: soft_mask_dir does not exist: {self.res_man.soft_mask_dir}")
                        elif not os.path.exists(os.path.dirname(soft_mask_path)):
                            print(f"  ERROR: Object directory does not exist: {os.path.dirname(soft_mask_path)}")
                        else:
                            # List available files in the object directory to help debug
                            obj_dir = os.path.dirname(soft_mask_path)
                            if os.path.exists(obj_dir):
                                available_files = sorted(os.listdir(obj_dir))
                                print(f"  Available files in {obj_dir}: {available_files[:10]}...")  # Show first 10 files
        
        # Create visualization mask by overlaying visible objects one by one
        # Start with empty mask
        vis_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        
        # Sort visible objects to ensure consistent ordering (later objects overwrite earlier ones)
        # Fixed mapping: channel i-1 = object ID i, so we just iterate through visible objects
        visible_object_ids = sorted([obj_id for obj_id in self.visible_objects if 1 <= obj_id <= self.num_objects])
        
        print(f"Composing visualization for frame {self.curr_ti}:")
        print(f"  - Fixed-size mask: {self.num_objects} channels (channel i-1 = object ID i)")
        print(f"  - Visible objects: {len(visible_object_ids)} ({visible_object_ids})")
        print(f"  - Multi-channel mask shape: {multi_channel_mask.shape}")
        
        # Overlay each visible object's mask one by one
        # Later objects will overwrite earlier ones in overlapping areas
        # Fixed mapping: channel i-1 = object ID i
        for obj_id in visible_object_ids:
            channel_idx = obj_id - 1  # Fixed mapping
            if channel_idx < multi_channel_mask.shape[0]:
                # Get binary mask for this object
                binary_mask = (multi_channel_mask[channel_idx] > 0.5).astype(np.uint8)
                num_pixels = np.sum(binary_mask > 0)
                print(f"  - Object {obj_id} (channel {channel_idx}): {num_pixels} pixels")
                # Overwrite: later objects overwrite earlier ones
                vis_mask[binary_mask > 0] = obj_id
            else:
                print(f"  - WARNING: Object {obj_id} channel {channel_idx} out of range (mask shape: {multi_channel_mask.shape})")
        
        print(f"Final visualization mask - max value: {vis_mask.max()}, non-zero pixels: {np.sum(vis_mask > 0)}")
                
        # Use visible_objects for basic visualization, vis_target_objects for popup/layer modes
        target_objects = self.vis_target_objects if self.vis_mode in ['popup', 'layer'] else visible_object_ids
        self.vis_image = get_visualization(self.vis_mode, self.curr_image_np, vis_mask,
                                           self.overlay_layer, target_objects)

    def update_canvas(self):
        self.compose_current_im()
        self.gui.set_canvas(self.vis_image)

    def update_current_image_fast(self, invalid_soft_mask: bool = False):
        """Update the current image with fast visualization"""
        if self.curr_prob is None:
            return
            
        # Save soft masks only for tracked objects (not visible objects)
        # This prevents overwriting existing soft masks for untracked objects
        if self.save_soft_mask:
            print(f"Saving soft masks for tracked objects: {self.tracked_objects}")
            
            # Prepare batch data for tracked objects only
            soft_masks = {}
            for obj_id in range(1, self.num_objects + 1):
                if obj_id in self.tracked_objects:
                    obj_mask = self.curr_prob[obj_id].cpu().numpy()
                    soft_masks[obj_id] = obj_mask
            
            # Use batch saving for better performance
            if soft_masks:
                self.res_man.save_batch_soft_masks(self.curr_ti, soft_masks, self.tracked_objects, self.save_all_visible)
                    
        # Update visualization
        self.show_current_frame()

    def show_current_frame(self, fast: bool = False, invalid_soft_mask: bool = False):
        """Show the current frame with proper visibility handling"""

        if self.curr_prob is None:
            print("No probability mask available")
            return
            
        # Ensure we have the torch tensors
        if self.curr_image_torch is None:
            self.convert_current_image_mask_torch()
            
        if self.curr_image_torch is None:
            print("Failed to convert image to torch format")
            return
        
        # Always create the proper combined mask that includes existing soft masks for untracked objects
        # This ensures that during propagation, we can see all visible objects, not just tracked ones
        self.compose_current_im()
        
        # Use visible_objects for basic visualization, vis_target_objects for popup/layer modes
        target_objects = self.vis_target_objects if self.vis_mode in ['popup', 'layer'] else list(self.visible_objects)
        
        # Get visualization using the composed image (which includes all visible objects)
        # Note: compose_current_im() already sets self.vis_image, so we don't need to call get_visualization again
        # Just ensure the image is properly formatted
        self.curr_image_torch = None
        self.vis_image = np.ascontiguousarray(self.vis_image)
        
        # Save visualization if needed
        save_visualization = self.save_visualization_mode in [
            'Propagation only (higher quality)', 'Always'
        ]
        if save_visualization and not invalid_soft_mask:
            self.res_man.save_visualization(self.curr_ti, self.vis_mode, self.vis_image)
            
        # Update GUI
        self.gui.set_canvas(self.vis_image)

        self.gui.update_slider(self.curr_ti)
        self.gui.frame_name.setText(self.res_man.names[self.curr_ti] + '.jpg')

    def set_vis_mode(self):
        self.vis_mode = self.gui.combo.currentText()
        self.show_current_frame()

    def save_current_mask(self):
        """Save inference masks to masks folder (ONLY tracked objects)
        
        IMPORTANT: Only tracked objects are saved to prevent untracked objects
        from interfering with inference. Untracked objects are excluded.
        """
        # Save to masks folder with ONLY tracked objects
        self.res_man.save_mask(self.curr_ti, self.curr_mask, tracked_objects=self.tracked_objects)

    def on_slider_update(self):
        """Handle timeline slider updates"""
        print(f"Slider update triggered: current value = {self.gui.tl_slider.value()}")
        self.curr_ti = self.gui.tl_slider.value()
        # if we are propagating, the on_run function will take care of everything
        # don't do duplicate work here
        if self.propagating:
            print("Propagation in progress, ignoring slider update")
            return
            
        new_ti = self.gui.tl_slider.value()
        print(f"Updating to frame {new_ti}")
        
        # Save current frame if needed
        if self.curr_frame_dirty:
            print("Saving current frame")
            self.save_current_mask()
            self.curr_frame_dirty = False

        # Reset interaction state
        self.reset_this_interaction()
        
        # Update current frame index
        self.curr_ti = new_ti
        
        # Load and show new frame
        print(f"Loading frame {self.curr_ti}")
        self.load_current_image_mask()
 
        self.convert_current_image_mask_torch()

        self.show_current_frame()
        

    def on_forward_propagation(self):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
            self.propagate_direction = 'none'
        else:
            self.propagate_fn = self.on_next_frame
            self.gui.forward_propagation_start()
            self.propagate_direction = 'forward'
            self.on_propagate()

    def on_propagate_step_forward(self):
        """Propagate forward through all frames but reset memory every frame (like Step forward repeatedly)."""
        if self.propagating:
            self.propagating = False
            self.propagate_direction = 'none'
        else:
            self.propagate_fn = self.on_next_frame
            self.gui.forward_propagation_start()
            self.propagate_direction = 'forward'
            self.propagate_step_forward_loop()

    def step_forward_propagation(self):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
            self.propagate_direction = 'none'
        else:
            self.propagate_fn = self.on_next_frame
            self.gui.forward_propagation_step()
            self.propagate_direction = 'forward'
            self.step_propagate()

    def on_backward_propagation(self):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
            self.propagate_direction = 'none'
        else:
            self.propagate_fn = self.on_prev_frame
            self.gui.backward_propagation_start()
            self.propagate_direction = 'backward'
            self.on_propagate()

    def on_pause(self):
        self.propagating = False
        self.gui.text(f'Propagation stopped at t={self.curr_ti}.')
        self.gui.pause_propagation()

    def on_propagate(self):
        # start to propagate
        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            self.convert_current_image_mask_torch()

            self.tracked_prob = self.curr_prob.clone()
            for obj_id in range(1, self.num_objects + 1):
                if obj_id not in self.tracked_objects:
                    self.tracked_prob[obj_id] = 0

            self.gui.text(f'Propagation started at t={self.curr_ti}.')
            self.processor.clear_sensory_memory()
            self.curr_prob = self.processor.step(self.curr_image_torch,
                                                 self.tracked_prob[1:],
                                                 idx_mask=False)
            self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)
            # clear
            self.interacted_prob = None
            self.reset_this_interaction()
            # override this for #41
            self.show_current_frame(fast=True, invalid_soft_mask=True)

            self.propagating = True
            self.gui.clear_all_mem_button.setEnabled(False)
            self.gui.clear_non_perm_mem_button.setEnabled(False)
            self.gui.tl_slider.setEnabled(False)

            dataset = PropagationReader(self.res_man, self.curr_ti, self.propagate_direction)
            loader = get_data_loader(dataset, self.cfg.num_read_workers)
            print(f"Propagation loader: {len(loader)} frames")
            i = 0
            # propagate till the end
            for data in loader:
                i += 1
                print(f"Propagation loop: {i}")
                if not self.propagating:
                    break
                    
                # Start timing
                frame_start_time = time.time()
                
                self.curr_image_np, self.curr_image_torch = data
                self.curr_image_torch = self.curr_image_torch.to(self.device, non_blocking=True)
                self.propagate_fn()  # This updates self.curr_ti

                self.curr_prob = self.processor.step(self.curr_image_torch)
                self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)

                self.save_current_mask()
                # Save soft masks only for tracked objects
                if self.save_soft_mask:
                    # Prepare batch data for tracked objects only
                    soft_masks = {}
                    for obj_id in range(1, self.num_objects + 1):
                        if obj_id in self.tracked_objects:
                            obj_mask = self.curr_prob[obj_id].cpu().numpy()
                            soft_masks[obj_id] = obj_mask
                    
                    # Use batch saving for better performance
                    if soft_masks:
                        self.res_man.save_batch_soft_masks(self.curr_ti, soft_masks, self.tracked_objects, self.save_all_visible)
                
                # Show current frame first to ensure visualization is created
                self.show_current_frame(fast=True)
                
                # Then invalidate cache for current frame to ensure fresh combined mask for next iteration
                self.res_man.invalidate(self.curr_ti)
                
                # Update performance stats
                frame_time = time.time() - frame_start_time
                self.update_performance_stats(frame_time)
                
                self.update_memory_gauges()
                self.gui.process_events()
                if self.curr_ti == 0 or self.curr_ti == self.T - 1:
                    break


            # stop the loop after one frame and clear memory
            self.propagating = False
            # self.on_clear_memory()

            self.curr_frame_dirty = False
            self.on_pause()
            self.on_slider_update()
            self.gui.process_events()

    def step_propagate(self):
        # start to propagate for one frame
        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            self.convert_current_image_mask_torch()

            self.tracked_prob = self.curr_prob.clone()
            for obj_id in range(1, self.num_objects + 1):
                if obj_id not in self.tracked_objects:
                    self.tracked_prob[obj_id] = 0

            self.gui.text(f'Propagation started at t={self.curr_ti}.')
            self.processor.clear_sensory_memory()
            self.curr_prob = self.processor.step(self.curr_image_torch,
                                                 self.tracked_prob[1:],
                                                 idx_mask=False)
            self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)
            # clear
            self.interacted_prob = None
            self.reset_this_interaction()
            # override this for #41
            self.show_current_frame(fast=True, invalid_soft_mask=True)

            self.propagating = True

            self.gui.clear_all_mem_button.setEnabled(False)
            self.gui.clear_non_perm_mem_button.setEnabled(False)
            self.gui.tl_slider.setEnabled(False)

            dataset = PropagationReader(self.res_man, self.curr_ti, self.propagate_direction)
            loader = get_data_loader(dataset, self.cfg.num_read_workers)

            # propagate for one frame only
            for data in loader:
                if not self.propagating:
                    break
                    
                # Start timing
                frame_start_time = time.time()
                    
                self.curr_image_np, self.curr_image_torch = data
                self.curr_image_torch = self.curr_image_torch.to(self.device, non_blocking=True)
                self.propagate_fn()

                self.curr_prob = self.processor.step(self.curr_image_torch)
                self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)

                self.save_current_mask()
                # Save soft masks only for tracked objects
                if self.save_soft_mask:
                    # Prepare batch data for tracked objects only
                    soft_masks = {}
                    for obj_id in range(1, self.num_objects + 1):
                        if obj_id in self.tracked_objects:
                            obj_mask = self.curr_prob[obj_id].cpu().numpy()
                            soft_masks[obj_id] = obj_mask
                    
                    # Use batch saving for better performance
                    if soft_masks:
                        self.res_man.save_batch_soft_masks(self.curr_ti, soft_masks, self.tracked_objects, self.save_all_visible)
                
                # Show current frame first to ensure visualization is created
                self.show_current_frame(fast=True)
                
                # Then invalidate cache for current frame to ensure fresh combined mask for next iteration
                self.res_man.invalidate(self.curr_ti)
                
                # Update performance stats
                frame_time = time.time() - frame_start_time
                self.update_performance_stats(frame_time)
                
                self.update_memory_gauges()
                self.gui.process_events()

                # stop the loop after one frame and clear memory
                self.propagating = False
                self.on_clear_memory()
                if self.curr_ti == 0 or self.curr_ti == self.T - 1:
                    break
                # Break after processing one frame (this is step_propagate, not full propagation)
                break

            self.propagating = False
            self.curr_frame_dirty = False
            self.on_pause()
            self.on_slider_update()
            self.gui.process_events()

    def propagate_step_forward_loop(self):
        """Propagate forward through all frames, clearing memory after each frame (each frame uses only previous frame mask)."""
        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            self.convert_current_image_mask_torch()

            self.tracked_prob = self.curr_prob.clone()
            for obj_id in range(1, self.num_objects + 1):
                if obj_id not in self.tracked_objects:
                    self.tracked_prob[obj_id] = 0

            self.gui.text(f'Propagation step forward started at t={self.curr_ti} (memory reset every frame).')
            self.processor.clear_sensory_memory()
            self.curr_prob = self.processor.step(self.curr_image_torch,
                                                 self.tracked_prob[1:],
                                                 idx_mask=False)
            self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)
            self.interacted_prob = None
            self.reset_this_interaction()
            self.show_current_frame(fast=True, invalid_soft_mask=True)

            self.propagating = True
            self.gui.clear_all_mem_button.setEnabled(False)
            self.gui.clear_non_perm_mem_button.setEnabled(False)
            self.gui.tl_slider.setEnabled(False)

            dataset = PropagationReader(self.res_man, self.curr_ti, self.propagate_direction)
            loader = get_data_loader(dataset, self.cfg.num_read_workers)

            for data in loader:
                if not self.propagating:
                    break

                frame_start_time = time.time()

                self.curr_image_np, self.curr_image_torch = data
                self.curr_image_torch = self.curr_image_torch.to(self.device, non_blocking=True)
                self.propagate_fn()  # advance to next frame

                self.curr_prob = self.processor.step(self.curr_image_torch)
                self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)

                self.save_current_mask()
                if self.save_soft_mask:
                    soft_masks = {}
                    for obj_id in range(1, self.num_objects + 1):
                        if obj_id in self.tracked_objects:
                            obj_mask = self.curr_prob[obj_id].cpu().numpy()
                            soft_masks[obj_id] = obj_mask
                    if soft_masks:
                        self.res_man.save_batch_soft_masks(self.curr_ti, soft_masks, self.tracked_objects, self.save_all_visible)

                self.show_current_frame(fast=True)
                self.res_man.invalidate(self.curr_ti)

                frame_time = time.time() - frame_start_time
                self.update_performance_stats(frame_time)

                self.on_clear_memory()
                self.update_memory_gauges()
                self.gui.process_events()
                # Re-init from previous frame mask and clear memory so this frame is independent
                self.convert_current_image_mask_torch()
                self.processor.clear_sensory_memory()
                tracked_prob = self.curr_prob.clone()
                for obj_id in range(1, self.num_objects + 1):
                    if obj_id not in self.tracked_objects:
                        tracked_prob[obj_id] = 0
                self.curr_prob = self.processor.step(self.curr_image_torch,
                                                    tracked_prob[1:],
                                                    idx_mask=False)
                if self.curr_ti == 0 or self.curr_ti == self.T - 1:
                    break

            self.propagating = False
            self.curr_frame_dirty = False
            self.on_pause()
            self.on_slider_update()
            self.gui.process_events()

    def pause_propagation(self):
        self.propagating = False

    def on_commit(self):
        """Commit current frame to permanent memory (ONLY tracked objects)
        
        IMPORTANT: Only tracked objects are committed to prevent untracked objects
        from interfering with future inference.
        """
        if self.interacted_prob is None:
            # get mask from disk
            self.load_current_image_mask()
        else:
            # get mask from interaction
            self.complete_interaction()
            self.update_interacted_mask()

        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            self.convert_current_image_mask_torch()
            
            # Filter to only tracked objects before committing
            tracked_prob = self.curr_prob.clone()
            for obj_id in range(1, self.num_objects + 1):
                if obj_id not in self.tracked_objects:
                    tracked_prob[obj_id] = 0
            
            self.gui.text(f'Permanent memory saved at {self.curr_ti} (tracked objects only).')
            self.curr_prob = self.processor.step(self.curr_image_torch,
                                                 tracked_prob[1:],
                                                 idx_mask=False,
                                                 force_permanent=True)
            self.update_memory_gauges()
            self.update_gpu_gauges()

    def on_play_video_timer(self):
        self.curr_ti += 1
        if self.curr_ti > self.T - 1:
            self.curr_ti = 0
        self.gui.tl_slider.setValue(self.curr_ti)

    def regenerate_visualization_with_all_objects(self):
        """Regenerate visualization images and masks in 'masks' folder for all frames,
        including all objects from all_masks or soft_masks (visualization = with background, masks = index image only).
        """
        self.gui.text('Regenerating visualization images and masks with all objects...')
        self.gui.process_events()
        
        # Save original state to restore later
        original_visible_objects = self.visible_objects.copy()
        original_tracked_objects = self.tracked_objects.copy()
        original_curr_ti = self.curr_ti
        
        # Find all frames that have masks
        frames_with_masks = set()
        
        # Check all_masks folder
        all_masks_dir = path.join(self.cfg['workspace'], 'all_masks')
        if path.exists(all_masks_dir):
            for mask_file in os.listdir(all_masks_dir):
                if mask_file.endswith('.npz') or mask_file.endswith('.npy'):
                    try:
                        # all_masks use 7-digit format (0000000.npz or 0000000.npy)
                        frame_idx = int(mask_file.replace('.npz', '').replace('.npy', ''))
                        if 0 <= frame_idx < self.T:
                            frames_with_masks.add(frame_idx)
                    except ValueError:
                        pass
        
        # Check soft_masks folder
        soft_mask_dir = self.res_man.soft_mask_dir
        if path.exists(soft_mask_dir):
            for obj_dir in os.listdir(soft_mask_dir):
                obj_path = path.join(soft_mask_dir, obj_dir)
                if path.isdir(obj_path):
                    for mask_file in os.listdir(obj_path):
                        if mask_file.endswith('.png'):
                            try:
                                # Soft masks use 7-digit format (0000000.png)
                                frame_idx = int(mask_file.replace('.png', ''))
                                if 0 <= frame_idx < self.T:
                                    frames_with_masks.add(frame_idx)
                            except ValueError:
                                pass
        
        if not frames_with_masks:
            self.gui.text('No masks found in all_masks or soft_masks folders')
            return False
        
        frames_with_masks = sorted(frames_with_masks)
        self.gui.text(f'Found {len(frames_with_masks)} frames with masks. Regenerating visualizations and masks...')
        
        # For each frame, find all objects that exist and regenerate visualization
        total_frames = len(frames_with_masks)
        for idx, ti in enumerate(frames_with_masks):
            # Find all objects that exist in this frame
            existing_objects = set()
            
            # Check all_masks
            all_masks_data = self.res_man.get_all_masks(ti)
            if all_masks_data is not None:
                multi_channel_mask = all_masks_data['mask']
                for obj_id in range(1, self.num_objects + 1):
                    channel_idx = obj_id - 1
                    if channel_idx < multi_channel_mask.shape[0]:
                        if np.any(multi_channel_mask[channel_idx] > 0.5):
                            existing_objects.add(obj_id)
            
            # Check soft_masks for any objects not found in all_masks
            for obj_id in range(1, self.num_objects + 1):
                if obj_id not in existing_objects:
                    soft_mask_path = os.path.join(soft_mask_dir, f'{obj_id}', f'{ti:07d}.png')
                    if os.path.exists(soft_mask_path):
                        existing_objects.add(obj_id)
            
            if not existing_objects:
                continue
            
            # Temporarily set visible_objects to include all existing objects
            self.visible_objects = existing_objects.copy()
            self.curr_ti = ti
            
            # Load image and mask for this frame
            try:
                self.curr_image_np = self.res_man.get_image(ti)
                if self.curr_image_np is None:
                    print(f"Warning: Could not load image for frame {ti}, skipping...")
                    continue
                
                self.curr_image_torch = None
                self.curr_prob = None
                self.curr_mask = None
                
                # Load mask from all_masks or soft_masks
                self.load_current_image_mask(force_from_all_masks=True)
                
                # Only convert to torch if we have a mask
                if self.curr_mask is not None:
                    self.convert_current_image_mask_torch()
                
                # Compose visualization with all objects
                self.compose_current_im()
                
                # Save visualization directly (synchronously) to ensure it's saved before export
                vis_dir = path.join(self.cfg['workspace'], 'visualization', self.vis_mode)
                os.makedirs(vis_dir, exist_ok=True)
                name = self.res_man.names[ti]
                
                # Convert RGB to BGR for OpenCV
                if self.vis_mode == 'rgba':
                    vis_image_bgr = cv2.cvtColor(self.vis_image, cv2.COLOR_RGBA2BGRA)
                    cv2.imwrite(path.join(vis_dir, name + '.png'), vis_image_bgr)
                else:
                    vis_image_bgr = cv2.cvtColor(self.vis_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(path.join(vis_dir, name + '.jpg'), vis_image_bgr)
                
                # Regenerate mask in 'masks' folder (single-channel index image, all objects, no background)
                if self.curr_mask is not None:
                    self.res_man.save_mask_sync(ti, self.curr_mask, tracked_objects=existing_objects)
                
            except Exception as e:
                print(f"Error regenerating visualization for frame {ti}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
            
            # Update progress
            if (idx + 1) % 10 == 0 or idx == total_frames - 1:
                progress = (idx + 1) / total_frames
                self.gui.progressbar_update(progress)
                self.gui.text(f'Regenerated {idx + 1}/{total_frames} frames (visualization + masks)')
                self.gui.process_events()
        
        # Restore original state
        self.visible_objects = original_visible_objects
        self.tracked_objects = original_tracked_objects
        self.curr_ti = original_curr_ti
        
        # Reload current frame to restore display
        if self.curr_ti is not None:
            self.load_current_image_mask()
            self.convert_current_image_mask_torch()
            self.show_current_frame()
        
        self.gui.text(f'Finished regenerating visualization images and masks for {total_frames} frames')
        return True

    def on_export_visualization(self):
        # Regenerate visualization images with all objects before exporting
        self.gui.text('Preparing video export: regenerating visualizations with all objects...')
        self.gui.process_events()
        
        if not self.regenerate_visualization_with_all_objects():
            self.gui.text('Failed to regenerate visualizations. Proceeding with existing images...')
        
        # NOTE: Save visualization at the end of propagation
        image_folder = path.join(self.cfg['workspace'], 'visualization', self.vis_mode)
        save_folder = self.cfg['workspace']
        if path.exists(image_folder):
            # Sorted so frames will be in order
            output_path = path.join(save_folder, f'visualization_{self.vis_mode}.mp4')
            self.gui.text(f'Exporting visualization video -- please wait')
            self.gui.process_events()
            convert_frames_to_video(image_folder,
                                    output_path,
                                    fps=self.output_fps,
                                    bitrate=self.output_bitrate,
                                    progress_callback=self.gui.progressbar_update)
            self.gui.text(f'Visualization exported to {output_path}')
            self.gui.progressbar_update(0)
        else:
            self.gui.text(f'No visualization images found in {image_folder}')

    def on_export_binary(self):
        # export masks in binary format for other applications, e.g., ProPainter
        mask_folder = path.join(self.cfg['workspace'], 'masks')
        save_folder = path.join(self.cfg['workspace'], 'binary_masks')
        if path.exists(mask_folder):
            os.makedirs(save_folder, exist_ok=True)
            self.gui.text(f'Exporting binary masks -- please wait')
            self.gui.process_events()
            convert_mask_to_binary(mask_folder,
                                   save_folder,
                                   self.vis_target_objects,
                                   progress_callback=self.gui.progressbar_update)
            self.gui.text(f'Binary masks exported to {save_folder}')
            self.gui.progressbar_update(0)
        else:
            self.gui.text(f'No masks found in {mask_folder}')

    def on_object_dial_change(self):
        object_id = self.gui.object_dial.value()
        self.hit_number_key(object_id)

    def on_frame_dial_change(self):
        """Handle frame dial changes"""
        frame_id = self.gui.frame_dial.value()
        self.gui.tl_slider.setValue(frame_id)
        # on_slider_update will be triggered automatically by the slider's valueChanged signal

    def on_fps_dial_change(self):
        self.output_fps = self.gui.export_dialog.fps_dial.value()

    def on_bitrate_dial_change(self):
        self.output_bitrate = self.gui.export_dialog.bitrate_dial.value()

    def update_interacted_mask(self):
        self.curr_prob = self.interacted_prob
        self.curr_mask = torch_prob_to_numpy_mask(self.interacted_prob)
        self.save_current_mask()
        self.show_current_frame()
        self.curr_frame_dirty = False

    def reset_this_interaction(self):
        self.complete_interaction()
        self.interacted_prob = None
        if self.click_ctrl is not None:
            self.click_ctrl.unanchor()

    def on_reset_mask(self):
        """Reset all masks in the current frame"""
        print(f"Resetting all masks for frame {self.curr_ti}")
        
        # Clear current frame masks
        self.curr_mask.fill(0)
        if self.curr_prob is not None:
            self.curr_prob.fill_(0)
        
        # Clear soft masks from disk for all objects in this frame
        for obj_id in range(1, self.num_objects + 1):
            soft_mask_path = os.path.join(self.res_man.soft_mask_dir, f'{obj_id}', f'{self.curr_ti:07d}.png')
            if os.path.exists(soft_mask_path):
                os.remove(soft_mask_path)
                print(f"Removed soft mask for object {obj_id} at frame {self.curr_ti}")
        
        # Clear cache for this frame to force regeneration
        self.res_man.invalidate(self.curr_ti)
        
        # Update GUI (don't save current mask as it would overwrite the reset)
        self.curr_frame_dirty = True
        self.reset_this_interaction()
        self.show_current_frame()
        
        print(f"Reset complete for frame {self.curr_ti}")

    def on_reset_object(self):
        """Reset masks for the current object only"""
        print(f"Resetting masks for object {self.curr_object} at frame {self.curr_ti}")
        
        # Clear current object from current frame masks
        self.curr_mask[self.curr_mask == self.curr_object] = 0
        if self.curr_prob is not None:
            self.curr_prob[self.curr_object] = 0
        
        # Clear soft mask from disk for current object in this frame
        soft_mask_path = os.path.join(self.res_man.soft_mask_dir, f'{self.curr_object}', f'{self.curr_ti:07d}.png')
        if os.path.exists(soft_mask_path):
            os.remove(soft_mask_path)
            print(f"Removed soft mask for object {self.curr_object} at frame {self.curr_ti}")
        
        # Clear cache for this frame to force regeneration
        self.res_man.invalidate(self.curr_ti)
        
        # Update GUI (don't save current mask as it would overwrite the reset)
        self.curr_frame_dirty = True
        self.reset_this_interaction()
        self.show_current_frame()
        
        print(f"Reset complete for object {self.curr_object} at frame {self.curr_ti}")

    def complete_interaction(self):
        if self.interaction is not None:
            self.interaction = None

    def on_prev_frame(self, step=1):
        new_ti = max(0, self.curr_ti - step)
        self.gui.tl_slider.setValue(new_ti)

    def on_next_frame(self, step=1):
        new_ti = min(self.curr_ti + step, self.length - 1)
        self.gui.tl_slider.setValue(new_ti)

    def update_gpu_gauges(self):
        if 'cuda' in self.device:
            info = torch.cuda.mem_get_info()
            global_free, global_total = info
            global_free /= (2**30)
            global_total /= (2**30)
            global_used = global_total - global_free

            self.gui.gpu_mem_gauge.setFormat(f'{global_used:.1f} GB / {global_total:.1f} GB')
            self.gui.gpu_mem_gauge.setValue(round(global_used / global_total * 100))

            used_by_torch = torch.cuda.max_memory_allocated() / (2**30)
            self.gui.torch_mem_gauge.setFormat(f'{used_by_torch:.1f} GB / {global_total:.1f} GB')
            self.gui.torch_mem_gauge.setValue(round(used_by_torch / global_total * 100 / 1024))
        elif 'mps' in self.device:
            mem_used = mps.current_allocated_memory() / (2**30)
            self.gui.gpu_mem_gauge.setFormat(f'{mem_used:.1f} GB')
            self.gui.gpu_mem_gauge.setValue(0)
            self.gui.torch_mem_gauge.setFormat('N/A')
            self.gui.torch_mem_gauge.setValue(0)
        else:
            self.gui.gpu_mem_gauge.setFormat('N/A')
            self.gui.gpu_mem_gauge.setValue(0)
            self.gui.torch_mem_gauge.setFormat('N/A')
            self.gui.torch_mem_gauge.setValue(0)

    def on_gpu_timer(self):
        self.update_gpu_gauges()

    def update_memory_gauges(self):
        try:
            curr_perm_tokens = self.processor.memory.work_mem.perm_size(0)
            self.gui.perm_mem_gauge.setFormat(f'{curr_perm_tokens} / {curr_perm_tokens}')
            self.gui.perm_mem_gauge.setValue(100)

            max_work_tokens = self.processor.memory.max_work_tokens
            max_long_tokens = self.processor.memory.max_long_tokens

            curr_work_tokens = self.processor.memory.work_mem.non_perm_size(0)
            curr_long_tokens = self.processor.memory.long_mem.non_perm_size(0)

            self.gui.work_mem_gauge.setFormat(f'{curr_work_tokens} / {max_work_tokens}')
            self.gui.work_mem_gauge.setValue(round(curr_work_tokens / max_work_tokens * 100))

            self.gui.long_mem_gauge.setFormat(f'{curr_long_tokens} / {max_long_tokens}')
            self.gui.long_mem_gauge.setValue(round(curr_long_tokens / max_long_tokens * 100))

        except AttributeError as e:
            self.gui.work_mem_gauge.setFormat('Unknown')
            self.gui.long_mem_gauge.setFormat('Unknown')
            self.gui.work_mem_gauge.setValue(0)
            self.gui.long_mem_gauge.setValue(0)

    def on_work_min_change(self):
        if self.initialized:
            self.gui.work_mem_min.setValue(
                min(self.gui.work_mem_min.value(),
                    self.gui.work_mem_max.value() - 1))
            self.update_config()

    def on_work_max_change(self):
        if self.initialized:
            self.gui.work_mem_max.setValue(
                max(self.gui.work_mem_max.value(),
                    self.gui.work_mem_min.value() + 1))
            self.update_config()

    def update_config(self):
        if self.initialized:
            with open_dict(self.cfg):
                self.cfg.long_term['min_mem_frames'] = self.gui.work_mem_min.value()
                self.cfg.long_term['max_mem_frames'] = self.gui.work_mem_max.value()
                self.cfg.long_term['max_num_tokens'] = self.gui.long_mem_max.value()
                self.cfg['mem_every'] = self.gui.mem_every_box.value()

            self.processor.update_config(self.cfg)

    def on_clear_memory(self):
        self.processor.clear_memory()
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        elif 'mps' in self.device:
            mps.empty_cache()
        # Clear mask cache to free memory
        self.res_man.clear_cache()
        self.processor.update_config(self.cfg)
        self.update_gpu_gauges()
        self.update_memory_gauges()

    def on_clear_non_permanent_memory(self):
        self.processor.clear_non_permanent_memory()
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        elif 'mps' in self.device:
            mps.empty_cache()
        # Clear mask cache to free memory
        self.res_man.clear_cache()
        self.processor.update_config(self.cfg)
        self.update_gpu_gauges()
        self.update_memory_gauges()

    def on_import_mask(self):
        file_name = self.gui.open_file('Mask')
        if len(file_name) == 0:
            return

        mask = self.res_man.import_mask(file_name, size=(self.h, self.w))

        shape_condition = ((len(mask.shape) == 2) and (mask.shape[-1] == self.w)
                           and (mask.shape[-2] == self.h))

        object_condition = (mask.max() <= self.num_objects)

        if not shape_condition:
            self.gui.text(f'Expected ({self.h}, {self.w}). Got {mask.shape} instead.')
        elif not object_condition:
            self.gui.text(f'Expected {self.num_objects} objects. Got {mask.max()} objects instead.')
        else:
            self.gui.text(f'Mask file {file_name} loaded.')
            self.curr_image_torch = self.curr_prob = None
            self.curr_mask = mask
            self.show_current_frame()
            self.save_current_mask()

    def on_import_layer(self):
        file_name = self.gui.open_file('Layer')
        if len(file_name) == 0:
            return

        self._try_load_layer(file_name)

    def _try_load_layer(self, file_name):
        try:
            layer = self.res_man.import_layer(file_name, size=(self.h, self.w))

            self.gui.text(f'Layer file {file_name} loaded.')
            self.overlay_layer = layer
            self.overlay_layer_torch = torch.from_numpy(layer).float().to(self.device) / 255
            self.show_current_frame()
        except FileNotFoundError:
            self.gui.text(f'{file_name} not found.')

    def on_set_save_visualization_mode(self):
        self.save_visualization_mode = self.gui.save_visualization_combo.currentText()

    def on_save_soft_mask_toggle(self):
        """Handle save soft mask checkbox toggle"""
        if hasattr(self, 'gui'):
            self.save_soft_mask = self.gui.save_soft_mask_checkbox.isChecked()
            print(f"Save soft mask toggled: {self.save_soft_mask}")
        else:
            print("GUI not initialized yet")

    def on_save_all_visible_toggle(self):
        """Handle include all visible objects in combined masks checkbox toggle"""
        if hasattr(self, 'gui'):
            self.save_all_visible = self.gui.save_all_visible_checkbox.isChecked()
            print(f"Include all visible objects in combined masks toggled: {self.save_all_visible}")
            print(f"Combined masks will include: {'all visible objects' if self.save_all_visible else 'tracked objects only'}")
        else:
            print("GUI not initialized yet")

    def on_mouse_motion_xy(self, x, y):
        self.last_ex = x
        self.last_ey = y

    def on_wheel_event(self, event):
        """Handle mouse wheel events for zooming"""
        # Check if image is loaded
        if not hasattr(self.gui, 'image_size') or self.gui.image_size is None:
            event.accept()
            return
        
        # Get mouse position relative to canvas
        mouse_pos = event.position()
        
        # Calculate zoom factor change
        zoom_delta = 0.1
        if event.angleDelta().y() > 0:
            # Zoom in
            new_zoom = min(self.zoom_factor + zoom_delta, 5.0)  # Max 5x zoom
        else:
            # Zoom out
            new_zoom = max(self.zoom_factor - zoom_delta, 1.0)  # Min 1x zoom (no zoom out)
        
        if new_zoom != self.zoom_factor:
            # Calculate zoom center in image coordinates
            canvas_size = self.gui.main_canvas.size()
            img_size = self.gui.image_size
            
            # Calculate base scale
            scale_w = canvas_size.width() / img_size.width()
            scale_h = canvas_size.height() / img_size.height()
            base_scale = min(scale_w, scale_h)
            
            # Current scaled image size
            current_scaled_w = img_size.width() * base_scale * self.zoom_factor
            current_scaled_h = img_size.height() * base_scale * self.zoom_factor
            
            # Mouse position relative to canvas center
            canvas_center_x = canvas_size.width() / 2
            canvas_center_y = canvas_size.height() / 2
            mouse_offset_x = mouse_pos.x() - canvas_center_x
            mouse_offset_y = mouse_pos.y() - canvas_center_y
            
            # Adjust pan to zoom towards mouse position
            zoom_ratio = new_zoom / self.zoom_factor
            
            if self.zoom_factor > 1.0:
                # Adjust pan based on zoom center
                self.pan_x = (self.pan_x - mouse_offset_x) * zoom_ratio + mouse_offset_x
                self.pan_y = (self.pan_y - mouse_offset_y) * zoom_ratio + mouse_offset_y
            else:
                # Starting to zoom in, initialize pan
                self.pan_x = -mouse_offset_x * (new_zoom - 1.0)
                self.pan_y = -mouse_offset_y * (new_zoom - 1.0)
            
            self.zoom_factor = new_zoom
            self.gui._update_canvas_display()
        
        event.accept()

    def on_mouse_press_for_pan(self, event):
        """Handle mouse press for panning"""
        from PySide6.QtCore import Qt
        
        # Handle panning with middle mouse button or Ctrl+Left button
        if (event.button() == Qt.MouseButton.MiddleButton or 
            (event.button() == Qt.MouseButton.LeftButton and 
             event.modifiers() == Qt.KeyboardModifier.ControlModifier)):
            if self.zoom_factor > 1.0:
                self.is_panning = True
                self.last_pan_pos = event.position()
                return True  # Event handled
        return False  # Event not handled, let GUI handle it

    def on_mouse_motion_for_pan(self, event):
        """Handle mouse motion for panning"""
        if self.is_panning and self.last_pan_pos is not None:
            current_pos = event.position()
            dx = current_pos.x() - self.last_pan_pos.x()
            dy = current_pos.y() - self.last_pan_pos.y()
            
            self.pan_x += dx
            self.pan_y += dy
            self.last_pan_pos = current_pos
            
            self.gui._update_canvas_display()
            return True  # Event handled
        return False  # Event not handled

    def on_mouse_release_for_pan(self, event):
        """Handle mouse release for panning"""
        if self.is_panning:
            self.is_panning = False
            self.last_pan_pos = None
            return True  # Event handled
        return False  # Event not handled

    def zoom_in(self):
        """Zoom in by 0.2x"""
        new_zoom = min(self.zoom_factor + 0.2, 5.0)
        if new_zoom != self.zoom_factor:
            self.zoom_factor = new_zoom
            self.gui._update_canvas_display()

    def zoom_out(self):
        """Zoom out by 0.2x"""
        new_zoom = max(self.zoom_factor - 0.2, 1.0)
        if new_zoom != self.zoom_factor:
            self.zoom_factor = new_zoom
            if self.zoom_factor == 1.0:
                self.pan_x = 0.0
                self.pan_y = 0.0
            self.gui._update_canvas_display()

    def reset_zoom(self):
        """Reset zoom and pan to default"""
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.gui._update_canvas_display()

    @property
    def h(self) -> int:
        return self.res_man.h

    @property
    def w(self) -> int:
        return self.res_man.w

    @property
    def T(self) -> int:
        return self.res_man.T

    def on_export_mask_metrics(self):
        output_filename = self.gui.export_dialog.mask_metrics_filename.text()
        if not output_filename.endswith('.csv'):
            output_filename += '.csv'
            
        # Handle both cases: just filename or full path
        # if Path(output_filename).is_absolute():
            # GUI contains full path, use it directly
        output_path = Path(output_filename)
        # else:
        #     # GUI contains just filename, construct full path in workspace
        #     output_path = Path(self.cfg['workspace']) / output_filename
            
        print(f"\nExporting mask metrics to {output_path}")
        print(f"Number of objects: {self.num_objects}")
        print(f"Object names: {self.name_objects}")
            
        mask_folder = Path(self.cfg['workspace']) / 'masks'
        if not mask_folder.exists():
            print(f"ERROR: Mask folder not found at {mask_folder}")
            self.gui.text('No masks folder found. Please track some objects first.')
            return
            
        print(f"Found mask folder at {mask_folder}")
        mask_files = list(mask_folder.glob('*.png'))
        print(f"Found {len(mask_files)} mask files")
        
        # Check if previous metrics file exists in workspace directory
        previous_df = None
        print(f"Looking for previous metrics file at: {output_path}")
        print(f"Workspace directory: {self.cfg['workspace']}")
        print(f"Output filename from GUI: {output_filename}")
        
        if output_path.exists():
            try:
                print(f"Found existing metrics file: {output_path}")
                previous_df = pd.read_csv(output_path)
                print(f"Loaded previous metrics with {len(previous_df)} rows")
                print(f"Previous metrics columns: {previous_df.columns.tolist()}")
                
                # Validate that the previous dataframe has the expected structure
                expected_columns = ['frame', 'object_id', 'object_name', 'area', 'perimeter', 
                                  'circularity', 'orientation', 'bbox_x', 'bbox_y', 
                                  'bbox_width', 'bbox_height', 'center_x', 'center_y']
                missing_columns = [col for col in expected_columns if col not in previous_df.columns]
                if missing_columns:
                    print(f"WARNING: Previous metrics file missing columns: {missing_columns}")
                    print("Will recalculate all metrics")
                    previous_df = None
                else:
                    print("Previous metrics file structure is valid")
                    
            except Exception as e:
                print(f"WARNING: Failed to load previous metrics file: {str(e)}")
                print("Will recalculate all metrics")
                previous_df = None
        else:
            print(f"No previous metrics file found at {output_path}, will calculate all metrics")
            
        try:
            print("\nCalculating mask metrics...")
            self.gui.text('Calculating mask metrics...')
            self.gui.progressbar_update(0.0)
            self.gui.process_events()

            def _mask_metrics_progress(p: float):
                self.gui.progressbar_update(p)
                self.gui.process_events()

            df = calculate_mask_metrics_batch(
                mask_folder,
                self.num_objects,
                self.name_objects,
                previous_df,
                progress_callback=_mask_metrics_progress,
            )

            if df.empty:
                print("WARNING: No metrics were calculated - DataFrame is empty")
                self.gui.text('No mask metrics were calculated. Please check if masks exist and contain valid objects.')
                return
                
            print(f"\nWriting metrics to CSV file...")
            print(f"DataFrame shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            df.to_csv(output_path, index=False)
            print(f"Successfully wrote {len(df)} rows to {output_path}")
            self.gui.progressbar_update(1.0)
            self.gui.process_events()

            # Provide feedback about what was calculated
            if previous_df is not None:
                new_rows = len(df) - len(previous_df)
                if new_rows > 0:
                    self.gui.text(f'Successfully exported mask metrics to {output_filename} (added {new_rows} new rows)')
                else:
                    self.gui.text(f'Successfully exported mask metrics to {output_filename} (no new calculations needed)')
            else:
                self.gui.text(f'Successfully exported mask metrics to {output_filename}')
            self.gui.progressbar_update(0.0)

            # If any pairwise metric checkbox is selected, also export pairwise metrics (.npz)
            if (self.gui.export_dialog.distance_cb.isChecked() or self.gui.export_dialog.overlap_cb.isChecked() or self.gui.export_dialog.contact_cb.isChecked()):
                pairwise_output_path = Path(self.cfg['workspace']) / "pairwise_metrics.npz"
                self.gui.export_dialog.export_mask_metrics_button.setEnabled(False)
                self.gui.export_dialog.export_mask_metrics_button.setText("Calculating...")
                self.gui.progressbar_update(0.0)
                self.pairwise_worker = PairwiseMetricsWorker(
                    mask_folder,
                    self.num_objects,
                    pairwise_output_path,
                    batch_size=self.pairwise_metrics_batch_size,
                    max_workers=self.pairwise_metrics_max_workers,
                    optimization_level=self.pairwise_metrics_optimization_level
                )
                self.pairwise_thread = QThread()
                self.pairwise_worker.moveToThread(self.pairwise_thread)
                self.pairwise_thread.started.connect(self.pairwise_worker.run)
                self.pairwise_worker.finished.connect(self.pairwise_thread.quit)
                self.pairwise_worker.finished.connect(self.pairwise_worker.deleteLater)
                self.pairwise_thread.finished.connect(self.pairwise_thread.deleteLater)
                self.pairwise_worker.progress.connect(self.gui.text)
                self.pairwise_worker.progress_value.connect(self.gui.progressbar_update)
                self.pairwise_worker.success.connect(self.gui.text)
                self.pairwise_worker.error.connect(self.gui.text)
                self.pairwise_thread.finished.connect(self._on_pairwise_metrics_finished)
                self.pairwise_thread.start()
                return  # Button re-enabled in _on_pairwise_metrics_finished
                
        except Exception as e:
            print(f"ERROR: Failed to export mask metrics: {str(e)}")
            self.gui.text(f'Error exporting mask metrics: {str(e)}')

    def _on_pairwise_metrics_finished(self):
        """Clean up after pairwise metrics calculation is complete"""
        # Re-enable the Export Mask Metrics button
        self.gui.export_dialog.export_mask_metrics_button.setEnabled(True)
        self.gui.export_dialog.export_mask_metrics_button.setText("Export Mask Metrics")
        # Reset progress bar
        self.gui.progressbar_update(0.0)

    def on_vis_checkbox_change(self, obj_id: int, state: int):
        """Handle visibility checkbox state change"""
        print(f"Visibility checkbox changed for object {obj_id}: {state == Qt.CheckState.Checked.value}")
        if state == Qt.CheckState.Checked.value:
            self.visible_objects.add(obj_id)
        else:
            self.visible_objects.discard(obj_id)
            # If show is unchecked, also uncheck track (can't track what you can't see)
            if obj_id in self.tracked_objects:
                self.tracked_objects.discard(obj_id)
                # Update the GUI checkbox state
                if hasattr(self, 'gui') and hasattr(self.gui, 'track_checkboxes'):
                    checkbox_index = obj_id - 1  # Convert to 0-based index
                    if 0 <= checkbox_index < len(self.gui.track_checkboxes):
                        self.gui.track_checkboxes[checkbox_index].setChecked(False)
                        print(f"Automatically unchecked track for object {obj_id} because show was unchecked")
        print(f"Visible objects: {self.visible_objects}")
        print(f"Tracked objects: {self.tracked_objects}")
        self.show_current_frame()

    def on_track_checkbox_change(self, obj_id: int, state: int):
        """Handle tracking checkbox state change"""
        print(f"Tracking checkbox changed for object {obj_id}: {state == Qt.CheckState.Checked.value}")
        was_tracked = obj_id in self.tracked_objects
        
        if state == Qt.CheckState.Checked.value:
            self.tracked_objects.add(obj_id)
            # If track is checked, also check show (you want to see what you're tracking)
            if obj_id not in self.visible_objects:
                self.visible_objects.add(obj_id)
                # Update the GUI checkbox state
                if hasattr(self, 'gui') and hasattr(self.gui, 'vis_checkboxes'):
                    checkbox_index = obj_id - 1  # Convert to 0-based index
                    if 0 <= checkbox_index < len(self.gui.vis_checkboxes):
                        self.gui.vis_checkboxes[checkbox_index].setChecked(True)
                        print(f"Automatically checked show for object {obj_id} because track was checked")
        else:
            self.tracked_objects.discard(obj_id)
        
        print(f"Visible objects: {self.visible_objects}")
        print(f"Tracked objects: {self.tracked_objects}")
        
        # If an object was newly tracked (wasn't tracked before, now is), reload mask from all_masks
        # and save it to masks folder if needed
        if state == Qt.CheckState.Checked.value and not was_tracked:
            print(f"Object {obj_id} newly tracked, reloading mask to include it from all_masks if available")
            
            # Check if mask exists in masks folder for current frame (get full mask, not filtered)
            # Pass None to get_mask to get all objects from the file
            mask_from_masks_folder = self.res_man.get_mask(self.curr_ti, tracked_objects=None)
            
            if mask_from_masks_folder is None:
                # No mask in masks folder, need to load from all_masks and save to masks folder
                print(f"No mask in masks folder for frame {self.curr_ti}, loading from all_masks...")
                
                # Force load from all_masks (skip masks folder); then save combined mask
                self.load_current_image_mask(force_from_all_masks=True)
                self.convert_current_image_mask_torch()
                print(f"Saving updated mask to masks folder with newly tracked object {obj_id}")
                self.save_current_mask()
                self.curr_frame_dirty = False
            else:
                # Mask exists in masks folder; check if it has all currently tracked objects
                unique_in_mask = set(int(x) for x in np.unique(mask_from_masks_folder) if x > 0)
                missing_tracked = self.tracked_objects - unique_in_mask
                if missing_tracked:
                    # Masks folder is incomplete (e.g. only object 2 after modify); rebuild from all_masks
                    print(f"Masks folder incomplete: missing tracked objects {missing_tracked}, loading from all_masks...")
                    self.load_current_image_mask(force_from_all_masks=True)
                    self.convert_current_image_mask_torch()
                    print(f"Saving updated mask to masks folder with all {len(self.tracked_objects)} tracked objects")
                    self.save_current_mask()
                    self.curr_frame_dirty = False
                else:
                    # All tracked objects already in mask, just reload for consistency
                    print(f"Object {obj_id} already in masks folder mask, reloading for consistency")
                    self.load_current_image_mask()
                    self.convert_current_image_mask_torch()
        elif state != Qt.CheckState.Checked.value and was_tracked:
            # Object was untracked, just reload mask (will exclude it)
            print(f"Object {obj_id} untracked, reloading mask to exclude it")
            self.load_current_image_mask()
            self.convert_current_image_mask_torch()
            # Save the updated mask (without the untracked object)
            self.save_current_mask()
            self.curr_frame_dirty = False
        
        self.show_current_frame()

    def clear_mask_cache(self):
        """Clear mask cache to free memory"""
        self.res_man.clear_cache()
        self.gui.text("Mask cache cleared to free memory.")

    def update_performance_stats(self, frame_time: float = None):
        """Update performance statistics"""
        if frame_time is None:
            frame_time = time.time() - self.performance_stats['last_frame_time']
        
        self.performance_stats['frames_processed'] += 1
        self.performance_stats['total_processing_time'] += frame_time
        self.performance_stats['avg_fps'] = self.performance_stats['frames_processed'] / self.performance_stats['total_processing_time']
        self.performance_stats['last_frame_time'] = time.time()
        
        # Update GUI with performance info
        if hasattr(self, 'gui') and self.performance_stats['frames_processed'] % 100 == 0:
            fps = self.performance_stats['avg_fps']
            self.gui.text(f"Performance: {fps:.1f} FPS, {self.performance_stats['frames_processed']} frames processed")

    def get_performance_stats(self) -> dict:
        """Get current performance statistics"""
        return self.performance_stats.copy()

    def update_checkbox_states(self):
        """Update checkbox states to reflect logical relationship between Show and Track"""
        if not hasattr(self, 'gui') or not hasattr(self.gui, 'vis_checkboxes') or not hasattr(self.gui, 'track_checkboxes'):
            return
            
        for obj_id in range(1, self.num_objects + 1):
            checkbox_index = obj_id - 1  # Convert to 0-based index
            if 0 <= checkbox_index < len(self.gui.vis_checkboxes) and 0 <= checkbox_index < len(self.gui.track_checkboxes):
                # Update track checkbox
                self.gui.track_checkboxes[checkbox_index].setChecked(obj_id in self.tracked_objects)
                # Update show checkbox
                self.gui.vis_checkboxes[checkbox_index].setChecked(obj_id in self.visible_objects)


class PairwiseMetricsWorker(QObject):
    """Worker class for calculating pairwise metrics in background thread"""
    finished = Signal()
    progress = Signal(str)
    progress_value = Signal(float)  # For progress bar updates (0.0 to 1.0)
    error = Signal(str)
    success = Signal(str)
    
    def __init__(self, mask_folder, num_objects, output_path, batch_size=50, max_workers=8, optimization_level='mega'):
        super().__init__()
        self.mask_folder = mask_folder
        self.num_objects = num_objects
        self.output_path = output_path
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.optimization_level = optimization_level
        
    def run(self):
        try:
            # Start performance monitoring
            start_global_monitoring(interval=2.0)
            print("Performance monitoring started")
            
            self.progress.emit("Calculating pairwise metrics (optimized)...")
            self.progress_value.emit(0.1)
            
            # Choose optimization level based on configuration
            print(f"Starting {self.optimization_level}-optimized pairwise metrics calculation...")
            print(f"  - Batch size: {self.batch_size}")
            print(f"  - Max workers: {self.max_workers}")
            print(f"  - Objects: {self.num_objects}")
            print(f"  - Optimization level: {self.optimization_level}")
            
            try:
                metrics_dict = calculate_all_pairwise_metrics_optimized(
                    str(self.mask_folder), 
                    self.num_objects, 
                    max_workers=self.max_workers
                )
            except Exception as optimization_error:
                error_msg = f"{self.optimization_level.capitalize()} optimization failed, falling back to standard version: {str(optimization_error)}"
                print(error_msg)
                self.progress.emit(error_msg)
                # Fallback to the standard optimized version
                print(f"Starting standard optimized pairwise metrics calculation...")
                metrics_dict = calculate_all_pairwise_metrics_optimized(
                    str(self.mask_folder), 
                    self.num_objects, 
                    max_workers=self.max_workers
                )
            
            self.progress_value.emit(0.8)
            
            self.progress.emit("Saving metrics to file...")
            print("Saving pairwise metrics to file...")
            self.progress_value.emit(0.9)
            
            # Save metrics
            save_pairwise_metrics(metrics_dict, str(self.output_path))
            self.progress_value.emit(1.0)
            
            # Stop performance monitoring and print summary
            stop_global_monitoring()
            print_global_summary()
            
            success_msg = f'Successfully saved pairwise metrics to {self.output_path.name}'
            print(success_msg)
            self.success.emit(success_msg)
            self.finished.emit()
            
        except Exception as e:
            # Stop performance monitoring on error
            stop_global_monitoring()
            
            error_msg = f'Error saving pairwise metrics: {str(e)}'
            print(error_msg)
            self.error.emit(error_msg)
            self.finished.emit()
