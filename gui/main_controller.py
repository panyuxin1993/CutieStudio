import os
from os import path
import logging
from pathlib import Path
import time

# fix conflicts between qt5 and cv2
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

import torch
try:
    from torch import mps
except:
    print('torch.MPS not available.')
from torch import autocast
from torchvision.transforms.functional import to_tensor
from omegaconf import DictConfig, open_dict
from PySide6.QtCore import Qt

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
from utils.mask_metrics import calculate_mask_metrics_batch, calculate_all_pairwise_metrics, save_pairwise_metrics

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
        
        print(f"Performance settings:")
        print(f"  - Batch save soft masks: {self.batch_save_soft_masks}")
        print(f"  - Enable mask cache: {self.enable_mask_cache}")
        print(f"  - Lazy saving: {self.lazy_saving}")
        print(f"  - Save only tracked: {self.save_only_tracked}")
        
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

    def load_current_image_mask(self, no_mask: bool = False):
        """Load the current frame's image and mask"""
        print(f"Loading current image mask for frame {self.curr_ti}")
        try:
            self.curr_image_np = self.res_man.get_image(self.curr_ti)
            print(f"Loaded image shape: {self.curr_image_np.shape}")
            self.curr_image_torch = None

            if not no_mask:
                loaded_mask = self.res_man.get_all_masks(self.curr_ti)
                if loaded_mask is None:
                    print("No mask found, using empty mask")
                    self.curr_mask.fill(0)
                else:
                    print("Loaded existing mask")
                    # # Get unique object IDs and remap them to sequential indices
                    # unique_ids = np.unique(loaded_mask)
                    # object_ids = [id for id in unique_ids if id > 0]
                    # id_map = {id: idx+1 for idx, id in enumerate(object_ids)}  # 0 stays 0 (background)
                    # remapped_mask = np.zeros_like(loaded_mask)
                    # for orig_id, new_id in id_map.items():
                    #     remapped_mask[loaded_mask == orig_id] = new_id
                    # print(f"Remapped object IDs: {id_map}")
                    # self.curr_mask = remapped_mask
                    self.curr_mask = loaded_mask.copy()
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
        # Get combined mask from all_masks or create from current probabilities
        combined_mask = self.res_man.get_all_masks(self.curr_ti)
        if combined_mask is None and self.curr_prob is not None:
            # Create combined mask directly from current probabilities
            combined_mask = self.res_man.create_combined_mask_from_probabilities(
                self.curr_ti, self.curr_prob, self.tracked_objects
            )
        elif combined_mask is None:
            # If no combined mask exists, create an empty mask
            combined_mask = np.zeros((self.h, self.w), dtype=np.uint8)
            
        # Only show masks for visible objects
        visible_mask = combined_mask.copy()
        for obj_id in range(1, self.num_objects + 1):
            if obj_id not in self.visible_objects:
                visible_mask[visible_mask == obj_id] = 0
                
        # Use visible_objects for basic visualization, vis_target_objects for popup/layer modes
        target_objects = self.vis_target_objects if self.vis_mode in ['popup', 'layer'] else list(self.visible_objects)
        self.vis_image = get_visualization(self.vis_mode, self.curr_image_np, visible_mask,
                                           self.overlay_layer, target_objects)

    def update_canvas(self):
        self.compose_current_im()
        self.gui.set_canvas(self.vis_image)

    def update_current_image_fast(self, invalid_soft_mask: bool = False):
        """Update the current image with fast visualization"""
        if self.curr_prob is None:
            return
            
        # Save soft masks for tracked objects using batch saving
        if self.save_soft_mask:
            print(f"Saving soft masks for tracked objects: {self.tracked_objects}")
            # Prepare batch data
            soft_masks = {}
            for obj_id in range(1, self.num_objects + 1):
                if obj_id in self.tracked_objects:
                    obj_mask = self.curr_prob[obj_id].cpu().numpy()
                    soft_masks[obj_id] = obj_mask
            
            # Use batch saving for better performance
            if soft_masks:
                self.res_man.save_batch_soft_masks(self.curr_ti, soft_masks, self.tracked_objects)
                    
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
            
        # Only show masks for visible objects
        visible_prob = self.curr_prob.clone()
        for obj_id in range(1, self.num_objects + 1):
            if obj_id not in self.visible_objects:
                visible_prob[obj_id] = 0
                
        # Use visible_objects for basic visualization, vis_target_objects for popup/layer modes
        target_objects = self.vis_target_objects if self.vis_mode in ['popup', 'layer'] else list(self.visible_objects)
        
        # Get visualization
        self.vis_image = get_visualization_torch(self.vis_mode, self.curr_image_torch,
                                               visible_prob, self.overlay_layer_torch,
                                               target_objects)
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
        # Save inference masks in masks folder
        self.res_man.save_mask(self.curr_ti, self.curr_mask)

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
                # Save soft masks for tracked objects using batch saving
                if self.save_soft_mask:
                    # Prepare batch data for all tracked objects
                    soft_masks = {}
                    for obj_id in range(1, self.num_objects + 1):
                        if obj_id in self.tracked_objects:
                            obj_mask = self.curr_prob[obj_id].cpu().numpy()
                            soft_masks[obj_id] = obj_mask
                    
                    # Use batch saving for better performance
                    if soft_masks:
                        self.res_man.save_batch_soft_masks(self.curr_ti, soft_masks, self.tracked_objects)
                self.show_current_frame(fast=True)

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

            self.gui.text(f'Propagation started at t={self.curr_ti}.')
            self.processor.clear_sensory_memory()
            self.curr_prob = self.processor.step(self.curr_image_torch,
                                                 self.curr_prob[1:],
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

            # propagate till the end
            for data in loader:
                if not self.propagating:
                    break
                self.curr_image_np, self.curr_image_torch = data
                self.curr_image_torch = self.curr_image_torch.to(self.device, non_blocking=True)
                self.propagate_fn()

                self.curr_prob = self.processor.step(self.curr_image_torch)
                self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)

                self.save_current_mask()
                # Save soft masks for tracked objects using batch saving
                if self.save_soft_mask:
                    # Prepare batch data for all tracked objects
                    soft_masks = {}
                    for obj_id in range(1, self.num_objects + 1):
                        if obj_id in self.tracked_objects:
                            obj_mask = self.curr_prob[obj_id].cpu().numpy()
                            soft_masks[obj_id] = obj_mask
                    
                    # Use batch saving for better performance
                    if soft_masks:
                        self.res_man.save_batch_soft_masks(self.curr_ti, soft_masks, self.tracked_objects)
                self.show_current_frame(fast=True)

                self.update_memory_gauges()
                self.gui.process_events()

                # stop the loop after one frame and clear memory
                self.propagating = False
                self.on_clear_memory()
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
        if self.interacted_prob is None:
            # get mask from disk
            self.load_current_image_mask()
        else:
            # get mask from interaction
            self.complete_interaction()
            self.update_interacted_mask()

        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            self.convert_current_image_mask_torch()
            self.gui.text(f'Permanent memory saved at {self.curr_ti}.')
            self.curr_prob = self.processor.step(self.curr_image_torch,
                                                 self.curr_prob[1:],
                                                 idx_mask=False,
                                                 force_permanent=True)
            self.update_memory_gauges()
            self.update_gpu_gauges()

    def on_play_video_timer(self):
        self.curr_ti += 1
        if self.curr_ti > self.T - 1:
            self.curr_ti = 0
        self.gui.tl_slider.setValue(self.curr_ti)

    def on_export_visualization(self):
        # NOTE: Save visualization at the end of propagation
        image_folder = path.join(self.cfg['workspace'], 'visualization', self.vis_mode)
        save_folder = self.cfg['workspace']
        if path.exists(image_folder):
            # Sorted so frames will be in order
            output_path = path.join(save_folder, f'visualization_{self.vis_mode}.mp4')
            self.gui.text(f'Exporting visualization -- please wait')
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

    def on_fps_dial_change(self):
        self.output_fps = self.gui.fps_dial.value()

    def on_bitrate_dial_change(self):
        self.output_bitrate = self.gui.bitrate_dial.value()

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
        self.curr_mask.fill(0)
        if self.curr_prob is not None:
            self.curr_prob.fill_(0)
        self.curr_frame_dirty = True
        self.save_current_mask()
        self.reset_this_interaction()
        self.show_current_frame()

    def on_reset_object(self):
        self.curr_mask[self.curr_mask == self.curr_object] = 0
        if self.curr_prob is not None:
            self.curr_prob[self.curr_object] = 0
        self.curr_frame_dirty = True
        self.save_current_mask()
        self.reset_this_interaction()
        self.show_current_frame()

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

    def on_mouse_motion_xy(self, x, y):
        self.last_ex = x
        self.last_ey = y

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
        output_filename = self.gui.mask_metrics_filename.text()
        if not output_filename.endswith('.csv'):
            output_filename += '.csv'
            
        print(f"\nExporting mask metrics to {output_filename}")
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
            
        try:
            print("\nCalculating mask metrics...")
            df = calculate_mask_metrics_batch(mask_folder, self.num_objects, self.name_objects)
            
            if df.empty:
                print("WARNING: No metrics were calculated - DataFrame is empty")
                self.gui.text('No mask metrics were calculated. Please check if masks exist and contain valid objects.')
                return
                
            print(f"\nWriting metrics to CSV file...")
            print(f"DataFrame shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            df.to_csv(output_filename, index=False)
            print(f"Successfully wrote {len(df)} rows to {output_filename}")
            self.gui.text(f'Successfully exported mask metrics to {output_filename}')
        except Exception as e:
            print(f"ERROR: Failed to export mask metrics: {str(e)}")
            self.gui.text(f'Error exporting mask metrics: {str(e)}')

    def on_save_pairwise_metrics(self):
        """Handle saving pairwise metrics"""
        # Get selected metrics
        selected_metrics = []
        if self.gui.distance_cb.isChecked():
            selected_metrics.append('distance')
        if self.gui.overlap_cb.isChecked():
            selected_metrics.append('overlap_ratio')
        if self.gui.contact_cb.isChecked():
            selected_metrics.append('contact_length')
            
        if not selected_metrics:
            self.gui.text("Please select at least one pairwise metric")
            return
            
        # Get output filename
        output_filename = "pairwise_metrics.npz"
        
        try:
            # Calculate metrics
            mask_folder = Path(self.cfg['workspace']) / 'masks'
            metrics_dict = calculate_all_pairwise_metrics(str(mask_folder), self.num_objects)
            
            # Save metrics
            output_path = Path(self.cfg['workspace']) / output_filename
            save_pairwise_metrics(metrics_dict, str(output_path))
            
            self.gui.text(f'Successfully saved pairwise metrics to {output_filename}')
        except Exception as e:
            self.gui.text(f'Error saving pairwise metrics: {str(e)}')

    def on_vis_checkbox_change(self, obj_id: int, state: int):
        """Handle visibility checkbox state change"""
        print(f"Visibility checkbox changed for object {obj_id}: {state == Qt.CheckState.Checked.value}")
        if state == Qt.CheckState.Checked.value:
            self.visible_objects.add(obj_id)
        else:
            self.visible_objects.discard(obj_id)
        print(f"Visible objects: {self.visible_objects}")
        self.show_current_frame()

    def on_track_checkbox_change(self, obj_id: int, state: int):
        """Handle tracking checkbox state change"""
        print(f"Tracking checkbox changed for object {obj_id}: {state == Qt.CheckState.Checked.value}")
        if state == Qt.CheckState.Checked.value:
            self.tracked_objects.add(obj_id)
        else:
            self.tracked_objects.discard(obj_id)
        print(f"Tracked objects: {self.tracked_objects}")
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
        if hasattr(self, 'gui') and self.performance_stats['frames_processed'] % 10 == 0:
            fps = self.performance_stats['avg_fps']
            self.gui.text(f"Performance: {fps:.1f} FPS, {self.performance_stats['frames_processed']} frames processed")

    def get_performance_stats(self) -> dict:
        """Get current performance statistics"""
        return self.performance_stats.copy()
