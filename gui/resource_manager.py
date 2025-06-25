import os
from os import path
import shutil
import collections
import logging
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from omegaconf import DictConfig, open_dict
from typing import Dict, Optional, Tuple, Literal, Union, List
import cv2
from PIL import Image
if not hasattr(Image, 'Resampling'):  # Pillow<9.0
    Image.Resampling = Image
import numpy as np

from cutie.utils.palette import davis_palette, davis_palette_np
from tqdm import tqdm

log = logging.getLogger()


# https://bugs.python.org/issue28178
# ah python ah why
class LRU:

    def __init__(self, func, maxsize=128):
        self.cache = collections.OrderedDict()
        self.func = func
        self.maxsize = maxsize

    def __call__(self, *args):
        cache = self.cache
        if args in cache:
            cache.move_to_end(args)
            return cache[args]
        result = self.func(*args)
        cache[args] = result
        if len(cache) > self.maxsize:
            cache.popitem(last=False)
        return result

    def invalidate(self, key):
        self.cache.pop(key, None)


@dataclass
class SaveItem:
    type: Literal['mask', 'visualization', 'soft_mask', 'batch_soft_mask']
    data: Union[Image.Image, np.ndarray, Dict]
    name: str = None  # only used for soft_mask


class ResourceManager:

    def __init__(self, cfg: DictConfig):
        # determine inputs
        images = cfg['images']
        video = cfg['video']
        self.workspace = cfg['workspace']
        self.max_size = cfg['max_overall_size']
        self.palette = davis_palette_np.flatten().tolist()  # Convert to list of RGB values

        # create temporary workspace if not specified
        if self.workspace is None:
            if images is not None:
                basename = path.basename(images)
            elif video is not None:
                basename = path.basename(video)[:-4]
            else:
                raise NotImplementedError('Either images, video, or workspace has to be specified')

            self.workspace = path.join('./workspace', basename)

        print(f'Workspace is in: {self.workspace}')
        with open_dict(cfg):
            cfg['workspace'] = self.workspace

        # determine the location of input images
        need_decoding = False
        need_resizing = False
        if path.exists(path.join(self.workspace, 'images')):
            pass
        elif images is not None:
            need_resizing = True
        elif video is not None:
            # will decode video into frames later
            need_decoding = True

        # create workspace subdirectories
        self.image_dir = path.join(self.workspace, 'images')
        self.mask_dir = path.join(self.workspace, 'masks')
        self.visualization_dir = path.join(self.workspace, 'visualization')
        self.soft_mask_dir = path.join(self.workspace, 'soft_masks')
        self.all_masks_dir = path.join(self.workspace, 'all_masks')
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)
        os.makedirs(self.soft_mask_dir, exist_ok=True)
        os.makedirs(self.all_masks_dir, exist_ok=True)

        # create all soft mask sub-directories
        for i in range(1, cfg['num_objects'] + 1):
            os.makedirs(path.join(self.soft_mask_dir, f'{i}'), exist_ok=True)

        # convert read functions to be buffered
        self.get_image = LRU(self._get_image_unbuffered, maxsize=cfg['buffer_size'])
        self.get_mask = LRU(self._get_mask_unbuffered, maxsize=cfg['buffer_size'])

        # extract frames from video
        if need_decoding:
            self._extract_frames(video)

        # copy/resize existing images to the workspace
        if need_resizing:
            self._copy_resize_frames(images)

        # read all frame names
        self.names = sorted(os.listdir(self.image_dir))
        self.names = [f[:-4] for f in self.names]  # remove extensions
        self.length = len(self.names)

        assert self.length > 0, f'No images found! Check {self.workspace}/images. Remove folder if necessary.'

        print(f'{self.length} images found.')

        self.height, self.width = self.get_image(0).shape[:2]

        # create the saver threads for saving the masks/visualizations
        self.save_queue = Queue(maxsize=cfg['save_queue_size'])
        self.num_save_threads = cfg['num_save_threads']
        self.save_threads = [
            Thread(target=self.save_thread, args=(self.save_queue, ))
            for _ in range(self.num_save_threads)
        ]
        for t in self.save_threads:
            t.daemon = True
            t.start()

        # Performance optimization: In-memory mask cache
        self.mask_cache = {}  # Cache for combined masks
        self.soft_mask_cache = {}  # Cache for individual soft masks
        
        # Get performance settings from config
        self.batch_save_soft_masks = cfg.get('performance', {}).get('batch_save_soft_masks', True)
        self.enable_mask_cache = cfg.get('performance', {}).get('enable_mask_cache', True)
        self.cache_size_limit = cfg.get('performance', {}).get('max_cache_size', 50)
        self.lazy_saving = cfg.get('performance', {}).get('lazy_saving', True)
        self.save_only_tracked = cfg.get('performance', {}).get('save_only_tracked', True)
        
        # Disable cache if not enabled
        if not self.enable_mask_cache:
            self.mask_cache = None
            self.soft_mask_cache = None

    def __del__(self):
        # Check if attributes exist before trying to access them
        if hasattr(self, 'num_save_threads') and hasattr(self, 'save_queue') and hasattr(self, 'save_threads'):
            try:
                for _ in range(self.num_save_threads):
                    self.save_queue.put(None)
                self.save_queue.join()
                for t in self.save_threads:
                    t.join()
            except Exception as e:
                # Ignore errors during cleanup
                pass

    def save_thread(self, queue: Queue):
        while True:
            args: SaveItem = queue.get()
            if args is None:
                queue.task_done()
                break
            print(f"Processing save item: type={args.type}, name={args.name}")
            if args.type == 'mask':
                # PIL image
                args.data.save(path.join(self.mask_dir, args.name + '.png'))
            elif args.type.startswith('visualization'):
                # numpy array, save with cv2
                vis_mode = args.type.split('_')[-1]
                os.makedirs(path.join(self.visualization_dir, vis_mode), exist_ok=True)
                if vis_mode == 'rgba':
                    data = cv2.cvtColor(args.data, cv2.COLOR_RGBA2BGRA).copy()
                    cv2.imwrite(path.join(self.visualization_dir, vis_mode, args.name + '.png'),
                                data)
                else:
                    data = cv2.cvtColor(args.data, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(path.join(self.visualization_dir, vis_mode, args.name + '.jpg'),
                                data)
            elif args.type == 'soft_mask':
                # numpy array, save each channel with cv2
                print(f"Saving soft mask for {args.name}")
                num_channels = args.data.shape[0]
                # first channel is background -- ignore
                for i in range(1, num_channels):
                    data = args.data[i]
                    data = (data * 255).astype(np.uint8)
                    save_path = path.join(self.soft_mask_dir, f'{i}', args.name + '.png')
                    print(f"Saving to {save_path}")
                    cv2.imwrite(save_path, data)
            elif args.type == 'batch_soft_mask':
                # Optimized batch saving of soft masks
                self._save_batch_soft_masks(args.data, args.name)
            else:
                raise NotImplementedError
            queue.task_done()

    def _save_batch_soft_masks(self, batch_data: Dict, frame_name: str):
        """Optimized batch saving of soft masks"""
        try:
            frame_idx = batch_data.get('frame_idx')
            soft_masks = batch_data.get('soft_masks', {})  # {obj_id: mask_array}
            tracked_objects = batch_data.get('tracked_objects', set())
            
            # Save individual soft masks for tracked objects
            for obj_id, mask_array in soft_masks.items():
                # Only save if object is tracked or if save_only_tracked is False
                if not self.save_only_tracked or obj_id in tracked_objects:
                    save_path = os.path.join(self.soft_mask_dir, f'{obj_id}', f'{frame_idx:07d}.png')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    binary_mask = (mask_array > 0.5).astype(np.uint8) * 255
                    cv2.imwrite(save_path, binary_mask)
            
            # Create combined mask in memory and save
            if soft_masks:
                # Get dimensions from first mask
                first_mask = next(iter(soft_masks.values()))
                h, w = first_mask.shape
                combined_mask = np.zeros((h, w), dtype=np.uint8)
                
                # Combine all masks
                for obj_id, mask_array in soft_masks.items():
                    # Only include tracked objects in combined mask
                    if obj_id in tracked_objects:
                        binary_mask = (mask_array > 0.5).astype(np.uint8)
                        combined_mask[binary_mask > 0] = obj_id
                
                # Save combined mask
                mask_img = Image.fromarray(combined_mask, mode='P')
                mask_img.putpalette(self.palette)
                mask_img.save(os.path.join(self.all_masks_dir, f'{frame_idx:07d}.png'))
                
                # Cache the combined mask if enabled
                if self.enable_mask_cache and self.mask_cache is not None:
                    self.mask_cache[frame_idx] = combined_mask.copy()
                    
                    # Clean up cache if too large
                    if len(self.mask_cache) > self.cache_size_limit:
                        oldest_key = min(self.mask_cache.keys())
                        del self.mask_cache[oldest_key]
        except Exception as e:
            print(f"Error in batch soft mask saving: {str(e)}")
            # Fall back to individual saving if batch saving fails
            self._fallback_individual_saving(batch_data)

    def _fallback_individual_saving(self, batch_data: Dict):
        """Fallback method for individual soft mask saving if batch saving fails"""
        try:
            frame_idx = batch_data.get('frame_idx')
            soft_masks = batch_data.get('soft_masks', {})
            tracked_objects = batch_data.get('tracked_objects', set())
            
            for obj_id, mask_array in soft_masks.items():
                if obj_id in tracked_objects:
                    save_path = os.path.join(self.soft_mask_dir, f'{obj_id}', f'{frame_idx:07d}.png')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    binary_mask = (mask_array > 0.5).astype(np.uint8) * 255
                    cv2.imwrite(save_path, binary_mask)
        except Exception as e:
            print(f"Error in fallback individual saving: {str(e)}")

    def _extract_frames(self, video: str):
        cap = cv2.VideoCapture(video)
        frame_index = 0
        print(f'Extracting frames from {video} into {self.image_dir}...')
        with tqdm() as bar:
            while (cap.isOpened()):
                _, frame = cap.read()
                if frame is None:
                    break
                h, w = frame.shape[:2]
                if self.max_size > 0 and min(h, w) > self.max_size:
                    new_w = (w * self.max_size // min(w, h))
                    new_h = (h * self.max_size // min(w, h))
                    frame = cv2.resize(frame, dsize=(new_w, new_h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(path.join(self.image_dir, f'{frame_index:07d}.jpg'), frame)
                frame_index += 1
                bar.update()
        print('Done!')

    def _copy_resize_frames(self, images: str):
        image_list = os.listdir(images)
        print(f'Copying/resizing frames into {self.image_dir}...')
        for image_name in tqdm(image_list):
            if self.max_size < 0:
                # just copy
                shutil.copy2(path.join(images, image_name), self.image_dir)
            else:
                frame = cv2.imread(path.join(images, image_name))
                h, w = frame.shape[:2]
                if self.max_size > 0 and min(h, w) > self.max_size:
                    new_w = (w * self.max_size // min(w, h))
                    new_h = (h * self.max_size // min(w, h))
                    frame = cv2.resize(frame, dsize=(new_w, new_h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(path.join(self.image_dir, image_name), frame)
        print('Done!')

    def add_to_queue_with_warning(self, item: SaveItem):
        if self.save_queue.full():
            print(
                'The save queue is full! You need more threads or faster IO. Program might pause.')
        self.save_queue.put(item)

    def save_mask(self, ti: int, mask: np.ndarray):
        # mask should be uint8 H*W without channels
        assert 0 <= ti < self.length
        assert isinstance(mask, np.ndarray)

        # Convert to PIL Image in 'P' mode
        mask_img = Image.fromarray(mask, mode='P')
        
        # Ensure palette is in correct format (list of RGB values)
        if isinstance(self.palette, bytes):
            # Convert binary palette to list of RGB values
            palette_list = []
            for i in range(0, len(self.palette), 3):
                palette_list.extend([self.palette[i], self.palette[i+1], self.palette[i+2]])
            mask_img.putpalette(palette_list)
        else:
            # If palette is already in correct format, use directly
            mask_img.putpalette(self.palette)
            
        self.invalidate(ti)
        self.add_to_queue_with_warning(SaveItem('mask', mask_img, self.names[ti]))

    def save_visualization(self, ti: int, vis_mode: str, image: np.ndarray):
        # image should be uint8 3*H*W
        assert 0 <= ti < self.length
        assert isinstance(image, np.ndarray)

        # Save visualization in visualization/davis folder
        vis_dir = path.join(self.workspace, 'visualization', 'davis')
        os.makedirs(vis_dir, exist_ok=True)
        name = self.names[ti]

        self.add_to_queue_with_warning(SaveItem(f'visualization_{vis_mode}', image, name))

    def save_soft_mask(self, ti: int, prob: np.ndarray, obj_id: int = None):
        """Save soft mask for a specific object or all objects as binary images"""
        print(f"Saving soft mask for frame {ti}, obj_id: {obj_id}")
        if obj_id is not None:
            # Save individual object mask in its subfolder
            save_path = os.path.join(self.workspace, 'soft_masks', f'{obj_id}', f'{ti:07d}.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Convert probability to binary mask
            binary_mask = (prob > 0.5).astype(np.uint8) * 255
            cv2.imwrite(save_path, binary_mask)
            print(f"Saved soft mask to {save_path}")
        else:
            # Save all object masks
            for obj_id in range(1, prob.shape[0]):
                save_path = os.path.join(self.workspace, 'soft_masks', f'{obj_id}', f'{ti:07d}.png')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # Convert probability to binary mask
                binary_mask = (prob[obj_id] > 0.5).astype(np.uint8) * 255
                cv2.imwrite(save_path, binary_mask)
                print(f"Saved soft mask to {save_path}")
        
        # Update all_masks after saving soft masks
        print(f"Updating all_masks for frame {ti}")
        self.update_all_masks(ti)

    def save_batch_soft_masks(self, ti: int, soft_masks: Dict[int, np.ndarray], tracked_objects: set):
        """Optimized batch saving of soft masks"""
        batch_data = {
            'frame_idx': ti,
            'soft_masks': soft_masks,
            'tracked_objects': tracked_objects
        }
        self.add_to_queue_with_warning(SaveItem('batch_soft_mask', batch_data, self.names[ti]))

    def update_all_masks(self, ti: int):
        """Combine all available masks from soft_masks into a single mask in all_masks"""
        print(f"Updating all_masks for frame {ti}")
        # Get all object directories
        obj_dirs = [d for d in os.listdir(self.soft_mask_dir) if os.path.isdir(os.path.join(self.soft_mask_dir, d))]
        
        if not obj_dirs:
            print("No object directories found in soft_masks")
            return
            
        # Find first valid mask to get dimensions
        h, w = None, None
        for obj_id in obj_dirs:
            mask_path = os.path.join(self.soft_mask_dir, obj_id, f'{ti:07d}.png')
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    h, w = mask.shape
                    break
                    
        if h is None or w is None:
            print(f"No valid masks found for frame {ti}")
            return
            
        # Create a combined mask
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Combine masks from all objects
        for obj_id in obj_dirs:
            mask_path = os.path.join(self.soft_mask_dir, obj_id, f'{ti:07d}.png')
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Use object ID as the mask value
                    combined_mask[mask > 127] = int(obj_id)
        
        # Save combined mask in DAVIS format
        mask_img = Image.fromarray(combined_mask, mode='P')
        mask_img.putpalette(self.palette)
        mask_img.save(os.path.join(self.all_masks_dir, f'{ti:07d}.png'))
        print(f"Saved combined mask to {os.path.join(self.all_masks_dir, f'{ti:07d}.png')}")

    def get_all_masks(self, ti: int) -> np.ndarray:
        """Get the combined mask from all_masks directory or cache"""
        # Check cache first if enabled
        if self.enable_mask_cache and self.mask_cache is not None and ti in self.mask_cache:
            return self.mask_cache[ti]
            
        # Load from disk if not in cache
        mask_path = os.path.join(self.all_masks_dir, f'{ti:07d}.png')
        if os.path.exists(mask_path):
            mask = Image.open(mask_path)
            mask_array = np.array(mask)
            # Cache the result if enabled
            if self.enable_mask_cache and self.mask_cache is not None:
                self.mask_cache[ti] = mask_array
                
                # Clean up cache if too large
                if len(self.mask_cache) > self.cache_size_limit:
                    oldest_key = min(self.mask_cache.keys())
                    del self.mask_cache[oldest_key]
                    
            return mask_array
        return None

    def create_combined_mask_from_probabilities(self, ti: int, prob: np.ndarray, tracked_objects: set) -> np.ndarray:
        """Create combined mask directly from probability tensor without I/O"""
        if prob is None:
            return np.zeros((self.height, self.width), dtype=np.uint8)
            
        # Convert probability tensor to numpy
        prob_np = prob.cpu().numpy() if hasattr(prob, 'cpu') else prob
        
        # Create combined mask
        combined_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Add each tracked object to the combined mask
        for obj_id in range(1, prob_np.shape[0]):
            if obj_id in tracked_objects:
                obj_mask = (prob_np[obj_id] > 0.5).astype(np.uint8)
                combined_mask[obj_mask > 0] = obj_id
        
        # Cache the result if enabled
        if self.enable_mask_cache and self.mask_cache is not None:
            self.mask_cache[ti] = combined_mask.copy()
            
            # Clean up cache if too large
            if len(self.mask_cache) > self.cache_size_limit:
                oldest_key = min(self.mask_cache.keys())
                del self.mask_cache[oldest_key]
                
        return combined_mask

    def _get_image_unbuffered(self, ti: int):
        # returns H*W*3 uint8 array
        assert 0 <= ti < self.length

        image = Image.open(path.join(self.image_dir, self.names[ti] + '.jpg')).convert('RGB')
        image = np.array(image)
        return image

    def _get_mask_unbuffered(self, ti: int):
        # returns H*W uint8 array
        assert 0 <= ti < self.length

        mask_path = path.join(self.mask_dir, self.names[ti] + '.png')
        if path.exists(mask_path):
            mask = Image.open(mask_path)
            mask = np.array(mask)
            return mask
        else:
            return None

    def import_mask(self, file_name: str, size: Optional[Tuple[int, int]] = None):
        # read an mask file and resize it to exactly match the canvas size
        image = Image.open(file_name)
        if size is not None:
            # PIL uses (width, height)
            image = image.resize((size[1], size[0]), resample=Image.Resampling.NEAREST)
        image = np.array(image)
        return image

    def import_layer(self, file_name: str, size: Tuple[int, int]):
        # read a RGBA/RGB file and resize it such that the entire layer is visible in the canvas
        # and then pad it to the canvas size (h, w)
        image = Image.open(file_name).convert('RGBA')
        im_w, im_h = image.size
        im_ratio = im_w / im_h
        canvas_ratio = size[1] / size[0]
        if im_ratio < canvas_ratio:
            # fit height
            new_h = size[0]
            new_w = int(new_h * im_ratio)
        else:
            # fit width
            new_w = size[1]
            new_h = int(new_w / im_ratio)
        image = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        image = np.array(image)
        # padding
        pad_h = (size[0] - new_h) // 2
        pad_w = (size[1] - new_w) // 2
        image = np.pad(image,
                       ((pad_h, size[0] - new_h - pad_h), (pad_w, size[1] - new_w - pad_w), (0, 0)),
                       mode='constant',
                       constant_values=0)

        return image

    def invalidate(self, ti: int):
        # the image buffer is never invalidated
        self.get_mask.invalidate((ti, ))
        # Also invalidate cached masks if cache is enabled
        if self.enable_mask_cache:
            if self.mask_cache is not None and ti in self.mask_cache:
                del self.mask_cache[ti]
            if self.soft_mask_cache is not None and ti in self.soft_mask_cache:
                del self.soft_mask_cache[ti]

    def __len__(self):
        return self.length

    @property
    def T(self) -> int:
        return self.length

    @property
    def h(self) -> int:
        return self.height

    @property
    def w(self) -> int:
        return self.width

    def clear_cache(self):
        """Clear all cached masks to free memory"""
        if self.enable_mask_cache:
            if self.mask_cache is not None:
                self.mask_cache.clear()
            if self.soft_mask_cache is not None:
                self.soft_mask_cache.clear()
            print("Mask cache cleared")
