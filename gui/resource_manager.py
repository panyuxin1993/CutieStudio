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

log = logging.getLogger(__name__)


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
        self.num_objects = cfg['num_objects']
        for i in range(1, self.num_objects + 1):
            os.makedirs(path.join(self.soft_mask_dir, f'{i}'), exist_ok=True)

        # convert read functions to be buffered
        self.get_image = LRU(self._get_image_unbuffered, maxsize=cfg['buffer_size'])
        # Note: get_mask now requires tracked_objects parameter, so we don't use LRU cache
        # Call _get_mask_unbuffered directly with tracked_objects parameter
        # For convenience, create a simple wrapper
        self.get_mask = self._get_mask_unbuffered

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
            log.debug('Processing save item: type=%s name=%s', args.type, args.name)
            if args.type == 'mask':
                # PIL image
                args.data.save(path.join(self.mask_dir, args.name + '.png'))
            elif args.type.startswith('visualization'):
                vis_mode = args.type.split('_')[-1]
                self._write_visualization_file(vis_mode, args.data, args.name)
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
        """Save all_masks NPZ from in-memory probabilities; optionally legacy per-object PNGs."""
        try:
            frame_idx = batch_data.get('frame_idx')
            soft_masks = batch_data.get('soft_masks', {})
            tracked_objects = batch_data.get('tracked_objects', set())
            save_all_visible = batch_data.get('save_all_visible', True)
            skip_disk_fallback = batch_data.get('skip_disk_fallback', False)
            all_masks_only = batch_data.get('all_masks_only', True)

            if not all_masks_only:
                for obj_id, mask_array in soft_masks.items():
                    if obj_id in tracked_objects:
                        save_path = os.path.join(self.soft_mask_dir, f'{obj_id}', f'{frame_idx:07d}.png')
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        binary_mask = (mask_array > 0.5).astype(np.uint8) * 255
                        cv2.imwrite(save_path, binary_mask)

            h, w = None, None
            if soft_masks:
                first_mask = next(iter(soft_masks.values()))
                h, w = first_mask.shape
            elif hasattr(self, 'height') and hasattr(self, 'width'):
                h, w = self.height, self.width
            elif not all_masks_only:
                for obj_id in range(1, self.num_objects + 1):
                    existing_mask_path = os.path.join(self.soft_mask_dir, f'{obj_id}', f'{frame_idx:07d}.png')
                    if os.path.exists(existing_mask_path):
                        existing_mask = cv2.imread(existing_mask_path, cv2.IMREAD_GRAYSCALE)
                        if existing_mask is not None:
                            h, w = existing_mask.shape
                            break
            else:
                npz_path = os.path.join(self.all_masks_dir, f'{frame_idx:07d}.npz')
                if os.path.exists(npz_path):
                    try:
                        prev = np.load(npz_path)['mask']
                        h, w = prev.shape[1], prev.shape[2]
                    except Exception:
                        pass

            if h is None or w is None:
                log.warning('Could not determine dimensions for frame %d, skipping all_masks', frame_idx)
                return

            existing_multi = None
            npz_path = os.path.join(self.all_masks_dir, f'{frame_idx:07d}.npz')
            if save_all_visible and not skip_disk_fallback and os.path.exists(npz_path):
                try:
                    existing_multi = np.load(npz_path)['mask']
                except Exception:
                    existing_multi = None

            multi_channel_mask = np.zeros((self.num_objects, h, w), dtype=np.uint8)

            for obj_id in range(1, self.num_objects + 1):
                channel_idx = obj_id - 1

                if obj_id in tracked_objects and obj_id in soft_masks:
                    mask_array = soft_masks[obj_id]
                    if mask_array.shape != (h, w):
                        mask_array = cv2.resize(mask_array, (w, h), interpolation=cv2.INTER_LINEAR)
                    multi_channel_mask[channel_idx] = (mask_array > 0.5).astype(np.uint8)
                elif save_all_visible and existing_multi is not None and channel_idx < existing_multi.shape[0]:
                    multi_channel_mask[channel_idx] = (existing_multi[channel_idx] > 0).astype(np.uint8)
                elif ((save_all_visible or obj_id in tracked_objects) and not skip_disk_fallback
                      and not all_masks_only):
                    existing_mask_path = os.path.join(self.soft_mask_dir, f'{obj_id}', f'{frame_idx:07d}.png')
                    if os.path.exists(existing_mask_path):
                        existing_mask = cv2.imread(existing_mask_path, cv2.IMREAD_GRAYSCALE)
                        if existing_mask is not None:
                            if existing_mask.shape != (h, w):
                                existing_mask = cv2.resize(existing_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                            multi_channel_mask[channel_idx] = (existing_mask > 127).astype(np.uint8)
            
            # Save multi-channel mask as compressed .npz file (much smaller than .npy)
            npz_path = os.path.join(self.all_masks_dir, f'{frame_idx:07d}.npz')
            np.savez_compressed(npz_path, mask=multi_channel_mask)
            
            # Count how many objects have non-zero masks
            non_empty_channels = np.sum([np.any(multi_channel_mask[i] > 0) for i in range(self.num_objects)])
            file_size = os.path.getsize(npz_path) / 1024  # Size in KB
            log.debug(
                'Saved all_masks for frame %d: %d/%d objects (%.1f KB)',
                frame_idx, non_empty_channels, self.num_objects, file_size)
            
            # Cache the multi-channel mask if enabled (store as uint8 to save memory)
            if self.enable_mask_cache and self.mask_cache is not None:
                # Store with implicit object IDs (channel i-1 = object ID i)
                self.mask_cache[frame_idx] = {
                    'mask': multi_channel_mask.copy(),  # Already uint8
                    'object_ids': list(range(1, self.num_objects + 1))  # Fixed mapping
                }
                
                # Clean up cache if too large
                if len(self.mask_cache) > self.cache_size_limit:
                    oldest_key = min(self.mask_cache.keys())
                    del self.mask_cache[oldest_key]
        except Exception as e:
            print(f"Error in batch soft mask saving: {str(e)}")
            # Fall back to individual saving if batch saving fails
            self._fallback_individual_saving(batch_data)

    def _fallback_individual_saving(self, batch_data: Dict):
        """Fallback: retry all_masks save without per-object PNGs."""
        try:
            fallback_data = dict(batch_data)
            fallback_data['all_masks_only'] = True
            frame_idx = batch_data.get('frame_idx')
            self._save_batch_soft_masks(fallback_data, self.names[frame_idx])
        except Exception as e:
            log.warning('Error in fallback all_masks saving: %s', e)

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
            log.debug('Save queue full (%d items), waiting for IO threads',
                      self.save_queue.qsize())
        self.save_queue.put(item, block=True)

    def wait_for_save_queue(self):
        """Block until all queued save tasks have been processed."""
        self.save_queue.join()

    def save_mask(self, ti: int, mask: np.ndarray, tracked_objects: set = None,
                  invalidate_cache: bool = True):
        """Save mask to masks folder for inference (ONLY tracked objects)
        
        IMPORTANT: This saves single-channel masks with ONLY tracked objects to prevent
        untracked objects from interfering with inference. Each pixel is the index of
        a tracked object or 0 (background/untracked).
        
        Args:
            ti: Frame index
            mask: Input mask (H*W) with object IDs
            tracked_objects: Set of object IDs that are being tracked (ONLY these are saved)
                           If None, extracts from mask
        """
        assert 0 <= ti < self.length
        assert isinstance(mask, np.ndarray)

        # Get dimensions from input mask
        h, w = mask.shape[0], mask.shape[1]
        
        # Determine tracked objects
        if tracked_objects is None:
            # Extract from input mask
            unique_ids = np.unique(mask)
            tracked_objects = set([int(id) for id in unique_ids if id > 0])
        
        if not tracked_objects:
            # No tracked objects, save empty mask
            inference_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            # Create inference mask with ONLY tracked objects
            # Untracked objects become 0 (background)
            inference_mask = np.zeros((h, w), dtype=np.uint8)
            for obj_id in tracked_objects:
                inference_mask[mask == obj_id] = obj_id
        
        # Convert to PIL Image in 'P' mode (single-channel with palette)
        mask_img = Image.fromarray(inference_mask, mode='P')
        
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
            
        if invalidate_cache:
            self.invalidate(ti)
        self.add_to_queue_with_warning(SaveItem('mask', mask_img, self.names[ti]))

    def _write_visualization_file(self, vis_mode: str, image: np.ndarray, name: str):
        os.makedirs(path.join(self.visualization_dir, vis_mode), exist_ok=True)
        if vis_mode == 'rgba':
            data = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA).copy()
            cv2.imwrite(path.join(self.visualization_dir, vis_mode, name + '.png'), data)
        else:
            data = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path.join(self.visualization_dir, vis_mode, name + '.jpg'), data)

    def save_mask_sync(self, ti: int, mask: np.ndarray, tracked_objects: set = None,
                       invalidate_cache: bool = True):
        """Save mask to masks folder synchronously (writes directly to disk).
        Same semantics as save_mask but does not use the queue. Use when masks must be on disk immediately.
        """
        assert 0 <= ti < self.length
        assert isinstance(mask, np.ndarray)
        h, w = mask.shape[0], mask.shape[1]
        if tracked_objects is None:
            unique_ids = np.unique(mask)
            tracked_objects = set([int(id) for id in unique_ids if id > 0])
        if not tracked_objects:
            inference_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            inference_mask = np.zeros((h, w), dtype=np.uint8)
            for obj_id in tracked_objects:
                inference_mask[mask == obj_id] = obj_id
        mask_img = Image.fromarray(inference_mask, mode='P')
        if isinstance(self.palette, bytes):
            palette_list = []
            for i in range(0, len(self.palette), 3):
                palette_list.extend([self.palette[i], self.palette[i+1], self.palette[i+2]])
            mask_img.putpalette(palette_list)
        else:
            mask_img.putpalette(self.palette)
        if invalidate_cache:
            self.invalidate(ti)
        mask_path = path.join(self.mask_dir, self.names[ti] + '.png')
        mask_img.save(mask_path)

    def save_visualization(self, ti: int, vis_mode: str, image: np.ndarray):
        assert 0 <= ti < self.length
        assert isinstance(image, np.ndarray)
        self.add_to_queue_with_warning(
            SaveItem(f'visualization_{vis_mode}', image, self.names[ti]))

    def save_visualization_sync(self, ti: int, vis_mode: str, image: np.ndarray):
        assert 0 <= ti < self.length
        assert isinstance(image, np.ndarray)
        self._write_visualization_file(vis_mode, image, self.names[ti])

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

    def _batch_soft_mask_data(self, ti: int, soft_masks: Dict[int, np.ndarray], tracked_objects: set,
                              save_all_visible: bool, skip_disk_fallback: bool,
                              all_masks_only: bool = True) -> Dict:
        return {
            'frame_idx': ti,
            'soft_masks': soft_masks,
            'tracked_objects': tracked_objects,
            'save_all_visible': save_all_visible,
            'skip_disk_fallback': skip_disk_fallback,
            'all_masks_only': all_masks_only,
        }

    def save_batch_soft_masks(self, ti: int, soft_masks: Dict[int, np.ndarray], tracked_objects: set,
                              save_all_visible: bool = True, skip_disk_fallback: bool = False,
                              all_masks_only: bool = True):
        """Queue all_masks save (and optional legacy per-object soft_mask PNGs)."""
        batch_data = self._batch_soft_mask_data(
            ti, soft_masks, tracked_objects, save_all_visible, skip_disk_fallback, all_masks_only)
        self.add_to_queue_with_warning(SaveItem('batch_soft_mask', batch_data, self.names[ti]))

    def save_batch_soft_masks_sync(self, ti: int, soft_masks: Dict[int, np.ndarray], tracked_objects: set,
                                   save_all_visible: bool = True, skip_disk_fallback: bool = False,
                                   all_masks_only: bool = True):
        """Save all_masks directly without using the async queue."""
        batch_data = self._batch_soft_mask_data(
            ti, soft_masks, tracked_objects, save_all_visible, skip_disk_fallback, all_masks_only)
        self._save_batch_soft_masks(batch_data, self.names[ti])

    def update_all_masks(self, ti: int):
        """Combine all available masks from soft_masks into fixed-size multi-channel format in all_masks
        
        Creates fixed-size mask (num_objects, H, W) where channel i-1 = object ID i.
        Uses uint8 binary format (0 or 1) to reduce file size by 4x compared to float32.
        No need for _ids.npy file since the mapping is fixed.
        """
        print(f"Updating all_masks for frame {ti}")
        
        # Create fixed-size multi-channel mask: (num_objects, H, W)
        # Channel i-1 always corresponds to object ID i
        # Use uint8 (0 or 1) instead of float32 to reduce file size by 4x
        multi_channel_mask = np.zeros((self.num_objects, self.height, self.width), dtype=np.uint8)
        
        # Load masks from all objects into their corresponding channels
        objects_with_masks = 0
        for obj_id in range(1, self.num_objects + 1):
            channel_idx = obj_id - 1
            mask_path = os.path.join(self.soft_mask_dir, f'{obj_id}', f'{ti:07d}.png')
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Resize if dimensions don't match
                    if mask.shape != (self.height, self.width):
                        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                    # Convert binary mask to uint8 (0 or 1)
                    multi_channel_mask[channel_idx] = (mask > 127).astype(np.uint8)
                    if np.any(multi_channel_mask[channel_idx] > 0):
                        objects_with_masks += 1
        
        # Save multi-channel mask as compressed .npz file (much smaller than .npy)
        # Use compression to achieve similar size to individual PNG soft masks
        npz_path = os.path.join(self.all_masks_dir, f'{ti:07d}.npz')
        np.savez_compressed(npz_path, mask=multi_channel_mask)
        
        file_size = os.path.getsize(npz_path) / 1024  # Size in KB
        print(f"Saved compressed multi-channel mask to {npz_path}: {objects_with_masks}/{self.num_objects} objects have masks (shape: {multi_channel_mask.shape}, dtype: {multi_channel_mask.dtype}, size: {file_size:.1f} KB)")
        
        # Update cache if enabled (store as uint8 to save memory)
        if self.enable_mask_cache and self.mask_cache is not None:
            self.mask_cache[ti] = {
                'mask': multi_channel_mask.copy(),  # Already uint8
                'object_ids': list(range(1, self.num_objects + 1))  # Fixed mapping
            }
            
            # Clean up cache if too large
            if len(self.mask_cache) > self.cache_size_limit:
                oldest_key = min(self.mask_cache.keys())
                del self.mask_cache[oldest_key]

    def get_all_masks(self, ti: int) -> Union[np.ndarray, Dict]:
        """Get the combined mask from all_masks directory or cache
        
        Returns fixed-size multi-channel mask (num_objects, H, W) where channel i-1 = object ID i.
        Uses uint8 binary format (0 or 1) for storage efficiency.
        Converts to float32 for compatibility with existing code.
        No need for _ids.npy file since the mapping is fixed.
        
        Returns:
            dict with 'mask' (num_objects*H*W float32) and 'object_ids' (list 1..num_objects) keys
            None if no mask exists
        """
        # Check cache first if enabled
        if self.enable_mask_cache and self.mask_cache is not None and ti in self.mask_cache:
            cached_result = self.mask_cache[ti].copy()
            # Convert uint8 to float32 for compatibility
            cached_result['mask'] = cached_result['mask'].astype(np.float32)
            return cached_result
            
        # Load fixed-size multi-channel format (.npz compressed or .npy for backward compatibility)
        npz_path = os.path.join(self.all_masks_dir, f'{ti:07d}.npz')
        npy_path = os.path.join(self.all_masks_dir, f'{ti:07d}.npy')
        
        # Try .npz first (new compressed format), then .npy (old format)
        if os.path.exists(npz_path):
            try:
                loaded_data = np.load(npz_path)
                multi_channel_mask = loaded_data['mask']
            except Exception as e:
                print(f"Error loading compressed mask for frame {ti}: {str(e)}")
                return None
        elif os.path.exists(npy_path):
            # Backward compatibility: load old .npy format
            try:
                multi_channel_mask = np.load(npy_path)
                
                # Verify it's uint8 format (new format)
                if multi_channel_mask.dtype != np.uint8:
                    raise ValueError(f"Mask file {npy_path} is in old float32 format. Please run the migration script to convert it to uint8 format.")
            except Exception as e:
                print(f"Error loading old format mask for frame {ti}: {str(e)}")
                return None
        else:
            return None
        
        # Common processing for both .npz and .npy formats
        try:
            # Verify fixed-size format
            if multi_channel_mask.shape[0] != self.num_objects:
                raise ValueError(f"Mask has {multi_channel_mask.shape[0]} channels, expected {self.num_objects}")
            
            # Ensure dimensions match current frame
            if multi_channel_mask.shape[1] != self.height or multi_channel_mask.shape[2] != self.width:
                print(f"Resizing mask for frame {ti} from {multi_channel_mask.shape[1:]} to ({self.height}, {self.width})")
                resized_channels = []
                for ch_idx in range(multi_channel_mask.shape[0]):
                    ch_resized = cv2.resize(multi_channel_mask[ch_idx], (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                    resized_channels.append(ch_resized)
                multi_channel_mask = np.stack(resized_channels, axis=0)
            
            # Convert uint8 to float32 for compatibility with existing code
            multi_channel_mask_float = multi_channel_mask.astype(np.float32)
            
            result = {
                'mask': multi_channel_mask_float,  # float32 for compatibility
                'object_ids': list(range(1, self.num_objects + 1))  # Fixed mapping: channel i-1 = object ID i
            }
            
            # Cache the result if enabled (store as uint8 to save memory)
            if self.enable_mask_cache and self.mask_cache is not None:
                self.mask_cache[ti] = {
                    'mask': multi_channel_mask,  # uint8 for memory efficiency
                    'object_ids': list(range(1, self.num_objects + 1))
                }
                
                # Clean up cache if too large
                if len(self.mask_cache) > self.cache_size_limit:
                    oldest_key = min(self.mask_cache.keys())
                    del self.mask_cache[oldest_key]
                    
            return result
        except Exception as e:
            print(f"Error processing multi-channel mask for frame {ti}: {str(e)}")
            return None

    def create_combined_mask_from_probabilities(self, ti: int, prob: np.ndarray, tracked_objects: set, save_all_visible: bool = True) -> Dict:
        """Create fixed-size multi-channel combined mask directly from probability tensor without I/O
        
        Returns fixed-size mask (num_objects, H, W) where channel i-1 = object ID i.
        
        Returns:
            dict with 'mask' (num_objects*H*W float32) and 'object_ids' (list 1..num_objects) keys
        """
        if prob is None:
            print(f"Warning: No probabilities provided for frame {ti}")
            return {
                'mask': np.zeros((self.num_objects, self.height, self.width), dtype=np.float32),
                'object_ids': list(range(1, self.num_objects + 1))
            }
            
        # Convert probability tensor to numpy
        prob_np = prob.cpu().numpy() if hasattr(prob, 'cpu') else prob
        print(f"Creating fixed-size multi-channel mask for frame {ti}, prob shape: {prob_np.shape}, tracked objects: {tracked_objects}")
        
        # Create fixed-size multi-channel mask: (num_objects, H, W)
        # Channel i-1 always corresponds to object ID i
        multi_channel_mask = np.zeros((self.num_objects, self.height, self.width), dtype=np.float32)
        
        # Fill channels with masks from tracked objects (current probabilities) - PRIORITY 1
        tracked_objects_added_from_prob = 0
        for obj_id in range(1, min(prob_np.shape[0], self.num_objects + 1)):
            if obj_id in tracked_objects:
                channel_idx = obj_id - 1
                obj_mask = (prob_np[obj_id] > 0.5)
                if np.any(obj_mask):
                    # Use current probability mask for tracked objects
                    multi_channel_mask[channel_idx] = prob_np[obj_id].astype(np.float32)
                    print(f"Added tracked object {obj_id} from current probabilities (channel {channel_idx}), pixels: {np.sum(obj_mask)}")
                    tracked_objects_added_from_prob += 1
        
        print(f"Added {tracked_objects_added_from_prob} tracked objects from current probabilities for frame {ti}")
        
        # Fill channels with existing soft masks for untracked objects if save_all_visible is enabled - PRIORITY 2
        untracked_objects_added = 0
        if save_all_visible:
            for obj_id in range(1, self.num_objects + 1):
                if obj_id not in tracked_objects:  # Only for untracked objects
                    channel_idx = obj_id - 1
                    # Only load if channel is still empty (tracked objects take priority)
                    if not np.any(multi_channel_mask[channel_idx] > 0.5):
                        existing_mask_path = os.path.join(self.soft_mask_dir, f'{obj_id}', f'{ti:07d}.png')
                        if os.path.exists(existing_mask_path):
                            existing_mask = cv2.imread(existing_mask_path, cv2.IMREAD_GRAYSCALE)
                            if existing_mask is not None:
                                # Resize if needed
                                if existing_mask.shape != (self.height, self.width):
                                    existing_mask = cv2.resize(existing_mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                                # Convert binary mask to probability (0.0 or 1.0)
                                binary_mask = (existing_mask > 127).astype(np.float32)
                                if np.any(binary_mask > 0.5):
                                    multi_channel_mask[channel_idx] = binary_mask
                                    print(f"Added existing soft mask for untracked object {obj_id} (channel {channel_idx}), pixels: {np.sum(binary_mask > 0.5)}")
                                    untracked_objects_added += 1
            
            print(f"Added {untracked_objects_added} untracked objects with existing masks for frame {ti}")
        
        total_non_zero = np.sum(multi_channel_mask > 0.5)
        print(f"Final fixed-size multi-channel mask for frame {ti}: shape {multi_channel_mask.shape}, total non-zero pixels: {total_non_zero}")
        
        result = {
            'mask': multi_channel_mask,
            'object_ids': list(range(1, self.num_objects + 1))  # Fixed mapping: channel i-1 = object ID i
        }
        
        # Cache the result if enabled
        if self.enable_mask_cache and self.mask_cache is not None:
            self.mask_cache[ti] = result
            
            # Clean up cache if too large
            if len(self.mask_cache) > self.cache_size_limit:
                oldest_key = min(self.mask_cache.keys())
                del self.mask_cache[oldest_key]
                
        return result

    def write_all_masks(self, ti: int, multi_channel_mask: np.ndarray):
        """Write fixed-size multi-channel mask to all_masks/{ti}.npz and update cache."""
        assert 0 <= ti < self.length
        if multi_channel_mask.shape[0] != self.num_objects:
            raise ValueError(
                f'Mask has {multi_channel_mask.shape[0]} channels, expected {self.num_objects}')
        h, w = multi_channel_mask.shape[1], multi_channel_mask.shape[2]
        if h != self.height or w != self.width:
            resized = []
            for ch_idx in range(multi_channel_mask.shape[0]):
                ch = multi_channel_mask[ch_idx]
                if ch.shape != (self.height, self.width):
                    ch = cv2.resize(ch, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                resized.append(ch)
            multi_channel_mask = np.stack(resized, axis=0)

        multi_uint8 = (multi_channel_mask > 0.5).astype(np.uint8)
        npz_path = os.path.join(self.all_masks_dir, f'{ti:07d}.npz')
        np.savez_compressed(npz_path, mask=multi_uint8)

        if self.enable_mask_cache and self.mask_cache is not None:
            self.mask_cache[ti] = {
                'mask': multi_uint8.copy(),
                'object_ids': list(range(1, self.num_objects + 1)),
            }
            if len(self.mask_cache) > self.cache_size_limit:
                oldest_key = min(self.mask_cache.keys())
                del self.mask_cache[oldest_key]

    def _get_image_unbuffered(self, ti: int):
        # returns H*W*3 uint8 array
        assert 0 <= ti < self.length

        image = Image.open(path.join(self.image_dir, self.names[ti] + '.jpg')).convert('RGB')
        image = np.array(image)
        return image

    def _get_mask_unbuffered(self, ti: int, tracked_objects: set = None):
        """Get mask from masks folder for inference (ONLY tracked objects)
        
        Returns single-channel mask (H*W) with ONLY tracked objects.
        Untracked objects are 0 (background).
        
        Args:
            ti: Frame index
            tracked_objects: Set of object IDs to filter. If None, returns all objects from file.
        
        Returns:
            H*W uint8 array with object IDs, or None if not found
        """
        assert 0 <= ti < self.length

        mask_path = path.join(self.mask_dir, self.names[ti] + '.png')
        if path.exists(mask_path):
            mask = Image.open(mask_path)
            mask = np.array(mask)
            
            # Filter to only tracked objects if specified
            if tracked_objects is not None:
                filtered_mask = np.zeros_like(mask)
                for obj_id in tracked_objects:
                    filtered_mask[mask == obj_id] = obj_id
                mask = filtered_mask
            
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
        # Note: get_mask is now a direct function call (not LRU cached), so no need to invalidate
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
