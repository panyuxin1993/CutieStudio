import numpy as np
import torch
import logging
from typing import List, Tuple, Union, Optional
import cv2
import os
from pathlib import Path
import re

class MaskTransformer:
    """
    Utility class for transforming between different mask formats:
    1. Single-channel mask with object IDs (idx_mask=True)
    2. Multi-channel binary masks (idx_mask=False)
    
    This class handles conversions between these formats and provides
    utilities for working with the DAVIS palette colors.
    """
    
    def __init__(self, davis_palette=None, num_objects=None):
        """
        Initialize the MaskTransformer.
        
        Args:
            davis_palette: Optional numpy array of RGB colors for DAVIS palette.
                          If None, will attempt to import from cutie.utils.palette.
            num_objects: Optional int specifying the number of objects to expect.
                        If provided, read_mask will be optimized to use this information
                        rather than analyzing each mask for object counts.
        """
        self.logger = logging.getLogger(__name__)
        self.num_objects = num_objects
        
        # Set up DAVIS palette
        if davis_palette is None:
            try:
                # Try to import from the cutie submodule with explicit path
                import sys
                import os
                cutie_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                sys.path.insert(0, cutie_dir)
                
                try:
                    from cutie.utils.palette import davis_palette_np
                    self.davis_palette = davis_palette_np
                except ImportError:
                    # Try alternative import path
                    from utils.palette import davis_palette_np
                    self.davis_palette = davis_palette_np
            except ImportError:
                self.logger.warning("Could not import DAVIS palette from cutie. Using default palette.")
                # Create a simple default palette with some distinct colors
                self.davis_palette = np.array([
                    [0, 0, 0],       # Background (black)
                    [255, 0, 0],     # Red
                    [0, 255, 0],     # Green
                    [0, 0, 255],     # Blue
                    [255, 255, 0],   # Yellow
                    [255, 0, 255],   # Magenta
                    [0, 255, 255],   # Cyan
                    [128, 0, 0],     # Maroon
                    [0, 128, 0],     # Dark Green
                    [0, 0, 128],     # Navy
                ], dtype=np.uint8)
        else:
            self.davis_palette = davis_palette
    
    def set_num_objects(self, num_objects: int):
        self.num_objects = num_objects

    def read_mask(self, 
                 mask_path: Union[str, Path], 
                 format: str = 'single_channel',
                 fast_read: bool = True) -> Tuple[np.ndarray, List[int], bool]:
        """
        Read a mask file and convert it to the desired format.
        
        Args:
            mask_path: Path to the mask file or directory (for 'separate_channel' format)
            format:  'single_channel', 'multi_channel', 'separate_channel', or 'separate_single_channel'
                     'separate_single_channel': Binary mask with object ID from parent folder name
            fast_read: If True, uses faster color matching by assuming objects use DAVIS palette colors in order.
                      This is much faster but may not work correctly if mask colors don't follow palette order.
            
        Returns:
            Tuple of (mask, object_ids, is_binary_mask)
            - mask: Mask in the target format
            - object_ids: List of object IDs found in the mask
            - is_binary_mask: True if mask is binary (only one object), False otherwise
        """
        mask_path = Path(mask_path)
        
        # Fast path: Use pre-determined number of objects when available
        if self.num_objects is not None and format not in ['separate_channel', 'separate_single_channel']:
            # Skip analysis of object numbers and directly use top N colors from palette
            return self._read_mask_with_known_objects(mask_path, format, fast_read)
        
        # Handle 'separate_single_channel' format - binary mask with object ID from parent folder
        if format == 'separate_single_channel':
            if not mask_path.is_file():
                raise ValueError(f"For 'separate_single_channel' format, mask_path must be a file: {mask_path}")
            
            # Get object ID from parent folder name
            parent_dir = mask_path.parent
            if not parent_dir.name.isdigit():
                raise ValueError(f"Parent directory name must be a digit for 'separate_single_channel' format: {parent_dir}")
            
            # Extract object ID from folder name
            object_id = int(parent_dir.name)
            self.logger.debug(f"Found object ID {object_id} from folder name {parent_dir.name}")
            
            # Set up ID mapping for consistency with other methods
            self._original_to_sequential_id_map = {object_id: 1}
            self._sequential_to_original_id_map = {1: object_id}
            
            # Read the binary mask
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise ValueError(f"Failed to read mask file: {mask_path}")
            
            # Create mask with proper object ID
            mask = np.zeros_like(mask_img, dtype=np.uint8)
            mask[mask_img > 127] = 1  # Use sequential ID 1 internally
            
            # Set return values
            object_ids = [1]  # Use sequential ID 1 internally
            is_binary_mask = True
            
            # Convert to target format if needed
            if format == 'multi_channel':
                mask = self.single_to_multi_channel(mask, object_ids)
            
            return mask, object_ids, is_binary_mask
            
        # Handle 'separate_channel' format - load from separate subfolders
        elif format == 'separate_channel':
            if not mask_path.is_file():
                raise ValueError(f"For 'separate_channel' format, mask_path must be a file: {mask_path}")
            
            # Get root path (parent of object folder)
            root_path = mask_path.parent.parent
            frame_name = mask_path.name
            
            # Find all object ID subfolders (named as numbers)
            object_dirs = sorted([d for d in root_path.iterdir() if d.is_dir() and d.name.isdigit()],
                               key=lambda x: int(x.name))
            
            if not object_dirs:
                raise ValueError(f"No object ID subfolders found in: {root_path}")
            
            # Get object IDs from folder names - use actual folder numbers as IDs
            object_ids = [int(d.name) for d in object_dirs]
            
            # Create mapping from folder name to sequential index
            id_mapping = {int(d.name): i+1 for i, d in enumerate(object_dirs)}
            
            # Store this mapping for use in save_masks
            self._original_to_sequential_id_map = id_mapping
            self._sequential_to_original_id_map = {v: k for k, v in id_mapping.items()}
            
            # Read first available mask to get dimensions
            sample_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if sample_mask is None:
                raise ValueError(f"Failed to read sample mask: {mask_path}")
            
            h, w = sample_mask.shape
            
            # Create multi-channel mask
            multi_mask = np.zeros((len(object_ids), h, w), dtype=np.float32)
            
            # Load mask from each object folder if available
            for idx, obj_dir in enumerate(object_dirs):
                mask_file = obj_dir / frame_name
                if mask_file.exists():
                    obj_mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                    if obj_mask is not None:
                        # Convert to binary (0 or 1)
                        multi_mask[idx] = (obj_mask > 127).astype(np.float32)
                else:
                    self.logger.debug(f"Mask file not found for object {obj_dir.name}, frame {frame_name}")
            
            is_binary_mask = len(object_ids) <= 1
            
            return multi_mask, object_ids, is_binary_mask
        
        # Regular file-based mask loading
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        # First try to read as grayscale to check if it's a grayscale image
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            raise ValueError(f"Failed to read mask file: {mask_path}")
        
        # Check if the file is actually grayscale by also trying to read as color
        mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        
        # If mask_bgr has the same dimensions as mask_gray plus a color channel,
        # then it's a color image. Otherwise, it's truly grayscale.
        if mask_bgr is not None and mask_bgr.shape[:2] == mask_gray.shape and mask_bgr.shape[2] == 3:
            # The image appears to be a color image - check if it's actually using colors
            # Convert BGR to RGB
            mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
            
            # Check if the image only contains grayscale values (R=G=B)
            is_true_grayscale = np.all(mask_rgb[:,:,0] == mask_rgb[:,:,1]) and np.all(mask_rgb[:,:,1] == mask_rgb[:,:,2])
            
            if is_true_grayscale:
                # It's a color image but actually grayscale - use the grayscale version
                self.logger.debug("Detected color image with grayscale values, treating as grayscale")
                mask = mask_gray
            else:
                # It's a real color image - convert RGB to object IDs using DAVIS palette
                h, w, _ = mask_rgb.shape
                mask = np.zeros((h, w), dtype=np.uint8)
                
                if fast_read:
                    # OPTIMIZED VERSION: Assume colors map to palette indices in order
                    # First, get unique colors in the mask
                    unique_colors = np.unique(mask_rgb.reshape(-1, 3), axis=0)
                    self.logger.debug(f"Found {len(unique_colors)} unique colors in mask")
                    
                    # Skip the first color (assumed to be background)
                    if len(unique_colors) > 1:
                        # Sort the unique colors by their occurrence in the mask (most frequent first)
                        color_counts = {}
                        for color in unique_colors:
                            color_match = np.all(mask_rgb == color.reshape(1, 1, 3), axis=2)
                            color_counts[tuple(color)] = np.sum(color_match)
                        
                        # Sort colors by frequency
                        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
                        
                        # Assume background is most common color
                        background_color = sorted_colors[0][0]
                        
                        # Create a lookup table mapping colors to object IDs
                        # Start with 1 to skip background (ID 0)
                        color_to_id = {}
                        next_id = 1
                        
                        # First, check if any colors match the DAVIS palette exactly
                        for color_idx, color in enumerate(self.davis_palette):
                            if color_idx == 0:  # Skip background
                                continue
                                
                            color_tuple = tuple(color)
                            if color_tuple in color_counts:
                                color_to_id[color_tuple] = color_idx
                        
                        # For any colors not in the palette, assign sequential IDs
                        for color, _ in sorted_colors:
                            if color == background_color:
                                color_to_id[color] = 0  # Background
                                continue
                                
                            if color not in color_to_id:
                                color_to_id[color] = next_id
                                next_id += 1
                        
                        # Apply the mapping to the mask in a single step
                        mask_flat = mask_rgb.reshape(-1, 3)
                        id_map = np.zeros(mask_flat.shape[0], dtype=np.uint8)
                        
                        for color, obj_id in color_to_id.items():
                            color_match = np.all(mask_flat == np.array(color), axis=1)
                            id_map[color_match] = obj_id
                        
                        mask = id_map.reshape(h, w)
                else:
                    # ORIGINAL VERSION: Find the closest DAVIS palette color for each pixel
                    for color_idx, color in enumerate(self.davis_palette):
                        # Compute color distance across the entire image in one vectorized operation
                        color_array = np.array(color, dtype=np.int32).reshape(1, 1, 3)
                        color_distances = np.sum((mask_rgb.astype(np.int32) - color_array)**2, axis=2)
                        
                        # Create mask where this color is the closest match (threshold for similarity)
                        color_match = (color_distances < 30)  # Threshold for color similarity
                        
                        # Use the actual color index as the object ID
                        if color_idx == 0:  # Background
                            mask[color_match] = 0
                        else:
                            # Assign actual palette index as object ID
                            mask[color_match] = color_idx
        else:
            # It's a true grayscale image - use the grayscale values directly as object IDs
            self.logger.debug("Detected grayscale image, using pixel values as object IDs")
            mask = mask_gray
        
        # Detect format and get object IDs
        _, object_ids, is_binary_mask = self.detect_mask_format(mask)
        
        
        return mask, object_ids, is_binary_mask
    
    def _read_mask_with_known_objects(self, 
                              mask_path: Union[str, Path], 
                              format: str = 'single_channel',
                              fast_read: bool = False) -> Tuple[np.ndarray, List[int], bool]:
        """
        Optimized mask reading when number of objects is known in advance.
        Uses the pre-specified number of objects (self.num_objects) to directly
        extract masks without analyzing the file for object counts.
        
        Args:
            mask_path: Path to the mask file
            format: Target format for the mask
            fast_read: If True, uses even faster processing
            
        Returns:
            Tuple of (mask, object_ids, is_binary_mask)
        """
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        # Read image
        mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        if mask_bgr is None:
            # Try grayscale if color read fails
            mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_gray is None:
                raise ValueError(f"Failed to read mask file: {mask_path}")
            
            # For grayscale images, use the values directly as object IDs
            mask = mask_gray
            # Get unique values excluding 0 (background)
            unique_ids = np.unique(mask)
            object_ids = [int(id) for id in unique_ids if id > 0]
            
            # Limit to the specified number of objects if needed
            if len(object_ids) > self.num_objects:
                object_ids = object_ids[:self.num_objects]
                
            # Create sequential IDs if needed
            if object_ids and max(object_ids) > len(object_ids) + 1:
                # Map original IDs to sequential IDs
                original_ids = sorted(object_ids)
                sequential_ids = list(range(1, len(original_ids) + 1))
                id_mapping = dict(zip(original_ids, sequential_ids))
                
                # Store mapping for later use
                self._original_to_sequential_id_map = id_mapping
                self._sequential_to_original_id_map = {v: k for k, v in id_mapping.items()}
                
                # Apply mapping to mask
                new_mask = np.zeros_like(mask)
                for orig_id, seq_id in id_mapping.items():
                    new_mask[mask == orig_id] = seq_id
                
                mask = new_mask
                object_ids = sequential_ids
            else:
                # No mapping needed
                self._original_to_sequential_id_map = {id: id for id in object_ids}
                self._sequential_to_original_id_map = {id: id for id in object_ids}
        else:
            # It's a color image - convert to RGB
            mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = mask_rgb.shape
            
            # Check if it's actually a grayscale image in RGB format
            is_true_grayscale = np.all(mask_rgb[:,:,0] == mask_rgb[:,:,1]) and np.all(mask_rgb[:,:,1] == mask_rgb[:,:,2])
            
            if is_true_grayscale:
                # It's a color image but actually grayscale
                mask_gray = mask_rgb[:,:,0]  # Just take one channel
                mask = mask_gray
                unique_ids = np.unique(mask)
                object_ids = [int(id) for id in unique_ids if id > 0]
                
                # Limit to specified number
                if len(object_ids) > self.num_objects:
                    object_ids = object_ids[:self.num_objects]
                
                # Setup ID mappings
                self._original_to_sequential_id_map = {id: id for id in object_ids}
                self._sequential_to_original_id_map = {id: id for id in object_ids}
            else:
                # It's a real color image - directly use top N palette colors
                mask = np.zeros((h, w), dtype=np.uint8)
                
                # Create a mapping of colors to object IDs using the first num_objects colors
                # from the DAVIS palette (excluding background at index 0)
                object_ids = list(range(1, self.num_objects + 1))
                
                # Create quick lookup for palette colors to search for
                palette_colors = [tuple(self.davis_palette[i]) for i in range(1, min(self.num_objects + 1, len(self.davis_palette)))]
                
                # For efficiency, reshape the mask for vectorized operations
                mask_flat = mask_rgb.reshape(-1, 3)
                id_map = np.zeros(mask_flat.shape[0], dtype=np.uint8)
                
                # For each color in palette (skipping background), assign the object ID
                for idx, color in enumerate(palette_colors):
                    obj_id = idx + 1  # Object IDs start at 1
                    color_array = np.array(color)
                    
                    if fast_read:
                        # Exact match for maximum speed
                        color_match = np.all(mask_flat == color_array, axis=1)
                    else:
                        # Approximate match with threshold
                        color_distances = np.sum((mask_flat.astype(np.int32) - color_array.astype(np.int32))**2, axis=1)
                        color_match = (color_distances < 30)  # Threshold for color similarity
                    
                    # Assign object ID where color matches
                    id_map[color_match] = obj_id
                
                # Reshape back to 2D
                mask = id_map.reshape(h, w)
                
                # Setup ID mappings - identity mapping since we're using palette indices directly
                self._original_to_sequential_id_map = {id: id for id in object_ids}
                self._sequential_to_original_id_map = {id: id for id in object_ids}
        
        # Handle conversion to requested format
        is_binary_mask = len(object_ids) <= 1
        
        if format == 'multi_channel':
            mask = self.single_to_multi_channel(mask, object_ids)
                
        return mask, object_ids, is_binary_mask
    
    def detect_mask_format(self, mask: np.ndarray) -> Tuple[str, List[int], bool]:
        """
        Detect the format of the input mask.
        
        Args:
            mask: Input mask as numpy array
            
        Returns:
            Tuple of (format_name, object_ids, is_binary_mask)
            - format_name: 'single_channel' or 'multi_channel'
            - object_ids: List of object IDs found in the mask
            - is_binary_mask: True if mask is binary (only one object), False otherwise
        """
        # Check if mask is multi-channel
        if len(mask.shape) == 3 and mask.shape[0] > 1:
            # Multi-channel mask (N x H x W)
            format_name = 'multi_channel'
            object_ids = list(range(1, mask.shape[0] + 1))  # Object IDs are 1-indexed
            is_binary_mask = len(object_ids) <= 1
        else:
            # Single-channel mask (H x W)
            if len(mask.shape) == 3:
                # If shape is (1, H, W), squeeze to (H, W)
                mask = mask.squeeze(0)
                
            format_name = 'single_channel'
            # Get unique object IDs (excluding background 0)
            unique_ids = np.unique(mask)
            object_ids = [int(id) for id in unique_ids if id > 0]
            
            # Map object IDs to sequential indices (1, 2, 3, ...) if they aren't already
            if object_ids and max(object_ids) > len(object_ids) + 1:
                # Create a mapping from original IDs to sequential indices
                original_ids = sorted(object_ids)
                sequential_ids = list(range(1, len(original_ids) + 1))
                id_mapping = dict(zip(original_ids, sequential_ids))
                
                # Store this mapping as an attribute for use in save_masks
                self._original_to_sequential_id_map = id_mapping
                self._sequential_to_original_id_map = {v: k for k, v in id_mapping.items()}
                
                # Apply the mapping to the mask
                new_mask = np.zeros_like(mask)
                for orig_id, seq_id in id_mapping.items():
                    new_mask[mask == orig_id] = seq_id
                    
                # Replace the mask and object IDs
                mask = new_mask
                object_ids = sequential_ids
            else:
                # No mapping needed, IDs are already sequential
                self._original_to_sequential_id_map = {id: id for id in object_ids}
                self._sequential_to_original_id_map = {id: id for id in object_ids}
                
            is_binary_mask = len(object_ids) <= 1
            
        self.logger.debug(f"Detected mask format: {format_name}, object IDs: {object_ids}, binary: {is_binary_mask}")
        return format_name, object_ids, is_binary_mask
    
    def single_to_multi_channel(self, 
                               mask: np.ndarray, 
                               object_ids: Optional[List[int]] = None) -> np.ndarray:
        """
        Convert a single-channel mask with object IDs to a multi-channel binary mask.
        
        Args:
            mask: Single-channel mask with object IDs (H x W)
            object_ids: Optional list of object IDs to include. If None, will use all IDs found in mask.
            
        Returns:
            Multi-channel binary mask (N x H x W) where N is the number of objects
        """
        # Ensure mask is 2D
        if len(mask.shape) == 3:
            mask = mask.squeeze(0)
            
        # Get object IDs if not provided
        if object_ids is None:
            unique_ids = np.unique(mask)
            object_ids = [int(id) for id in unique_ids if id > 0]
            
        # Sort object IDs for consistency
        object_ids = sorted(object_ids)
        
        # Create multi-channel mask
        h, w = mask.shape
        num_objects = len(object_ids)
        
        # Fast vectorized implementation - create all channels at once
        if num_objects > 0:
            # Create a single numpy array for all channels
            multi_mask = np.zeros((num_objects, h, w), dtype=np.float32)
            
            # Create binary masks for each object ID
            # The channel index does not need to match the object ID
            for i, obj_id in enumerate(object_ids):
                multi_mask[i] = (mask == obj_id).astype(np.float32)
                
            return multi_mask
        else:
            # Return empty mask if no objects
            return np.zeros((0, h, w), dtype=np.float32)
    
    def multi_to_single_channel(self, 
                               mask: np.ndarray, 
                               object_ids: Optional[List[int]] = None) -> np.ndarray:
        """
        Convert a multi-channel binary mask to a single-channel mask with object IDs.
        
        Args:
            mask: Multi-channel binary mask (N x H x W)
            object_ids: Optional list of object IDs to assign to each channel.
                       If None, will use sequential IDs starting from 1.
            
        Returns:
            Single-channel mask with object IDs (H x W)
        """
        # Get dimensions
        n_channels, h, w = mask.shape
        
        # Create object IDs if not provided
        if object_ids is None:
            # When object IDs are not provided, we need to use real DAVIS palette indices
            # not just sequential numbers from 1
            # This is important for maintaining compatibility with DAVIS palette colors
            object_ids = list(range(1, n_channels + 1))
        
        # Ensure we have the right number of object IDs
        if len(object_ids) != n_channels:
            raise ValueError(f"Number of object IDs ({len(object_ids)}) must match number of channels ({n_channels})")
            
        # Fast vectorized implementation
        # Initialize with zeros (background)
        single_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create a binary threshold version of the multi-channel mask
        binary_masks = (mask > 0.5)
        
        # Process in reverse order to handle overlapping regions (later channels take precedence)
        for idx in range(n_channels - 1, -1, -1):
            obj_id = object_ids[idx]
            # Set object ID in single-channel mask where this channel is active
            single_mask[binary_masks[idx]] = obj_id
            
        return single_mask
    
    def convert_mask(self, 
                    mask: Union[np.ndarray, torch.Tensor], 
                    target_format: str,
                    object_ids: Optional[List[int]] = None) -> Tuple[Union[np.ndarray, torch.Tensor], bool]:
        """
        Convert mask to the specified format.
        
        Args:
            mask: Input mask as numpy array or torch tensor
            target_format: 'single_channel' or 'multi_channel'
            object_ids: Optional list of object IDs
            
        Returns:
            Tuple of (converted_mask, idx_mask_value)
            - converted_mask: Mask in the target format
            - idx_mask_value: Value for idx_mask parameter (True for single_channel, False for multi_channel)
        """
        # Convert torch tensor to numpy if needed
        is_tensor = isinstance(mask, torch.Tensor)
        device = mask.device if is_tensor else None
        
        if is_tensor:
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask
            
        # Detect current format
        current_format, detected_object_ids, _ = self.detect_mask_format(mask_np)
        
        # Use detected object IDs if none provided
        if object_ids is None:
            object_ids = detected_object_ids
            
        # Convert if needed
        if current_format != target_format:
            if target_format == 'multi_channel':
                mask_np = self.single_to_multi_channel(mask_np, object_ids)
                idx_mask_value = False
            else:  # target_format == 'single_channel'
                mask_np = self.multi_to_single_channel(mask_np, object_ids)
                idx_mask_value = True
        else:
            # Format already matches
            idx_mask_value = (target_format == 'single_channel')
            
        # Convert back to torch tensor if input was tensor
        if is_tensor:
            mask_out = torch.from_numpy(mask_np).to(device)
        else:
            mask_out = mask_np
            
        return mask_out, idx_mask_value
    
    def visualize_mask(self, 
                      mask: np.ndarray, 
                      frame: Optional[np.ndarray] = None,
                      alpha: float = 0.5) -> np.ndarray:
        """
        Visualize mask using DAVIS palette colors.
        
        Args:
            mask: Mask as numpy array (can be single-channel or multi-channel)
            frame: Optional background frame to overlay mask on
            alpha: Transparency of the mask overlay (0-1)
            
        Returns:
            RGB visualization of the mask
        """
        # Detect mask format
        format_name, object_ids, _ = self.detect_mask_format(mask)
        
        # Convert to single-channel if needed
        if format_name == 'multi_channel':
            mask = self.multi_to_single_channel(mask, object_ids)
            
        # Create RGB visualization
        h, w = mask.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Apply colors from DAVIS palette
        for obj_id in object_ids:
            if obj_id < len(self.davis_palette):
                color = self.davis_palette[obj_id]
                vis[mask == obj_id] = color
                
        # Overlay on frame if provided
        if frame is not None:
            # Ensure frame has the right shape
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
                
            # Convert frame to RGB if it's BGR
            if frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
                
            # Create mask of non-zero regions
            mask_region = (mask > 0)
            
            # Blend visualization with frame
            vis_final = frame_rgb.copy()
            vis_final[mask_region] = cv2.addWeighted(
                frame_rgb[mask_region], 
                1 - alpha,
                vis[mask_region], 
                alpha, 
                0
            )
            return vis_final
        else:
            return vis 
    
    def save_masks(self,
                  mask: np.ndarray,
                  output_path: Union[str, Path],
                  frame_idx: int,
                  format_type: str = 'davis',
                  object_ids: Optional[List[int]] = None) -> None:
        """
        Save masks in the specified format.
        
        Args:
            mask: Input mask as numpy array (single-channel with object IDs or multi-channel)
            output_path: Path to save the masks (directory)
            frame_idx: Frame index for filename
            format_type: 'davis' for single-channel colored mask using DAVIS palette,
                        'binary' for separate binary masks in output_path,
                        'separate_channel' for separate binary masks in subfolders named by object ID
            object_ids: Optional list of object IDs. If None, will detect from mask.
            
        Returns:
            None
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Detect mask format and get object IDs if not provided
        mask_format, detected_obj_ids, is_binary_mask = self.detect_mask_format(mask)
        if object_ids is None:
            object_ids = detected_obj_ids
            
        if not object_ids:
            self.logger.warning(f"No objects found in mask for frame {frame_idx}")
            return

        # Ensure object_ids is properly sorted
        object_ids = sorted(object_ids)
        
        # Enhanced debug logging
        self.logger.debug(f"Saving mask with format {mask_format}, object IDs: {object_ids} for frame {frame_idx}")
        self.logger.debug(f"Mask shape: {mask.shape}, detected object IDs: {detected_obj_ids}")
        
        # Check if any objects are missing in the actual mask data
        missing_objects = []
        for obj_id in object_ids:
            if mask_format == 'single_channel':
                if not np.any(mask == obj_id):
                    missing_objects.append(obj_id)
            elif mask_format == 'multi_channel':
                # For multi-channel, check the corresponding channel
                obj_idx = object_ids.index(obj_id)
                if obj_idx < mask.shape[0] and not np.any(mask[obj_idx] > 0):
                    missing_objects.append(obj_id)
                
        if missing_objects:
            self.logger.debug(f"Warning: Objects {missing_objects} are specified but not present in the mask data")
        
        # Fast path for saving binary masks - avoid unnecessary conversions
        if format_type.lower() in ['separate_channel']:
            # Direct saving for binary format - create all masks in one pass
            
            # If mask is already single channel with object IDs, we can process directly
            if mask_format == 'single_channel':
                for obj_id in object_ids:
                    # Map sequential ID back to original ID for folder name if mapping exists
                    original_id = self._sequential_to_original_id_map.get(obj_id, obj_id)
                    
                    # Create subfolder for this object using original ID
                    obj_dir = output_path / str(original_id)
                    obj_dir.mkdir(exist_ok=True)
                    
                    # Extract binary mask directly without conversion
                    binary_mask = (mask == obj_id).astype(np.uint8) * 255
                    
                    # Save binary mask
                    output_file = obj_dir / f"{frame_idx:07d}.png"
                    cv2.imwrite(str(output_file), binary_mask)
                    
                self.logger.debug(f"Saved binary masks for {len(object_ids)} objects to {output_path}")
                return
            elif mask_format == 'multi_channel':
                # If mask is multi-channel, save each channel directly
                for idx, obj_id in enumerate(object_ids):
                    # Map sequential ID back to original ID for folder name if mapping exists
                    original_id = self._sequential_to_original_id_map.get(obj_id, obj_id)
                    
                    # Create subfolder for this object using original ID
                    obj_dir = output_path / str(original_id)
                    obj_dir.mkdir(exist_ok=True)
                    
                    # Get binary mask for this channel
                    binary_mask = (mask[idx] > 0.5).astype(np.uint8) * 255
                    
                    # Save binary mask
                    output_file = obj_dir / f"{frame_idx:07d}.png"
                    cv2.imwrite(str(output_file), binary_mask)
                    
                self.logger.debug(f"Saved binary masks for {len(object_ids)} objects to {output_path}")
                return
        
        # Original implementation for other cases
        # Ensure mask is in the right format for processing
        if mask_format == 'multi_channel' and format_type == 'davis':
            # Convert multi-channel to single-channel for DAVIS format with explicit object IDs
            mask = self.multi_to_single_channel(mask, object_ids)
            
        elif mask_format == 'single_channel' and format_type in ['separate_channel']:
            # Convert single-channel to multi-channel for binary format
            mask = self.single_to_multi_channel(mask, object_ids)
            
        # Save in the requested format
        if format_type.lower() == 'davis':
            # Create a colored mask using DAVIS palette
            h, w = mask.shape
            colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Log the object IDs we're about to save
            self.logger.debug(f"Saving DAVIS format mask with {len(object_ids)} objects: {object_ids}")
            
            # Create a copy of the mask to ensure we don't modify the original
            working_mask = mask.copy()
            
            # Fill with colors from DAVIS palette
            objects_saved = 0
            for obj_id in object_ids:
                if obj_id >= len(self.davis_palette):
                    self.logger.warning(f"Object ID {obj_id} exceeds DAVIS palette size ({len(self.davis_palette)})")
                    # Use a fallback color if the object ID is out of range
                    palette_idx = (obj_id % (len(self.davis_palette) - 1)) + 1
                else:
                    palette_idx = obj_id
                    
                # Get the color for this object
                color = self.davis_palette[palette_idx]
                
                # Find regions for this object
                obj_region = (working_mask == obj_id)
                
                # Check if the object exists in the mask
                if np.any(obj_region):
                    colored_mask[obj_region] = color
                    objects_saved += 1
                else:
                    self.logger.debug(f"Object {obj_id} not found in mask for frame {frame_idx}")
            
            # Log how many objects were actually saved
            self.logger.debug(f"Actually saved {objects_saved}/{len(object_ids)} objects to DAVIS format mask")
            
            # Save the colored mask
            output_file = output_path / f"{frame_idx:07d}.png"
            # Convert RGB to BGR for OpenCV
            colored_mask_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_file), colored_mask_bgr)
            
            self.logger.debug(f"Saved DAVIS format mask to {output_file}")
            
        elif format_type.lower() in [ 'separate_channel']:
            # Save separate binary masks in subfolders (fallback path)
            for idx, obj_id in enumerate(object_ids):
                # Map sequential ID back to original ID for folder name if mapping exists
                original_id = self._sequential_to_original_id_map.get(obj_id, obj_id)
                
                # Create subfolder for this object using original ID
                obj_dir = output_path / str(original_id)
                obj_dir.mkdir(exist_ok=True)
                
                # Get binary mask for this object
                if mask_format == 'multi_channel':
                    binary_mask = (mask[idx] > 0.5).astype(np.uint8) * 255
                else:
                    binary_mask = (mask == obj_id).astype(np.uint8) * 255
                
                # Save binary mask
                output_file = obj_dir / f"{frame_idx:07d}.png"
                cv2.imwrite(str(output_file), binary_mask)
                
            self.logger.debug(f"Saved binary masks for {len(object_ids)} objects to {output_path}")
        elif format_type.lower() in ['binary']:
            # Enhanced handling for binary mask format
            
            # Check dimensions and mask type to determine how to process
            if len(mask.shape) == 3 and mask.shape[0] <= 3:  # Multi-channel mask (CHW format)
                # Assume it's a multi-channel mask with one channel per object
                # For binary format, we combine all objects into a single binary mask
                combined_mask = np.zeros((mask.shape[1], mask.shape[2]), dtype=np.uint8)
                for idx, obj_id in enumerate(object_ids):
                    combined_mask = np.logical_or(combined_mask, mask[idx] > 0.5)
                
                # Scale to proper 8-bit range for saving
                save_mask = combined_mask.astype(np.uint8) * 255
                
            elif len(mask.shape) == 3 and mask.shape[2] <= 3:  # Color mask (HWC format)
                # Convert to grayscale for binary
                if mask.shape[2] == 3:  # RGB
                    gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                else:  # Single channel expanded
                    gray_mask = mask[:, :, 0]
                    
                # Threshold to binary
                save_mask = (gray_mask > 127).astype(np.uint8) * 255
                
            elif len(mask.shape) == 2:  # Single-channel mask
                # If mask has values of 0 and 1, scale to 0 and 255
                if np.max(mask) <= 1:
                    save_mask = mask.astype(np.uint8) * 255
                else:
                    # If mask already has values between 0-255, use as is
                    # but ensure any value > 0 is set to 255 (proper binary)
                    save_mask = (mask > 0).astype(np.uint8) * 255
            else:
                raise ValueError(f"Unsupported mask shape for binary format: {mask.shape}")
                
            # Save the binary mask
            output_file = output_path / f"{frame_idx:07d}.png"
            cv2.imwrite(str(output_file), save_mask)
            self.logger.debug(f"Saved binary mask to {output_file}")
            
        else:
            raise ValueError(f"Unsupported format_type: {format_type}. Use 'davis', 'binary', or 'separate_channel'.") 

    def mask_info(self,
                 mask_path: Union[str, Path],
                 fast_read: bool = True) -> Tuple[int, List[int]]:
        """
        Quickly extract information about a mask file without processing the actual masks.
        This is an optimized version of read_mask that only returns the number of objects
        and their IDs.
        
        Args:
            mask_path: Path to the mask file or directory
            fast_read: If True, uses faster color matching (recommended)
            
        Returns:
            Tuple of (num_objects, object_ids)
            - num_objects: Number of objects in the mask
            - object_ids: List of object IDs found in the mask
        """
        # Fast path for known number of objects
        if self.num_objects is not None:
            mask_path = Path(mask_path)
            
            # Handle directory case - just return pre-specified number
            if mask_path.is_dir():
                # For directories, use sequential IDs from 1 to num_objects
                object_ids = list(range(1, self.num_objects + 1))
                return self.num_objects, object_ids
                
            # Handle separate_single_channel case
            parent_dir = mask_path.parent
            if parent_dir.name.isdigit() and mask_path.is_file():
                object_id = int(parent_dir.name)
                return 1, [object_id]
                
            # For regular files, just return pre-specified information
            # We're using palette colors in order, so object IDs are 1 to num_objects
            object_ids = list(range(1, self.num_objects + 1))
            return self.num_objects, object_ids
                
        # Original implementation for when num_objects is not specified
        mask_path = Path(mask_path)
        
        # Handle separate_channel directory format
        if mask_path.is_dir():
            # Find all object ID subfolders (named as numbers)
            object_dirs = sorted([d for d in mask_path.iterdir() if d.is_dir() and d.name.isdigit()],
                               key=lambda x: int(x.name))
            
            if not object_dirs:
                self.logger.warning(f"No object ID subfolders found in: {mask_path}")
                return 0, []
            
            # Get object IDs from folder names
            object_ids = [int(d.name) for d in object_dirs]
            return len(object_ids), object_ids
        
        # Handle separate_single_channel format (mask file in an object ID folder)
        parent_dir = mask_path.parent
        if parent_dir.name.isdigit() and mask_path.is_file():
            object_id = int(parent_dir.name)
            return 1, [object_id]
        
        # Handle regular file formats
        if not mask_path.exists():
            self.logger.warning(f"Mask file not found: {mask_path}")
            return 0, []

        # First check if it's a grayscale or color image
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            self.logger.warning(f"Failed to read mask file: {mask_path}")
            return 0, []
            
        # Check if grayscale (binary/indexed) mask or color mask
        mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        if mask_bgr is None or (mask_bgr.shape[:2] == mask_gray.shape and 
                               mask_bgr.shape[2] == 3 and 
                               np.all(mask_bgr[:,:,0] == mask_bgr[:,:,1]) and 
                               np.all(mask_bgr[:,:,1] == mask_bgr[:,:,2])):
            # It's a grayscale mask - count unique non-zero values
            unique_values = np.unique(mask_gray)
            # Filter out zero (background)
            object_ids = [int(v) for v in unique_values if v > 0]
            return len(object_ids), object_ids
        else:
            # It's a color mask - convert RGB to object IDs using palette
            mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
            
            # Get unique colors in the mask
            unique_colors = np.unique(mask_rgb.reshape(-1, 3), axis=0)
            
            # Skip the first color (assumed to be background) if there are multiple colors
            if len(unique_colors) <= 1:
                return 0, []
                
            # Sort the unique colors by their occurrence in the mask (most frequent first)
            color_counts = {}
            for color in unique_colors:
                color_match = np.all(mask_rgb == color.reshape(1, 1, 3), axis=2)
                color_counts[tuple(color)] = np.sum(color_match)
            
            # Sort colors by frequency
            sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Assume background is most common color
            background_color = sorted_colors[0][0]
            
            # Start with object ID 1
            next_id = 1
            object_ids = []
            
            # Check against the DAVIS palette if we want to preserve exact IDs
            if fast_read:
                # Create a mapping of colors to object IDs
                color_to_id = {}
                
                # First, check if any colors match the DAVIS palette exactly
                for color_idx, color in enumerate(self.davis_palette):
                    if color_idx == 0:  # Skip background
                        continue
                        
                    color_tuple = tuple(color)
                    if color_tuple in color_counts:
                        color_to_id[color_tuple] = color_idx
                        object_ids.append(color_idx)
                
                # For any colors not in the palette, assign sequential IDs
                for color, _ in sorted_colors:
                    if color == background_color:
                        continue
                        
                    if color not in color_to_id:
                        color_to_id[color] = next_id
                        object_ids.append(next_id)
                        next_id += 1
            else:
                # Simpler approach: count non-background colors
                object_ids = list(range(1, len(sorted_colors)))
            
            return len(object_ids), sorted(object_ids) 