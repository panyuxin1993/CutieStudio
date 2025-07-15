import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from cutie.utils.palette import davis_palette_np
from typing import Dict, List, Tuple
import concurrent.futures
from functools import lru_cache
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import mmap
import time

def get_davis_color_mapping():
    """
    Returns the standard DAVIS dataset color mapping.
    Colors are in RGB format, matching the palette used in cutie/utils/palette.py.
    """
    # Convert numpy array to list of tuples for easier comparison
    davis_colors = [tuple(color) for color in davis_palette_np]
    return davis_colors

def calculate_mask_areas(mask_dir, num_objects=None, object_names=None):
    """
    Calculate areas for each object mask in DAVIS dataset frames.
    
    Args:
        mask_dir (str or Path): Directory containing mask PNG files
        num_objects (int, optional): Number of objects to track
        object_names (list, optional): List of object names
        
    Returns:
        pd.DataFrame: DataFrame with frame indices and object areas
    """
    mask_dir = Path(mask_dir)
    mask_files = sorted(mask_dir.glob('*.png'))  # Get all PNG files
    
    if not mask_files:
        raise ValueError(f"No PNG files found in {mask_dir}")
    
    # Get standard DAVIS colors
    davis_colors = get_davis_color_mapping()
    
    # Initialize results dictionary with predefined columns
    results = {
        'frame': []  # Frame index column
    }
    
    # Add columns for each object using provided names or default names
    if object_names:
        for i, name in enumerate(object_names, 1):
            results[name] = []
    else:
        for i in range(1, (num_objects or 1) + 1):
            results[f'object_{i}'] = []
    
    # Create color to object ID mapping using DAVIS palette
    color_to_id = {}
    for i in range(1, (num_objects or 1) + 1):
        color_to_id[tuple(davis_colors[i])] = i
    
    print("\nObject color mapping (DAVIS standard order):")
    print("Background: RGB(0, 0, 0)")
    for i, (color, obj_id) in enumerate(color_to_id.items(), 1):
        name = object_names[i-1] if object_names else f'object_{i}'
        print(f"{name}: RGB{color}")
    print()
    
    # Process each mask
    print("Processing masks...")
    for mask_file in tqdm(mask_files):
        # Get frame index from filename
        frame_idx = int(mask_file.stem)
        
        # Read mask and convert to RGB
        mask = cv2.imread(str(mask_file))
        if mask is None:
            raise ValueError(f"Could not read mask file: {mask_file}")
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        # Add frame index
        results['frame'].append(frame_idx)
        
        # Calculate area for each predefined object
        if object_names:
            for i, name in enumerate(object_names, 1):
                color = tuple(davis_colors[i])
                color_mask = np.all(mask_rgb == color, axis=2)
                area = np.sum(color_mask)
                results[name].append(area)
        else:
            for i in range(1, (num_objects or 1) + 1):
                color = tuple(davis_colors[i])
                color_mask = np.all(mask_rgb == color, axis=2)
                area = np.sum(color_mask)
                results[f'object_{i}'].append(area)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df

def calculate_mask_metrics(mask_path: str, num_objects: int = None, object_names: List[str] = None) -> pd.DataFrame:
    """
    Calculate comprehensive metrics for each mask in a frame.
    
    Args:
        mask_path: Path to the mask file
        num_objects: Number of objects to expect
        object_names: Optional list of object names
        
    Returns:
        DataFrame containing metrics for each object in the mask
    """
    # Get frame index from filename
    frame_idx = int(Path(mask_path).stem)
    print(f"Processing frame {frame_idx}")
    
    # Initialize list to store metrics
    metrics_list = []
    
    # Get soft mask directory
    soft_mask_dir = Path(mask_path).parent.parent / 'soft_masks'
    if not soft_mask_dir.exists():
        print(f"ERROR: Soft masks directory not found at {soft_mask_dir}")
        return pd.DataFrame()
    
    # Process each object
    for obj_id in range(1, num_objects + 1 if num_objects else 100):  # Use large default to find all objects
        # Get soft mask path for this object
        obj_mask_path = soft_mask_dir / str(obj_id) / f"{frame_idx:07d}.png"
        
        if not obj_mask_path.exists():
            print(f"Soft mask not found for object {obj_id} in frame {frame_idx}")
            continue
            
        # Read binary mask
        mask = cv2.imread(str(obj_mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to read mask for object {obj_id} in frame {frame_idx}")
            continue
            
        # Convert to binary
        binary_mask = (mask > 127).astype(np.uint8)
        
        # Calculate basic metrics
        area = np.sum(binary_mask)
        if area == 0:
            continue
            
        # Calculate contour
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
            
        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Calculate perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate center of mass using moments
        M = cv2.moments(contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            # If moments calculation fails, use bounding box center
            center_x = x + w // 2
            center_y = y + h // 2
        
        # Calculate orientation using PCA
        # Get coordinates of non-zero points
        y_coords, x_coords = np.where(binary_mask > 0)
        coords = np.column_stack((x_coords, y_coords))
        
        # Center the coordinates
        mean = np.mean(coords, axis=0)
        centered_coords = coords - mean
        
        # Calculate covariance matrix
        cov = np.cov(centered_coords.T)
        
        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Calculate orientation angle (in degrees)
        orientation = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        
        # Calculate circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Create metrics dictionary
        metrics = {
            'frame': frame_idx,
            'object_id': obj_id,
            'object_name': object_names[obj_id-1] if object_names and obj_id <= len(object_names) else f'Object_{obj_id}',
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'orientation': orientation,
            'bbox_x': x,
            'bbox_y': y,
            'bbox_width': w,
            'bbox_height': h,
            'center_x': center_x,
            'center_y': center_y
        }
        
        metrics_list.append(metrics)
    
    # Convert to DataFrame
    if metrics_list:
        df = pd.DataFrame(metrics_list)
        return df
    else:
        return pd.DataFrame()

def calculate_mask_metrics_batch(mask_folder: str, num_objects: int = None, object_names: List[str] = None) -> pd.DataFrame:
    """
    Calculate metrics for all masks in a folder using soft masks.
    
    Args:
        mask_folder: Path to folder containing mask files
        num_objects: Number of objects to expect
        object_names: Optional list of object names
        
    Returns:
        DataFrame containing metrics for all frames and objects
    """
    mask_folder = Path(mask_folder)
    if not mask_folder.exists():
        print(f"ERROR: Mask folder not found: {mask_folder}")
        return pd.DataFrame()
        
    # Get soft mask directory
    soft_mask_dir = mask_folder.parent / 'soft_masks'
    if not soft_mask_dir.exists():
        print(f"ERROR: Soft masks directory not found at {soft_mask_dir}")
        return pd.DataFrame()
        
    # Get all frame indices from any object's soft masks
    frame_indices = set()
    for obj_dir in soft_mask_dir.iterdir():
        if obj_dir.is_dir():
            frame_indices.update([int(f.stem) for f in obj_dir.glob('*.png')])
            
    if not frame_indices:
        print("ERROR: No soft mask files found")
        return pd.DataFrame()
        
    print(f"Found {len(frame_indices)} frames with soft masks")
    
    # Process each frame
    all_metrics = []
    for frame_idx in sorted(frame_indices):
        # Create a dummy mask path to pass to calculate_mask_metrics
        dummy_mask_path = mask_folder / f"{frame_idx:07d}.png"
        frame_metrics = calculate_mask_metrics(str(dummy_mask_path), num_objects, object_names)
        if not frame_metrics.empty:
            all_metrics.append(frame_metrics)
            
    if not all_metrics:
        print("WARNING: No metrics were calculated for any frames")
        return pd.DataFrame()
        
    # Combine all frame metrics
    return pd.concat(all_metrics, ignore_index=True)

def calculate_pairwise_metrics(mask1: np.ndarray, mask2: np.ndarray) -> dict:
    """
    Calculate pairwise metrics between two binary masks.
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        Dictionary containing:
        - distance: Euclidean distance between centroids
        - overlap_ratio: IoU-like metric (intersection area / union area)
        - contact_length: Length of shared border
    """
    # Calculate centroids
    M1 = cv2.moments(mask1)
    M2 = cv2.moments(mask2)
    
    if M1["m00"] == 0 or M2["m00"] == 0:
        return {
            'distance': float('inf'),
            'overlap_ratio': 0.0,
            'contact_length': 0.0
        }
    
    # Get centroids
    c1_x = M1["m10"] / M1["m00"]
    c1_y = M1["m01"] / M1["m00"]
    c2_x = M2["m10"] / M2["m00"]
    c2_y = M2["m01"] / M2["m00"]
    
    # Calculate Euclidean distance
    distance = np.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)
    
    # Calculate overlap ratio
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    overlap_ratio = intersection / union if union > 0 else 0.0
    
    # Calculate contact length
    # Dilate both masks
    kernel = np.ones((3,3), np.uint8)
    dilated1 = cv2.dilate(mask1, kernel, iterations=1)
    dilated2 = cv2.dilate(mask2, kernel, iterations=1)
    
    # Find shared border
    shared_border = np.logical_and(dilated1, dilated2)
    contact_length = np.sum(shared_border)
    
    return {
        'distance': float(distance),
        'overlap_ratio': float(overlap_ratio),
        'contact_length': float(contact_length)
    }

def calculate_frame_pairwise_metrics(mask_dir: str, frame_idx: int, num_objects: int) -> np.ndarray:
    """
    Calculate pairwise metrics for all object pairs in a frame.
    
    Args:
        mask_dir: Directory containing soft masks
        frame_idx: Frame index
        num_objects: Number of objects
        
    Returns:
        Structured numpy array containing pairwise metrics
    """
    # Define dtype for structured array
    dtype = [
        ('obj1', 'i4'),
        ('obj2', 'i4'),
        ('distance', 'f4'),
        ('overlap_ratio', 'f4'),
        ('contact_length', 'f4')
    ]
    
    # Initialize array for all pairs
    n_pairs = (num_objects * (num_objects - 1)) // 2
    metrics = np.zeros(n_pairs, dtype=dtype)
    
    # Get soft mask directory
    soft_mask_dir = Path(mask_dir).parent / 'soft_masks'
    if not soft_mask_dir.exists():
        return metrics
    
    # Calculate metrics for each pair
    pair_idx = 0
    for i in range(1, num_objects + 1):
        for j in range(i + 1, num_objects + 1):
            # Read masks
            mask1_path = soft_mask_dir / str(i) / f"{frame_idx:07d}.png"
            mask2_path = soft_mask_dir / str(j) / f"{frame_idx:07d}.png"
            
            if not (mask1_path.exists() and mask2_path.exists()):
                continue
                
            mask1 = cv2.imread(str(mask1_path), cv2.IMREAD_GRAYSCALE)
            mask2 = cv2.imread(str(mask2_path), cv2.IMREAD_GRAYSCALE)
            
            if mask1 is None or mask2 is None:
                continue
                
            # Convert to binary
            mask1 = (mask1 > 127).astype(np.uint8)
            mask2 = (mask2 > 127).astype(np.uint8)
            
            # Calculate metrics
            pair_metrics = calculate_pairwise_metrics(mask1, mask2)
            
            # Store in array
            metrics[pair_idx] = (i, j, 
                               pair_metrics['distance'],
                               pair_metrics['overlap_ratio'],
                               pair_metrics['contact_length'])
            pair_idx += 1
    
    return metrics

def calculate_all_pairwise_metrics(mask_dir: str, num_objects: int) -> dict:
    """
    Calculate pairwise metrics for all frames and object pairs.
    
    Args:
        mask_dir: Directory containing masks
        num_objects: Number of objects
        
    Returns:
        Dictionary containing:
        - metrics: Dictionary mapping frame indices to metrics arrays
        - frame_indices: List of frame indices
    """
    # Get all frame indices
    soft_mask_dir = Path(mask_dir).parent / 'soft_masks'
    if not soft_mask_dir.exists():
        return {'metrics': {}, 'frame_indices': []}
        
    # Get frames from first object directory
    obj_dir = soft_mask_dir / '1'
    if not obj_dir.exists():
        return {'metrics': {}, 'frame_indices': []}
        
    frame_indices = [int(f.stem) for f in obj_dir.glob('*.png')]
    frame_indices.sort()
    
    # Calculate metrics for each frame
    metrics = {}
    for frame_idx in frame_indices:
        frame_metrics = calculate_frame_pairwise_metrics(mask_dir, frame_idx, num_objects)
        metrics[frame_idx] = frame_metrics
    
    return {
        'metrics': metrics,
        'frame_indices': frame_indices
    }

def save_pairwise_metrics(metrics_dict: dict, output_path: str):
    """
    Save pairwise metrics to a .npz file.
    
    Args:
        metrics_dict: Dictionary containing metrics and frame indices
        output_path: Path to save the metrics
    """
    np.savez(output_path,
             metrics=metrics_dict['metrics'],
             frame_indices=metrics_dict['frame_indices'])

def load_pairwise_metrics(input_path: str) -> dict:
    """
    Load pairwise metrics from a .npz file.
    
    Args:
        input_path: Path to the metrics file
        
    Returns:
        Dictionary containing metrics and frame indices
    """
    data = np.load(input_path, allow_pickle=True)
    return {
        'metrics': data['metrics'].item(),
        'frame_indices': data['frame_indices']
    }

def main():
    """Main function to run the mask area calculation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Calculate object areas from DAVIS dataset masks.'
    )
    parser.add_argument('mask_dir', type=str, help='Directory containing mask PNG files')
    parser.add_argument('--output', type=str, default=None, 
                      help='Output CSV file path (default: ../mask_area.csv relative to mask_dir)')
    parser.add_argument('--num_objects', type=int, default=None,
                      help='Number of objects to track')
    parser.add_argument('--object_names', nargs='+', type=str, default=None,
                      help='Names of the objects to track')
    
    args = parser.parse_args()
    
    try:
        # Set default output path if not specified
        if args.output is None:
            mask_dir = Path(args.mask_dir)
            output_path = mask_dir.parent / 'mask_area.csv'
        else:
            output_path = Path(args.output)
            
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Calculate areas
        df = calculate_mask_areas(args.mask_dir, args.num_objects, args.object_names)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
        
    return 0

if __name__ == '__main__':
    exit(main()) 

def calculate_all_pairwise_metrics_optimized(mask_dir: str, num_objects: int, max_workers: int = 16) -> dict:
    """
    Optimized version of calculate_all_pairwise_metrics that uses batch loading and parallel processing.
    
    Args:
        mask_dir: Directory containing masks
        num_objects: Number of objects
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary containing:
        - metrics: Dictionary mapping frame indices to metrics arrays
        - frame_indices: List of frame indices
    """
    # Get all frame indices
    soft_mask_dir = Path(mask_dir).parent / 'soft_masks'
    if not soft_mask_dir.exists():
        return {'metrics': {}, 'frame_indices': []}
        
    # Get frames from first object directory
    obj_dir = soft_mask_dir / '1'
    if not obj_dir.exists():
        return {'metrics': {}, 'frame_indices': []}
        
    frame_indices = [int(f.stem) for f in obj_dir.glob('*.png')]
    frame_indices.sort()
    
    print(f"Processing {len(frame_indices)} frames with {num_objects} objects using {max_workers} workers...")
    
    # Use parallel processing for frame calculations
    metrics = {}
    
    def process_frame(frame_idx):
        """Process a single frame - optimized for parallel execution"""
        return frame_idx, calculate_frame_pairwise_metrics_optimized(mask_dir, frame_idx, num_objects)
    
    # Use ThreadPoolExecutor for I/O-bound operations
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all frame processing tasks
        future_to_frame = {executor.submit(process_frame, frame_idx): frame_idx for frame_idx in frame_indices}
        
        # Collect results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_frame), 
                          total=len(frame_indices), 
                          desc="Processing frames"):
            frame_idx, frame_metrics = future.result()
            metrics[frame_idx] = frame_metrics
    
    return {
        'metrics': metrics,
        'frame_indices': frame_indices
    }

def calculate_frame_pairwise_metrics_optimized(mask_dir: str, frame_idx: int, num_objects: int) -> np.ndarray:
    """
    Optimized version that loads all object masks for a frame at once.
    
    Args:
        mask_dir: Directory containing soft masks
        frame_idx: Frame index
        num_objects: Number of objects
        
    Returns:
        Structured numpy array containing pairwise metrics
    """
    # Define dtype for structured array
    dtype = [
        ('obj1', 'i4'),
        ('obj2', 'i4'),
        ('distance', 'f4'),
        ('overlap_ratio', 'f4'),
        ('contact_length', 'f4')
    ]
    
    # Initialize array for all pairs
    n_pairs = (num_objects * (num_objects - 1)) // 2
    metrics = np.zeros(n_pairs, dtype=dtype)
    
    # Get soft mask directory
    soft_mask_dir = Path(mask_dir).parent / 'soft_masks'
    if not soft_mask_dir.exists():
        return metrics
    
    # Load all object masks for this frame at once
    object_masks = {}
    for obj_id in range(1, num_objects + 1):
        mask_path = soft_mask_dir / str(obj_id) / f"{frame_idx:07d}.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                object_masks[obj_id] = (mask > 127).astype(np.uint8)
    
    if len(object_masks) < 2:
        return metrics
    
    # Calculate metrics for each pair using vectorized operations
    pair_idx = 0
    for i in range(1, num_objects + 1):
        for j in range(i + 1, num_objects + 1):
            if i in object_masks and j in object_masks:
                mask1 = object_masks[i]
                mask2 = object_masks[j]
                
                # Calculate metrics using optimized function
                pair_metrics = calculate_pairwise_metrics_optimized(mask1, mask2)
                
                # Store in array
                metrics[pair_idx] = (i, j, 
                                   pair_metrics['distance'],
                                   pair_metrics['overlap_ratio'],
                                   pair_metrics['contact_length'])
            pair_idx += 1
    
    return metrics

def calculate_pairwise_metrics_optimized(mask1: np.ndarray, mask2: np.ndarray) -> dict:
    """
    Optimized version of calculate_pairwise_metrics that uses vectorized operations.
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        Dictionary containing metrics
    """
    # Check if masks are empty
    if mask1.sum() == 0 or mask2.sum() == 0:
        return {
            'distance': float('inf'),
            'overlap_ratio': 0.0,
            'contact_length': 0.0
        }
    
    # Calculate centroids using vectorized operations
    y1, x1 = np.where(mask1 > 0)
    y2, x2 = np.where(mask2 > 0)
    
    if len(x1) == 0 or len(x2) == 0:
        return {
            'distance': float('inf'),
            'overlap_ratio': 0.0,
            'contact_length': 0.0
        }
    
    # Calculate centroids
    c1_x, c1_y = np.mean(x1), np.mean(y1)
    c2_x, c2_y = np.mean(x2), np.mean(y2)
    
    # Calculate Euclidean distance
    distance = np.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)
    
    # Calculate overlap ratio using vectorized operations
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    overlap_ratio = intersection / union if union > 0 else 0.0
    
    # Calculate contact length using optimized dilation
    # Use a smaller kernel for better performance
    kernel = np.ones((3, 3), np.uint8)
    dilated1 = cv2.dilate(mask1, kernel, iterations=1)
    dilated2 = cv2.dilate(mask2, kernel, iterations=1)
    
    # Find shared border
    shared_border = np.logical_and(dilated1, dilated2)
    contact_length = np.sum(shared_border)
    
    return {
        'distance': float(distance),
        'overlap_ratio': float(overlap_ratio),
        'contact_length': float(contact_length)
    }

@lru_cache(maxsize=128)
def get_frame_indices_cached(soft_mask_dir: str, obj_id: int) -> Tuple[int, ...]:
    """
    Cached function to get frame indices for an object.
    
    Args:
        soft_mask_dir: Path to soft masks directory
        obj_id: Object ID
        
    Returns:
        Tuple of frame indices
    """
    obj_dir = Path(soft_mask_dir) / str(obj_id)
    if not obj_dir.exists():
        return ()
    return tuple(int(f.stem) for f in obj_dir.glob('*.png'))

def calculate_all_pairwise_metrics_batch_optimized(mask_dir: str, num_objects: int, batch_size: int = 10) -> dict:
    """
    Ultra-optimized version that processes frames in batches to minimize I/O overhead.
    
    Args:
        mask_dir: Directory containing masks
        num_objects: Number of objects
        batch_size: Number of frames to process in each batch
        
    Returns:
        Dictionary containing metrics and frame indices
    """
    # Get all frame indices
    soft_mask_dir = Path(mask_dir).parent / 'soft_masks'
    if not soft_mask_dir.exists():
        return {'metrics': {}, 'frame_indices': []}
        
    # Get frames from first object directory
    obj_dir = soft_mask_dir / '1'
    if not obj_dir.exists():
        return {'metrics': {}, 'frame_indices': []}
        
    frame_indices = [int(f.stem) for f in obj_dir.glob('*.png')]
    frame_indices.sort()
    
    print(f"Processing {len(frame_indices)} frames in batches of {batch_size}...")
    print(f"Total batches: {(len(frame_indices) + batch_size - 1) // batch_size}")
    
    metrics = {}
    
    # Process frames in batches
    for batch_start in tqdm(range(0, len(frame_indices), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(frame_indices))
        batch_frames = frame_indices[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}: frames {batch_start}-{batch_end-1}")
        
        # Load all masks for this batch at once
        batch_masks = load_batch_masks(soft_mask_dir, batch_frames, num_objects)
        
        # Process each frame in the batch
        for frame_idx in batch_frames:
            if frame_idx in batch_masks:
                frame_metrics = calculate_frame_metrics_from_batch(batch_masks[frame_idx], num_objects)
                metrics[frame_idx] = frame_metrics
        
        # Update performance monitoring
        try:
            from utils.performance_monitor import update_global_frame_count
            update_global_frame_count(len(metrics))
        except ImportError:
            pass
        
        print(f"Completed batch {batch_start//batch_size + 1}: processed {len([f for f in batch_frames if f in batch_masks])} frames")
        
        # Clear batch_masks to free memory after each batch
        del batch_masks
    
    return {
        'metrics': metrics,
        'frame_indices': frame_indices
    }

def load_batch_masks(soft_mask_dir: Path, frame_indices: List[int], num_objects: int) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Load all object masks for a batch of frames with optimized I/O.
    
    Args:
        soft_mask_dir: Path to soft masks directory
        frame_indices: List of frame indices to load
        num_objects: Number of objects
        
    Returns:
        Dictionary mapping frame_idx to object_masks dictionary
    """
    batch_masks = {}
    
    # Pre-check which files exist to avoid repeated path operations
    existing_files = {}
    for frame_idx in frame_indices:
        existing_files[frame_idx] = {}
        for obj_id in range(1, num_objects + 1):
            mask_path = soft_mask_dir / str(obj_id) / f"{frame_idx:07d}.png"
            if mask_path.exists():
                existing_files[frame_idx][obj_id] = mask_path
    
    # Load masks in batches for better I/O performance
    for frame_idx in frame_indices:
        frame_masks = {}
        for obj_id in range(1, num_objects + 1):
            if obj_id in existing_files[frame_idx]:
                mask_path = existing_files[frame_idx][obj_id]
                try:
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        # Convert to binary immediately to save memory
                        frame_masks[obj_id] = (mask > 127).astype(np.uint8)
                except Exception as e:
                    print(f"Warning: Failed to load mask {mask_path}: {e}")
                    continue
        
        if len(frame_masks) >= 2:  # Only include frames with at least 2 objects
            batch_masks[frame_idx] = frame_masks
    
    return batch_masks

def calculate_frame_metrics_from_batch(object_masks: Dict[int, np.ndarray], num_objects: int) -> np.ndarray:
    """
    Calculate pairwise metrics for a frame using pre-loaded masks.
    
    Args:
        object_masks: Dictionary mapping object_id to mask array
        num_objects: Number of objects
        
    Returns:
        Structured numpy array containing pairwise metrics
    """
    # Define dtype for structured array
    dtype = [
        ('obj1', 'i4'),
        ('obj2', 'i4'),
        ('distance', 'f4'),
        ('overlap_ratio', 'f4'),
        ('contact_length', 'f4')
    ]
    
    # Initialize array for all pairs
    n_pairs = (num_objects * (num_objects - 1)) // 2
    metrics = np.zeros(n_pairs, dtype=dtype)
    
    # Calculate metrics for each pair
    pair_idx = 0
    for i in range(1, num_objects + 1):
        for j in range(i + 1, num_objects + 1):
            if i in object_masks and j in object_masks:
                mask1 = object_masks[i]
                mask2 = object_masks[j]
                
                # Calculate metrics
                pair_metrics = calculate_pairwise_metrics_optimized(mask1, mask2)
                
                # Store in array
                metrics[pair_idx] = (i, j, 
                                   pair_metrics['distance'],
                                   pair_metrics['overlap_ratio'],
                                   pair_metrics['contact_length'])
            pair_idx += 1
    
    return metrics 
