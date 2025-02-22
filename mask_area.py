import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from cutie.utils.palette import davis_palette_np

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