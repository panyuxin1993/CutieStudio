import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def get_davis_color_mapping():
    """
    Returns the standard DAVIS dataset color mapping.
    Colors are in RGB format.
    """
    # Standard DAVIS color palette (RGB format)
    davis_colors = [
        (0, 0, 0),       # Background (black)
        (128, 0, 0),     # Object 1 (red)
        (0, 128, 0),     # Object 2 (green)
        (128, 128, 0),   # Object 3 (yellow)
        (0, 0, 128),     # Object 4 (blue)
        (128, 0, 128),   # Object 5 (magenta)
        (0, 128, 128),   # Object 6 (cyan)
        (128, 128, 128), # Object 7 (gray)
        (64, 0, 0),      # Object 8 (dark red)
        (192, 0, 0),     # Object 9 (bright red)
        (64, 128, 0),    # Object 10 (olive)
        (192, 128, 0),   # Object 11 (orange)
        (64, 0, 128),    # Object 12 (purple)
        (192, 0, 128),   # Object 13 (pink)
        (64, 128, 128),  # Object 14 (teal)
        (192, 128, 128)  # Object 15 (rosy)
    ]
    return davis_colors

def calculate_mask_areas(mask_dir):
    """
    Calculate areas for each object mask in DAVIS dataset frames.
    
    The masks contain multiple colors, where each unique color represents
    a different object. Colors follow the standard DAVIS dataset palette.
    
    Args:
        mask_dir (str or Path): Directory containing mask PNG files
        
    Returns:
        pd.DataFrame: DataFrame with frame indices and object areas
    """
    mask_dir = Path(mask_dir)
    mask_files = sorted(mask_dir.glob('*.png'))  # Get all PNG files
    
    if not mask_files:
        raise ValueError(f"No PNG files found in {mask_dir}")
        
    # Read first mask to get number of objects
    first_mask = cv2.imread(str(mask_files[0]))  # Read as BGR
    if first_mask is None:
        raise ValueError(f"Could not read mask file: {mask_files[0]}")
    
    # Convert to RGB for more intuitive color representation
    first_mask_rgb = cv2.cvtColor(first_mask, cv2.COLOR_BGR2RGB)
    
    # Find unique colors by reshaping to 2D array of RGB values
    colors = first_mask_rgb.reshape(-1, 3)
    unique_colors = np.unique(colors, axis=0)
    
    # Get standard DAVIS colors
    davis_colors = get_davis_color_mapping()
    
    # Create mapping from RGB values to sequential IDs based on DAVIS order
    color_to_id = {}
    print("\nObject color mapping (DAVIS standard order):")
    print("Background: RGB(0, 0, 0)")
    
    # First find which DAVIS colors are present in the mask
    next_id = 1
    for davis_color in davis_colors[1:]:  # Skip background
        if any(np.all(color == davis_color) for color in unique_colors):
            color_to_id[davis_color] = next_id
            print(f"Object {next_id}: RGB{davis_color} ({get_color_name(davis_color)})")
            next_id += 1
    
    # Handle any colors not in standard DAVIS palette (should be rare)
    for color in unique_colors:
        color_tuple = tuple(color)
        if color_tuple == (0, 0, 0):  # Skip background
            continue
        if color_tuple not in color_to_id and not any(np.all(color == dc) for dc in davis_colors):
            color_to_id[color_tuple] = next_id
            print(f"Object {next_id}: RGB{color_tuple} (Non-standard color)")
            next_id += 1
    print()
    
    # Initialize results dictionary
    results = {
        'frame': []  # Frame index column
    }
    
    # Add columns for each object using sequential IDs
    for obj_id in range(1, len(color_to_id) + 1):
        results[f'object_{obj_id}'] = []
    
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
        
        # Calculate area for each object
        for color, obj_id in color_to_id.items():
            # Create boolean mask for this color
            color_mask = np.all(mask_rgb == color, axis=2)
            area = np.sum(color_mask)
            results[f'object_{obj_id}'].append(area)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df

def get_color_name(rgb):
    """Returns a human-readable name for standard DAVIS colors"""
    color_names = {
        (128, 0, 0): "Red",
        (0, 128, 0): "Green",
        (128, 128, 0): "Yellow",
        (0, 0, 128): "Blue",
        (128, 0, 128): "Magenta",
        (0, 128, 128): "Cyan",
        (128, 128, 128): "Gray",
        (64, 0, 0): "Dark Red",
        (192, 0, 0): "Bright Red",
        (64, 128, 0): "Olive",
        (192, 128, 0): "Orange",
        (64, 0, 128): "Purple",
        (192, 0, 128): "Pink",
        (64, 128, 128): "Teal",
        (192, 128, 128): "Rosy"
    }
    return color_names.get(rgb, "Unknown")

def main():
    """Main function to run the mask area calculation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Calculate object areas from DAVIS dataset masks. '
                  'Colors are mapped according to standard DAVIS palette.'
    )
    parser.add_argument('mask_dir', type=str, help='Directory containing mask PNG files')
    parser.add_argument('--output', type=str, default=None, 
                      help='Output CSV file path (default: ../mask_area.csv relative to mask_dir)')
    
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
        df = calculate_mask_areas(args.mask_dir)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
        print("Column names use sequential object IDs following DAVIS color order")
        print("See the color mapping above to identify which color corresponds to each object")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
        
    return 0

if __name__ == '__main__':
    exit(main()) 