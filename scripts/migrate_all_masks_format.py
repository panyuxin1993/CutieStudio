"""
Migration script to convert all_masks files to compressed format.

This script converts all .npy files in the all_masks directory to compressed .npz format
with uint8 binary masks (0 or 1), achieving similar compression to individual PNG soft masks.

Usage:
    python scripts/migrate_all_masks_format.py --workspace <workspace_path>
    python scripts/migrate_all_masks_format.py --all_masks_dir <all_masks_directory>
"""

import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def migrate_all_masks_directory(all_masks_dir: str, dry_run: bool = False, skip_existing: bool = False):
    """
    Migrate all .npy files in all_masks directory to compressed .npz format with uint8.
    
    Converts from:
    - Old: .npy files with float32 (4 bytes/pixel) or uint8 (1 byte/pixel, uncompressed)
    - New: .npz files with uint8 (1 byte/pixel, compressed) - similar size to individual PNG masks
    
    No backups are created (impractical for large datasets). Use --skip-existing to resume
    after interruption.
    
    Args:
        all_masks_dir: Path to the all_masks directory
        dry_run: If True, only report what would be converted without actually converting
        skip_existing: If True, skip .npy files that already have a corresponding .npz
    """
    all_masks_path = Path(all_masks_dir)
    
    if not all_masks_path.exists():
        print(f"Error: Directory {all_masks_dir} does not exist")
        return
    
    # Find all .npy files (ignore .npz files as they're already in new format)
    npy_files = list(all_masks_path.glob("*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {all_masks_dir}")
        # Check if there are .npz files
        npz_files = list(all_masks_path.glob("*.npz"))
        if npz_files:
            print(f"Found {len(npz_files)} .npz files (already in compressed format)")
        return
    
    print(f"Found {len(npy_files)} .npy files in {all_masks_dir}")
    
    # For large batches, skip detailed analysis and just collect files
    # All .npy files need migration to .npz (compressed format)
    files_to_migrate = []
    total_old_size = 0
    total_new_size = 0
    
    print("Scanning files...")
    # Use tqdm for progress during scanning
    for npy_file in tqdm(npy_files, desc="Scanning", unit="files"):
        try:
            # Get file size first (fast)
            old_size = npy_file.stat().st_size
            total_old_size += old_size
            
            # Only load file if we need to check dtype (for small batches) or estimate size
            # For large batches, just add to migration list
            if len(npy_files) > 1000:
                # Large batch: skip detailed analysis, just add all files
                files_to_migrate.append(npy_file)
                # Rough estimate: assume 5x compression
                total_new_size += old_size / 5.0
            else:
                # Small batch: check dtype and estimate more accurately
                mask = np.load(npy_file)
                files_to_migrate.append(npy_file)
                
                # Convert to uint8 if needed for size estimation
                if mask.dtype == np.float32:
                    mask_uint8 = (mask > 0.5).astype(np.uint8)
                elif mask.dtype == np.uint8:
                    mask_uint8 = mask
                else:
                    print(f"  {npy_file.name}: Unexpected dtype {mask.dtype}, skipping")
                    files_to_migrate.pop()  # Remove from list
                    total_old_size -= old_size
                    continue
                
                # Estimate new size: compressed .npz should be similar to sum of individual PNG masks
                array_size_bytes = mask_uint8.nbytes
                estimated_compression = 5.0  # Conservative estimate
                estimated_new_size += array_size_bytes / estimated_compression + 1000
        except Exception as e:
            print(f"\n  {npy_file.name}: Error reading file - {e}")
    
    if not files_to_migrate:
        print("No files need migration")
        return
    
    print(f"\nFiles to migrate: {len(files_to_migrate)}")
    print(f"Total size before: {total_old_size / (1024**2):.2f} MB")
    if len(npy_files) > 1000:
        print(f"Estimated size after (compressed): ~{total_new_size / (1024**2):.2f} MB (rough estimate)")
        print(f"Estimated space saved: ~{(total_old_size - total_new_size) / (1024**2):.2f} MB (~{((total_old_size - total_new_size) / total_old_size * 100):.1f}%)")
        print(f"\nNote: For large batches, size estimates are approximate. Actual compression")
        print(f"      will be calculated after migration. Compressed .npz files should be")
        print(f"      similar in size to the sum of individual PNG soft masks.")
    else:
        print(f"Estimated size after (compressed): {total_new_size / (1024**2):.2f} MB")
        print(f"Estimated space saved: {(total_old_size - total_new_size) / (1024**2):.2f} MB ({((total_old_size - total_new_size) / total_old_size * 100):.1f}%)")
        print(f"\nNote: Compressed .npz files should be similar in size to the sum of individual PNG soft masks")
    
    if dry_run:
        print("\n[DRY RUN] Would migrate the following files:")
        for f in files_to_migrate[:10]:  # Show first 10
            print(f"  {f.name} -> {f.stem}.npz")
        if len(files_to_migrate) > 10:
            print(f"  ... and {len(files_to_migrate) - 10} more files")
        return
    
    # Migrate files
    print("\nMigrating files...")
    migrated_count = 0
    error_count = 0
    skipped_count = 0
    actual_old_migrated = 0  # Sum of .npy sizes we migrated (before delete)
    actual_new_size = 0     # Sum of .npz sizes we created
    
    for npy_file in tqdm(files_to_migrate, desc="Converting files"):
        try:
            # Use absolute paths to avoid Windows path resolution issues
            npy_path = npy_file.resolve()
            npz_path = npy_path.with_suffix('.npz')
            
            # Skip if .npz already exists (resume after interruption)
            if skip_existing and npz_path.exists():
                skipped_count += 1
                continue
            
            # Load the mask
            mask = np.load(str(npy_path))
            
            # Convert to binary uint8 (0 or 1) if needed
            if mask.dtype == np.float32:
                mask_uint8 = (mask > 0.5).astype(np.uint8)
            elif mask.dtype == np.uint8:
                mask_uint8 = mask
            else:
                raise ValueError(f"Unsupported dtype: {mask.dtype}")
            
            # Save new compressed .npz format (no backup - impractical for large datasets)
            np.savez_compressed(str(npz_path), mask=mask_uint8)
            
            # Verify the conversion
            loaded_data = np.load(str(npz_path))
            loaded_mask = loaded_data['mask']
            if loaded_mask.dtype != np.uint8:
                raise ValueError("Conversion failed - file is not uint8")
            
            # Track sizes before deleting .npy
            actual_old_migrated += npy_path.stat().st_size
            actual_new_size += npz_path.stat().st_size
            
            # Remove old .npy file only after successful conversion
            npy_path.unlink()
            
            migrated_count += 1
            
        except Exception as e:
            print(f"\nError migrating {npy_file.name}: {e}")
            error_count += 1
    
    print(f"\nMigration complete!")
    print(f"  Successfully migrated: {migrated_count} files")
    if skipped_count > 0:
        print(f"  Skipped (already have .npz): {skipped_count} files")
    if error_count > 0:
        print(f"  Errors: {error_count} files")
    
    # Calculate actual space saved (only for files we actually migrated)
    if migrated_count > 0 and actual_old_migrated > 0 and actual_new_size > 0:
        actual_saved = actual_old_migrated - actual_new_size
        print(f"\nActual space saved (this run): {actual_saved / (1024**2):.2f} MB ({(actual_saved / actual_old_migrated * 100):.1f}%)")
        print(f"Average .npz file size: {actual_new_size / migrated_count / 1024:.1f} KB per file")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate all_masks files to compressed .npz format with uint8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate all_masks in a workspace (converts .npy to compressed .npz)
  python scripts/migrate_all_masks_format.py --workspace ./workspace/my_video
  
  # Migrate a specific all_masks directory
  python scripts/migrate_all_masks_format.py --all_masks_dir ./workspace/my_video/all_masks
  
  # Dry run to see what would be converted
  python scripts/migrate_all_masks_format.py --workspace ./workspace/my_video --dry-run
  
  # Resume after interruption (skip .npy that already have .npz)
  python scripts/migrate_all_masks_format.py --all_masks_dir ./workspace/my_video/all_masks --skip-existing
  
Note: No backups are created (impractical for large datasets). Use --skip-existing to resume.
This converts .npy files to compressed .npz format, achieving similar
file sizes to the sum of individual PNG soft masks (typically 10-50x smaller
than uncompressed .npy files for binary masks).
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--workspace', type=str, help='Path to workspace directory (contains all_masks folder)')
    group.add_argument('--all_masks_dir', type=str, help='Path directly to all_masks directory')
    
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be converted without actually converting')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip .npy files that already have a .npz (resume after interruption)')
    
    args = parser.parse_args()
    
    # Determine all_masks directory
    if args.workspace:
        all_masks_dir = os.path.join(args.workspace, 'all_masks')
    else:
        all_masks_dir = args.all_masks_dir
    
    print(f"Migrating all_masks in: {all_masks_dir}")
    if args.dry_run:
        print("[DRY RUN MODE - No files will be modified]")
    print()
    
    migrate_all_masks_directory(all_masks_dir, dry_run=args.dry_run, skip_existing=args.skip_existing)


if __name__ == '__main__':
    main()
