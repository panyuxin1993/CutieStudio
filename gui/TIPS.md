### Tips

Core mechanism: annotate objects at one or more frames and use propagation to complete the video.
Use permanent memory to store accurate segmentation (commit good frames to it) for best results.
The first frame to enter the memory bank is always committed to the permanent memory.
Reset memory if needed.

GUI Layout:

- The main canvas is in the center, showing the current frame with object masks.
- The right panel contains the object list and memory controls:
  - Object list shows all objects with their IDs, names, and control checkboxes
  - "Show" checkbox controls whether an object's mask is visible in the main canvas
  - "Track" checkbox controls whether an object is included in propagation
  - **Logical relationship**: When "Track" is checked, "Show" is automatically checked (you want to see what you're tracking). When "Show" is unchecked, "Track" is automatically unchecked (you can't track what you can't see).
  - Memory gauges and controls are below the object list:
    - "Reset all memory" clears both permanent and temporary memory
    - "Reset non-permanent memory" clears only temporary memory
    - "Clear mask cache" frees memory by clearing cached mask data
  - The 'manual' buttom will show tips 
  - console output
- The bottom panel contains frame navigation and propagation controls
- The left panel contains additional controls:
  - Mask metrics export controls
  - Pairwise metrics controls with checkboxes for different metric types
  - Save soft mask toggle for tracking
  - Include all visible objects in combined masks toggle - when enabled, includes all visible objects in combined masks

Controls:

- Use left-click for foreground annotation and right-click for background annotation.
- Use number keys or the spinbox to change the object to be operated on. If it does not respond, most likely the correct number of objects was not specified during program startup.
- Use left/right arrows to move between frames, shift+arrow to move by 10 frames, and alt/option+arrow to move to the start/end.
- Use F/space and B to propagate forward and backward, respectively.
- Use C to commit a frame to permanent memory.
- **Reset buttons**:
  - "Reset frame" clears all masks in the current frame (removes soft masks from disk and updates display immediately)
  - "Reset object" clears only the current object's masks in the current frame (removes soft mask from disk and updates display immediately)
  - Both buttons now provide immediate visual feedback - masks disappear from the display right away
- **Object visibility and tracking controls**: The "Show" and "Track" checkboxes are logically linked:
  - Checking "Track" automatically checks "Show" (you want to see what you're tracking)
  - Unchecking "Show" automatically unchecks "Track" (you can't track what you can't see)
  - You can uncheck "Track" without affecting "Show" (allows showing objects without tracking them)
- Memory can be corrupted by bad segmentations. Make good use of "reset memory" and do not commit bad segmentations.
- "Export as video" only aggregates visualizations that are saved on disks. You need to check "save overlay" for that to happen.

Memory Management:

- "Reset all memory" completely clears the model's memory, including permanent memory. Use this when you want to start fresh.
- "Reset non-permanent memory" clears temporary memory while preserving permanent memory. Use this to free up memory without losing committed frames.
- "Clear mask cache" removes cached mask data from memory to free up RAM. This is useful for long videos or when working with limited memory. The cache will be rebuilt as needed when you navigate to frames.

Visualizations:

- Middle-click on target objects to toggle some visualization effects (for layered, popout, RGBA, and binary mask export).
- Soft masks are saved in the 'soft_masks' directory, with one subdirectory per object
- Combined masks are saved in the 'all_masks' directory, containing all objects in a single mask
- **Propagation logic**: During propagation, tracked objects from current probabilities are shown first, followed by existing soft masks for untracked objects (if enabled)
- Soft masks are only saved for the "propagated" frames, not for the interacted frames. To save all frames, utilize forward and backward propagation.
- The "Include all visible objects in combined masks" checkbox controls whether combined masks include all visible objects or only tracked objects:
  - When enabled (default): Combined masks include tracked objects from current probabilities plus existing soft masks for untracked objects
  - When disabled: Combined masks only include tracked objects from current probabilities
  - Soft masks are always saved only for tracked objects to prevent overwriting existing data for untracked objects
  - This allows you to visualize all objects while preserving existing soft masks for objects that are visible but not tracked
- During tracking (propagation), all visible objects are now shown in real-time, including untracked objects with existing soft masks
- Real-time visualization now works correctly regardless of whether previous results exist or not
- All tracked objects are now visible during propagation, even when some objects have existing soft masks and others don't
- For some visualizations (layered and RGBA), the images saved during propagation will be higher quality with soft edges. This is because we have access to the soft mask only during propagation. Set the save visualization mode to "Propagation only" to only save during propagation.
- The "layered" visualization mode inserts an RGBA layer between the foreground and the background. Use "import layer" to select a new layer.

Exporting:

- Exported binary/soft masks can be used in other applications like ProPainter. Note inpainting prefer over-segmentation over under-segmentation -- use a larger dilation radius if needed
- Mask metrics can be exported to CSV format, including area, perimeter, circularity, and orientation metrics
- Pairwise metrics can be exported to NPZ format, including:
  - Distance between object centroids
  - Overlap ratio between objects
  - Contact length between objects
- Use the checkboxes to select which pairwise metrics to calculate and save

About:
This project is a customized fork of Cutie (hkchengrex/Cutie) with additional features for mask analysis and visualization. 
Major improvements include:
- Enhanced mask metrics and analysis
- Pairwise object metrics
- Improved object visibility and tracking
- Advanced mask visualization options
- Extended export capabilities

Original project: hkchengrex/Cutie
