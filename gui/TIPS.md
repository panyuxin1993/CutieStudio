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

Controls:

- Use left-click for foreground annotation and right-click for background annotation.
- Use number keys or the spinbox to change the object to be operated on. If it does not respond, most likely the correct number of objects was not specified during program startup.
- Use left/right arrows to move between frames, shift+arrow to move by 10 frames, and alt/option+arrow to move to the start/end.
- Use F/space and B to propagate forward and backward, respectively.
- Use C to commit a frame to permanent memory.
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
- Soft masks are only saved for the "propagated" frames, not for the interacted frames. To save all frames, utilize forward and backward propagation.
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
