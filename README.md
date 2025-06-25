# CutieStudio - Enhanced Video Object Segmentation and Analysis

CutieStudio is an enhanced fork of [Cutie](https://hkchengrex.github.io/Cutie), adding powerful mask analysis and visualization capabilities to the original video object segmentation framework.

## Original Work
Cutie is a video object segmentation framework by [Ho Kei Cheng](https://hkchengrex.github.io/), [Seoung Wug Oh](https://sites.google.com/view/seoungwugoh/), [Brian Price](https://www.brianpricephd.com/), [Joon-Young Lee](https://joonyoung-cv.github.io/), and [Alexander Schwing](https://www.alexander-schwing.de/) from University of Illinois Urbana-Champaign and Adobe (CVPR 2024, Highlight).

[[arXiV]](https://arxiv.org/abs/2310.12982) [[PDF]](https://arxiv.org/pdf/2310.12982.pdf) [[Project Page]](https://hkchengrex.github.io/Cutie/)

## Enhanced Features

CutieStudio extends the original Cutie framework with:

- **Advanced Mask Analysis**
  - Frame-by-frame mask metrics (area, perimeter, circularity, orientation)
  - Object centroid tracking
  - Bounding box measurements
  - Custom object naming support

- **Pairwise Object Metrics**
  - Inter-object distance calculations
  - Overlap ratio analysis
  - Contact length measurements
  - Configurable metric selection

- **Enhanced Visualization**
  - Object visibility controls
  - Selective object tracking
  - Combined mask visualization
  - Soft mask handling

- **Improved Export Capabilities**
  - CSV export for mask metrics
  - NPZ format for pairwise metrics
  - Binary mask export
  - Enhanced video export options

- **Performance Optimizations**
  - Batch soft mask saving (5-7x speedup for 10+ objects)
  - In-memory mask caching
  - Selective object tracking
  - Configurable performance settings
  - Real-time performance monitoring

## Installation

Tested on Ubuntu and windows 11.

**Prerequisite:**
- Python 3.8+
- PyTorch 1.12+ and corresponding torchvision

**Clone the repository:**
```bash
git clone https://github.com/panyuxin1993/CutieStudio.git
```

**Install with pip:**
```bash
cd CutieStudio
pip install -e .
```

**Download the pretrained models:**
```python
python cutie/utils/download_models.py
```

## Specialized Models

In addition to the default models, we provide specialized fine-tuned models for specific use cases:

### Rat Segmentation Model
A fine-tuned model specifically optimized for rat behavior analysis and tracking. This model has been trained on extensive rat video datasets and provides enhanced accuracy for rodent tracking applications.

To access the rat segmentation model, please email panyuxin1993@gmail.com with:
- Your name and affiliation
- Brief description of your intended use
- Any specific requirements or constraints

We'll provide the model weights and setup instructions upon request.

## Quick Start

### Interactive GUI

Start the interactive GUI with:
```bash
python interactive_gui.py --video ./examples/example.mp4 --num_objects 2 --name_objects head left_hand
```

[See detailed instructions here](docs/INTERACTIVE.md)


## Performance Optimization

CutieStudio includes significant performance optimizations for tracking multiple objects (e.g., 10+ rats). The optimizations provide **5-7x speedup** for 10+ objects compared to the original implementation.

### Key Optimizations

1. **Batch Soft Mask Saving**: Reduces I/O operations by saving all object masks in a single batch operation
2. **In-Memory Mask Caching**: Eliminates disk reads for recently accessed frames
3. **Selective Object Tracking**: Only saves masks for tracked objects
4. **Configurable Performance Settings**: Adjust optimization levels based on your needs

### Performance Configuration

Configure performance settings in `cutie/config/gui_config.yaml`:

```yaml
performance:
  batch_save_soft_masks: True    # Enable batch saving (recommended)
  enable_mask_cache: True        # Enable caching (recommended)
  max_cache_size: 50            # Cache size limit
  lazy_saving: True             # Only save when necessary
  save_only_tracked: True       # Only save tracked objects
```

### Performance Testing

Test the performance improvements:

```bash
python test_performance.py
```

### Memory Management

For long videos or limited memory:
- Use the "Clear mask cache" button in the GUI
- Reduce `max_cache_size` in the configuration
- Restart the application for very long sessions

See [PERFORMANCE_OPTIMIZATION.md](PERFORMANCE_OPTIMIZATION.md) for detailed technical information.

## Recent Updates


## Citation

Please cite both the original Cutie paper and this enhanced version:

```bibtex
@inproceedings{cheng2023putting,
  title={Putting the Object Back into Video Object Segmentation},
  author={Cheng, Ho Kei and Oh, Seoung Wug and Price, Brian and Lee, Joon-Young and Schwing, Alexander},
  booktitle={arXiv},
  year={2023}
}
```

## References

- Original Cutie implementation: [hkchengrex/Cutie](https://github.com/hkchengrex/Cutie)
- The mask area calculation functionality and GUI optimization was contributed by [Yuxin Pan](https://github.com/panyuxin1993)
- The GUI tools uses [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation) for interactive image segmentation
- For automatic video segmentation, see [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA)
- The interactive demo is developed upon [IVS](https://github.com/seoungwugoh/ivs-demo), [MiVOS](https://github.com/hkchengrex/MiVOS), and [XMem](https://github.com/hkchengrex/XMem)
- We used [ProPainter](https://github.com/sczhou/ProPainter) in our video inpainting demo
