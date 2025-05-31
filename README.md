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

## Installation

Tested on Ubuntu only.

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

### Mask Analysis

Export mask metrics to CSV:
```bash
python interactive_gui.py --video ./examples/example.mp4 --num_objects 2 --name_objects head left_hand --export_metrics
```

### Pairwise Metrics

Calculate and save pairwise object metrics:
```bash
python interactive_gui.py --video ./examples/example.mp4 --num_objects 2 --name_objects head left_hand --export_pairwise
```

## Recent Updates

### 2024-03
- Added comprehensive mask metrics system
  - Area, perimeter, circularity calculations
  - Object centroid tracking
  - Bounding box measurements
  - Custom object naming support
- Implemented pairwise object metrics
  - Distance between object centroids
  - Overlap ratio analysis
  - Contact length measurements
- Enhanced visualization system
  - Object visibility controls
  - Selective object tracking
  - Combined mask visualization
- Improved export capabilities
  - CSV export for mask metrics
  - NPZ format for pairwise metrics
  - Enhanced video export options

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
