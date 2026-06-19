### Cutie GUI manual

Core workflow: annotate objects on key frames, propagate through the video, and commit good frames to **permanent memory** for stable tracking. The first frame that enters the memory bank is always committed to permanent memory. Use **Reset memory** if tracking drifts.

---

### Object controls (Show / Track)

Each object has two checkboxes:

| Checkbox | Meaning |
|----------|---------|
| **Show** | Include this object in the **on-screen overlay** when browsing or propagating. |
| **Track** | Include this object in **inference** (propagation and `masks/` saves). |

**Rules:**

- Checking **Track** automatically checks **Show**.
- Unchecking **Show** automatically unchecks **Track**.
- You may uncheck **Track** while leaving **Show** checked (visible but not propagated).

When you change Show or Track, the app rebuilds **`all_masks`** for the current frame (all objects for display), then updates **`masks/`** (tracked objects only for inference).

---

### Mask storage (workspace)

| Location | Contents | Used for |
|----------|----------|----------|
| **`masks/`** | Single indexed PNG per frame | Inference input; **tracked objects only** |
| **`all_masks/`** | Multi-channel `{frame}.npz` (one channel per object ID) | Overlay when browsing, mask metrics export, flexible propagation saves |
| **`seed_frames/`** | Multi-channel `{frame}.npz` (submitted anchors only) | Bidirectional propagation **seed** masks (explicitly submitted frames) |
| **`soft_masks/{id}/`** | Legacy per-object PNGs | Fallback if `all_masks` is missing |
| **`forward_masks/`**, **`backward_masks/`** | Indexed PNG per frame | Forward/backward inference outputs from bidirectional fill |
| **`iou_table.csv`** | Per-object IoU per frame (wide CSV) | IoU plot; below-threshold re-runs |

**Browsing (slider / frame dial):** The overlay is built from **`all_masks` + Show**, not directly from `masks/`. After fast propagation (which only wrote `masks/`), the first visit to a frame may build `all_masks` from disk sources.

---

### Propagation modes

The status line shows which mode is active, for example `[fast (masks only)]` or `[flexible (all_masks, fast overlay)]`.

**Save mode (disk I/O during propagation)**

| Condition | Label | What is saved each frame |
|-----------|--------|---------------------------|
| All objects **tracked and shown** | `fast (masks only)` | `masks/` only (fastest) |
| Otherwise | `flexible (all_masks, ...)` | `masks/` + **`all_masks/`** NPZ (if "Save soft mask" is checked) |

**Overlay mode (display speed during propagation)**

| Condition | Overlay path |
|-----------|----------------|
| **Track** set equals **Show** set | **Fast GPU overlay** from live probabilities (same idea as full fast mode) |
| Untracked-but-visible objects exist | **Full compose** - loads/merges `all_masks`, slower but correct |

**Tips for speed**

- Uncheck **Save soft mask** to skip `all_masks` writes (fastest flexible run).
- Uncheck **Include all visible objects in combined masks** to avoid reading previous NPZ when saving `all_masks`.
- **Save visualization** is not written frame-by-frame during propagation (export rebuilds overlays). Use **None** while propagating; use **Always** when scrubbing if you want preview images on disk.

**Console timing:** After propagation, the console prints average ms/frame for `inference`, `save`, `visualize`, and `ui` (e.g. `GUI: Propagation complete (N frames) avg ms/frame: ...`). Every 100 frames, a running average is printed. To see Python log lines as well, start the app with `--log-level INFO`.

---

### Propagation buttons

- **Propagate forward / backward** - standard propagation with memory carried across frames.
- **Step forward** - advance one frame without full propagation loop.
- **Propagate step forward** - propagate one frame and **clear memory** each step (useful when objects cross or occlude; reduces swapping IDs).
- **Submit seed frame** - copy the **current frame** masks (tracked objects only) to **`seed_frames/`** as anchors for bidirectional propagation. Re-submitting the same frame overwrites that seed. Does not replace normal `masks/` / `all_masks/` storage.
- **Commit to permanent memory** - pin the current frame in CUTIE memory (separate from submitting a seed).

**Heuristics (rats / occlusion):**

- Partial occlusion / split body: try **Propagate forward**.
- Static subject + nearby motion: try **Propagate step forward** for the static animal.
- Chase scenarios: step-forward on one rat, full forward on the other.

---

### Bidirectional propagation (gap fill)

Use this when you have a few **key annotated frames** and want to fill the video by running inference **forward and backward** from those anchors, then merge results.

**Workflow**

1. Annotate and correct masks on key frames (tracked objects only matter for seeds).
2. On each anchor frame, click **Submit seed frame**. Seeds are stored in **`seed_frames/`** only — frames with masks elsewhere are **not** used automatically.
3. In the **IoU plot panel** (bottom of the window), set **From** / **To** for the frame range. Optionally check **Only infer low IoU frames** to re-run only frames where any object IoU is below the **Threshold** (requires a prior bidirectional run with IoU data).
4. Click **Propagate Bidirectionally from annotations**.

**Requirements**

- At least one submitted seed must fall inside the selected frame range; otherwise propagation is blocked.
- Stop forward/backward propagation before starting bidirectional fill.

**What it writes**

| Output | Description |
|--------|-------------|
| **`forward_masks/`**, **`backward_masks/`** | Separate forward/backward inference masks |
| **`iou_table.csv`** | Forward vs backward IoU per object per frame |
| **`masks/`** | Merged masks (if **Overwrite inferred frames in masks/** is checked). **Submitted seed frames are never overwritten.** |

Merge picks forward or backward per frame based on **seed proximity** (nearest submitted seed wins).

**Typical use**

- First pass: set **From/To** covering the whole clip (or a segment), leave **Only infer low IoU frames** unchecked, with seeds on start/middle/end key frames.
- Refinement: check **Only infer low IoU frames**, adjust **Threshold** if needed, run again to update only weak frames within the same **From/To** range.

---

### IoU plot panel

Located at the **bottom** of the main window (below the timeline).

| Control | Purpose |
|---------|---------|
| **IoU plot window (frames)** | Width of the visible frame window in the scatter plot |
| **From / To** | Frame range for bidirectional inference |
| **Threshold** | IoU cutoff line on the plot; frames below this (when optional box is checked) are re-inferred |
| **Only infer low IoU frames** | If checked, re-infer only low-IoU frames in **From/To**; if unchecked, infer all frames in range |
| **Propagate Bidirectionally from annotations** | Start bidirectional gap fill |
| **Overwrite inferred frames in masks/** | Write merged results to `masks/` |
| **Scroll frames** | Pan the plot horizontally |

**Plot markers**

- Colored dots — IoU per object per frame (after a bidirectional run).
- Black dashed vertical line — current timeline frame.
- Black dotted horizontal line — IoU threshold.
- **Green dashed vertical lines** — submitted **seed frames** (visible even before the first IoU run).

**Click a dot** to jump the timeline to that frame and select that object.

---

### GUI layout

- **Center:** main canvas (current frame + mask overlay).
- **Right:** object list (Show / Track), memory gauges, **Manual**, console.
- **Bottom:** timeline slider, frame dial, propagation buttons (**Submit seed frame**, forward/backward, commit, …), visualization mode, export-related toggles.
- **Below timeline:** **IoU plot panel** (bidirectional controls, seed markers, IoU scatter plot).
- **Export...** dialog: mask metrics, visualization video, binary masks.

---

### Controls

- **Left-click** - foreground; **right-click** - background.
- **Number keys** or object spinbox - active object (must match `--num_objects` at startup).
- **Arrow keys** - prev/next frame; **Shift+arrow** - +/-10 frames; **Alt+arrow** - first/last frame.
- **F / Space** - propagate forward; **B** - propagate backward.
- **C** - commit current frame to permanent memory.
- **Submit seed frame** (button) - save current frame as a bidirectional propagation anchor in `seed_frames/`.
- **Middle-click** on canvas - toggle overlay target objects (popup, layer, RGBA, binary export).
- **Reset frame** - clear all masks on current frame (disk + display).
- **Reset object** - clear current object on current frame.

**Memory**

- **Reset all memory** - permanent + temporary.
- **Reset non-permanent memory** - temporary only.
- **Clear mask cache** - free RAM; masks reload from disk when needed.

---

### Overlay options

Visualization mode combo: `mask`, `davis`, `fade`, `light`, `popup`, `layer`, `rgba`.

**Save visualization** combo:

| Setting | Behavior |
|---------|----------|
| **None** | No automatic overlay files on disk |
| **Always** | Save overlay when viewing frames (not during propagation) |
| **Propagation only** | Same as None during propagation; use **Export video** to regenerate |

**Export as video** rebuilds visualization images from **`all_masks` / `soft_masks`** (all objects on disk) before encoding - you do not need per-frame saves during propagation.

**Layer mode:** use **Import layer** to insert an RGBA image between foreground and background.

---

### Export

- **Mask metrics** - reads **`all_masks`** when present (falls back to `soft_masks`).
- **Pairwise metrics** - NPZ with distance, overlap, contact length (select metrics in export dialog).
- **Binary / soft masks** - for tools like ProPainter (inpainting often prefers slight over-segmentation; increase dilation if needed).

---

### Command-line (optional)

```text
python interactive_gui.py --workspace PATH --num_objects N [--log-level INFO]
```

`--log-level INFO` enables extra propagation timing lines in the terminal in addition to the GUI console messages.

---

### About

Customized fork of [Cutie](https://github.com/hkchengrex/Cutie) with multi-object Show/Track control, combined **`all_masks`** storage, mask/pairwise metrics export, flexible vs fast propagation paths, **submitted seed frames** for bidirectional gap fill, and an interactive **IoU plot** for quality review and targeted re-runs.
