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
| **`soft_masks/{id}/`** | Legacy per-object PNGs | Fallback if `all_masks` is missing |

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

**Heuristics (rats / occlusion):**

- Partial occlusion / split body: try **Propagate forward**.
- Static subject + nearby motion: try **Propagate step forward** for the static animal.
- Chase scenarios: step-forward on one rat, full forward on the other.

---

### GUI layout

- **Center:** main canvas (current frame + mask overlay).
- **Right:** object list (Show / Track), memory gauges, **Manual**, console.
- **Bottom:** timeline slider, frame dial, propagation buttons, visualization mode, export-related toggles.
- **Export...** dialog: mask metrics, visualization video, binary masks.

---

### Controls

- **Left-click** - foreground; **right-click** - background.
- **Number keys** or object spinbox - active object (must match `--num_objects` at startup).
- **Arrow keys** - prev/next frame; **Shift+arrow** - +/-10 frames; **Alt+arrow** - first/last frame.
- **F / Space** - propagate forward; **B** - propagate backward.
- **C** - commit current frame to permanent memory.
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

Customized fork of [Cutie](https://github.com/hkchengrex/Cutie) with multi-object Show/Track control, combined **`all_masks`** storage, mask/pairwise metrics export, and flexible vs fast propagation paths.
