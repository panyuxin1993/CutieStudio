"""Interactive IoU scatter plot for bidirectional fill results."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from omegaconf import DictConfig
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cutie.utils.palette import davis_palette_np
from utils.inference_eval import frames_below_iou_threshold


class IoUPlotCanvas(QWidget):
    """Scatter plot canvas: IoU vs frame for each object."""

    point_clicked = Signal(int, int)

    MARGIN_LEFT = 48
    MARGIN_RIGHT = 12
    MARGIN_TOP = 8
    MARGIN_BOTTOM = 28
    DOT_RADIUS = 4
    HIT_RADIUS = 8
    PLACEHOLDER_TEXT = 'Run bidirectional fill to generate IoU data'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        self._data: Optional[Dict] = None
        self._window_frames = 100
        self._scroll_start = 0
        self._threshold = 0.5
        self._current_frame: Optional[int] = None
        self._seed_frames: List[int] = []
        self._total_frames: int = 1
        self._dot_positions: List[Tuple[int, int, float, float]] = []

    def set_data(self, data: Optional[Dict]) -> None:
        self._data = data
        self._dot_positions = []
        self.update()

    def set_window_frames(self, n: int) -> None:
        self._window_frames = max(1, n)
        self.update()

    def set_scroll_start(self, start: int) -> None:
        self._scroll_start = max(0, start)
        self.update()

    def set_threshold(self, threshold: float) -> None:
        self._threshold = max(0.0, min(1.0, threshold))
        self.update()

    def set_current_frame(self, frame_idx: int) -> None:
        self._current_frame = frame_idx
        self.update()

    def set_seed_frames(self, seed_frames: List[int]) -> None:
        self._seed_frames = sorted(set(seed_frames))
        self.update()

    def set_total_frames(self, total_frames: int) -> None:
        self._total_frames = max(1, total_frames)
        self.update()

    def _plot_rect(self):
        w = self.width()
        h = self.height()
        return (
            self.MARGIN_LEFT,
            self.MARGIN_TOP,
            w - self.MARGIN_LEFT - self.MARGIN_RIGHT,
            h - self.MARGIN_TOP - self.MARGIN_BOTTOM,
        )

    def _visible_frame_range(self) -> Tuple[int, int]:
        last_frame = max(0, self._total_frames - 1)
        start = self._scroll_start
        end = min(start + self._window_frames - 1, last_frame)
        return start, end

    def _frame_to_x(self, frame_idx: int, x0: float, plot_w: float, start: int, end: int) -> float:
        if end <= start:
            return x0 + plot_w / 2
        return x0 + (frame_idx - start) / (end - start) * plot_w

    def _iou_to_y(self, iou: float, y0: float, plot_h: float) -> float:
        return y0 + plot_h * (1.0 - iou)

    def _build_dot_positions(self) -> None:
        self._dot_positions = []
        if not self._data:
            return

        x0, y0, plot_w, plot_h = self._plot_rect()
        start, end = self._visible_frame_range()
        if plot_w <= 0 or plot_h <= 0:
            return

        for obj_id in self._data['object_ids']:
            obj_frames = self._data['values'].get(obj_id, {})
            for frame_idx, iou in obj_frames.items():
                if frame_idx < start or frame_idx > end:
                    continue
                px = self._frame_to_x(frame_idx, x0, plot_w, start, end)
                py = self._iou_to_y(iou, y0, plot_h)
                self._dot_positions.append((frame_idx, obj_id, px, py))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        painter.fillRect(0, 0, w, h, QColor(245, 245, 245))

        x0, y0, plot_w, plot_h = self._plot_rect()
        start, end = self._visible_frame_range()
        has_iou_data = bool(self._data and self._data.get('frame_indices'))

        if not has_iou_data and not self._seed_frames:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.PLACEHOLDER_TEXT)
            painter.end()
            return

        if not has_iou_data:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(
                int(x0), int(y0 - 2), int(plot_w), 16,
                Qt.AlignmentFlag.AlignCenter,
                'Submitted seed frames (run propagation for IoU data)',
            )

        painter.setPen(QPen(QColor(180, 180, 180), 1))
        painter.drawRect(int(x0), int(y0), int(plot_w), int(plot_h))

        if has_iou_data:
            painter.setPen(QColor(80, 80, 80))
            for tick_iou in (0.0, 0.5, 1.0):
                ty = self._iou_to_y(tick_iou, y0, plot_h)
                painter.drawLine(int(x0), int(ty), int(x0 + plot_w), int(ty))
                painter.drawText(4, int(ty + 4), f'{tick_iou:.1f}')

            threshold_y = self._iou_to_y(self._threshold, y0, plot_h)
            pen = QPen(QColor(0, 0, 0), 1, Qt.PenStyle.DotLine)
            painter.setPen(pen)
            painter.drawLine(int(x0), int(threshold_y), int(x0 + plot_w), int(threshold_y))

        seed_pen = QPen(QColor(0, 140, 0), 1, Qt.PenStyle.DashLine)
        painter.setPen(seed_pen)
        for seed_idx in self._seed_frames:
            if seed_idx < start or seed_idx > end:
                continue
            sx = self._frame_to_x(seed_idx, x0, plot_w, start, end)
            painter.drawLine(int(sx), int(y0), int(sx), int(y0 + plot_h))

        if self._current_frame is not None and start <= self._current_frame <= end:
            cx = self._frame_to_x(self._current_frame, x0, plot_w, start, end)
            painter.setPen(QPen(QColor(0, 0, 0), 1, Qt.PenStyle.DashLine))
            painter.drawLine(int(cx), int(y0), int(cx), int(y0 + plot_h))

        self._build_dot_positions()
        if has_iou_data:
            for frame_idx, obj_id, px, py in self._dot_positions:
                if obj_id < len(davis_palette_np):
                    r, g, b = davis_palette_np[obj_id]
                else:
                    r, g, b = 128, 128, 128
                color = QColor(int(r), int(g), int(b))
                painter.setBrush(color)
                painter.setPen(QPen(color.darker(120), 1))
                painter.drawEllipse(int(px - self.DOT_RADIUS), int(py - self.DOT_RADIUS),
                                    self.DOT_RADIUS * 2, self.DOT_RADIUS * 2)

        painter.setPen(QColor(80, 80, 80))
        mid_y = int(y0 + plot_h + 20)
        painter.drawText(int(x0), mid_y, str(start))
        painter.drawText(int(x0 + plot_w - 30), mid_y, str(end))
        painter.drawText(int(x0 + plot_w / 2 - 20), mid_y, 'frame')

        painter.end()

    def mousePressEvent(self, event):
        if not self._dot_positions or event.button() != Qt.MouseButton.LeftButton:
            return

        mx, my = event.position().x(), event.position().y()
        best = None
        best_dist = self.HIT_RADIUS ** 2

        for frame_idx, obj_id, px, py in self._dot_positions:
            dist = (mx - px) ** 2 + (my - py) ** 2
            if dist <= best_dist:
                best_dist = dist
                best = (frame_idx, obj_id)

        if best is not None:
            self.point_clicked.emit(best[0], best[1])


class IoUPlotPanel(QWidget):
    """Controls + canvas for interactive IoU plot."""

    point_clicked = Signal(int, int)

    def __init__(self, controller, cfg: DictConfig, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.cfg = cfg

        bi_cfg = cfg.get('bidirectional', {})
        default_window = int(bi_cfg.get('iou_plot_window_frames', 100))
        default_threshold = float(bi_cfg.get('iou_plot_threshold', 0.5))
        panel_height = int(bi_cfg.get('iou_plot_height', 200))
        max_frames = max(1, controller.T)
        last_frame = max(0, max_frames - 1)

        self.setMinimumHeight(panel_height)
        self.setMaximumHeight(panel_height + 60)

        self._data: Optional[Dict] = None
        self._updating_scroll = False
        self._total_frames = max_frames

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        controls = QHBoxLayout()
        controls.addWidget(QLabel('IoU plot window (frames):'))
        self.window_spin = QSpinBox()
        self.window_spin.setRange(10, max_frames)
        self.window_spin.setValue(min(default_window, max_frames))
        self.window_spin.valueChanged.connect(self._on_window_changed)
        controls.addWidget(self.window_spin)

        controls.addWidget(QLabel('From'))
        self.range_start_spin = QSpinBox()
        self.range_start_spin.setRange(0, last_frame)
        self.range_start_spin.setValue(0)
        controls.addWidget(self.range_start_spin)

        controls.addWidget(QLabel('To'))
        self.range_end_spin = QSpinBox()
        self.range_end_spin.setRange(0, last_frame)
        self.range_end_spin.setValue(last_frame)
        controls.addWidget(self.range_end_spin)

        controls.addWidget(QLabel('Threshold:'))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(default_threshold)
        self.threshold_spin.valueChanged.connect(self._on_threshold_changed)
        controls.addWidget(self.threshold_spin)

        self.only_low_iou_cb = QCheckBox('Only infer low IoU frames')
        self.only_low_iou_cb.setToolTip(
            'When checked, re-infer only frames in the From–To range where any '
            'object IoU is below the threshold (requires a prior bidirectional run). '
            'When unchecked, infer all frames in the From–To range.'
        )
        controls.addWidget(self.only_low_iou_cb)

        self.propagate_button = QPushButton('Propagate Bidirectionally from annotations')
        self.propagate_button.setMinimumWidth(280)
        self.propagate_button.setToolTip(
            'Run forward and backward inference from submitted seed frames in range, '
            'compare masks (IoU), save iou_table.csv, forward/backward masks, '
            'and merged masks to masks/. Submitted seed frames are preserved.'
        )
        controls.addWidget(self.propagate_button)

        self.overwrite_cb = QCheckBox('Overwrite inferred frames in masks/')
        self.overwrite_cb.setChecked(bool(bi_cfg.get('overwrite_inferred', True)))
        self.overwrite_cb.setToolTip(
            'Write merged gap-fill results to masks/. Annotated seed frames are never overwritten.'
        )
        controls.addWidget(self.overwrite_cb)

        controls.addStretch(1)
        layout.addLayout(controls)

        self.canvas = IoUPlotCanvas(self)
        self.canvas.point_clicked.connect(self.point_clicked.emit)
        layout.addWidget(self.canvas, 1)

        scroll_row = QHBoxLayout()
        scroll_row.addWidget(QLabel('Scroll frames:'))
        self.scroll_slider = QSlider(Qt.Orientation.Horizontal)
        self.scroll_slider.setMinimum(0)
        self.scroll_slider.setMaximum(0)
        self.scroll_slider.setValue(0)
        self.scroll_slider.valueChanged.connect(self._on_scroll_changed)
        scroll_row.addWidget(self.scroll_slider, 1)
        layout.addLayout(scroll_row)

        self.canvas.set_window_frames(self.window_spin.value())
        self.canvas.set_threshold(self.threshold_spin.value())
        self.canvas.set_total_frames(max_frames)
        self._update_inference_scope_controls()

    def get_propagation_settings(self) -> Dict:
        """Return frame selection and overwrite options for bidirectional propagation."""
        only_low_iou = self.only_low_iou_cb.isChecked()
        threshold = float(self.threshold_spin.value())
        start = int(self.range_start_spin.value())
        end = int(self.range_end_spin.value())
        if start > end:
            start, end = end, start

        settings = {
            'only_low_iou': only_low_iou,
            'frame_range': (start, end),
            'user_frame_range': (start, end),
            'threshold': threshold,
            'overwrite_inferred': self.overwrite_cb.isChecked(),
            'output_frames': None,
        }

        if only_low_iou:
            target_frames = frames_below_iou_threshold(self._data or {}, threshold)
            target_frames = [f for f in target_frames if start <= f <= end]
            if target_frames:
                settings['output_frames'] = set(target_frames)
        return settings

    def set_frame_count(self, total_frames: int) -> None:
        last_frame = max(0, total_frames - 1)
        self._total_frames = max(1, total_frames)
        self.range_start_spin.setMaximum(last_frame)
        self.range_end_spin.setMaximum(last_frame)
        self.range_end_spin.setValue(last_frame)
        self.window_spin.setMaximum(max(10, total_frames))
        self.canvas.set_total_frames(total_frames)
        self._update_scroll_range()

    def set_propagation_running(self, running: bool) -> None:
        if running:
            self.propagate_button.setEnabled(False)
            self.overwrite_cb.setEnabled(False)
            self.only_low_iou_cb.setEnabled(False)
            self.range_start_spin.setEnabled(False)
            self.range_end_spin.setEnabled(False)
            self.propagate_button.setText('Propagation running...')
        else:
            self.propagate_button.setEnabled(True)
            self.overwrite_cb.setEnabled(True)
            self._update_inference_scope_controls()
            self.propagate_button.setText('Propagate Bidirectionally from annotations')

    def _update_inference_scope_controls(self) -> None:
        has_iou = self._data is not None and bool(self._data.get('frame_indices'))
        self.only_low_iou_cb.setEnabled(has_iou)
        self.threshold_spin.setEnabled(has_iou)
        if not has_iou:
            self.only_low_iou_cb.setChecked(False)
        self.range_start_spin.setEnabled(True)
        self.range_end_spin.setEnabled(True)

    def set_seed_frames(self, seed_frames: List[int]) -> None:
        self.canvas.set_seed_frames(seed_frames)
        has_plot = self._has_plot_content()
        self.window_spin.setEnabled(has_plot)
        self.scroll_slider.setEnabled(has_plot)
        self._update_scroll_range()

    def load_data(self, data: Optional[Dict]) -> None:
        self._data = data
        self.canvas.set_data(data)
        self._update_scroll_range()
        has_plot = self._has_plot_content()
        self.window_spin.setEnabled(has_plot)
        self.scroll_slider.setEnabled(has_plot)
        self._update_inference_scope_controls()

    def _has_plot_content(self) -> bool:
        has_iou = self._data is not None and bool(self._data.get('frame_indices'))
        return has_iou or self._total_frames > 1 or bool(self.canvas._seed_frames)

    def set_current_frame(self, frame_idx: int, auto_scroll: bool = True) -> None:
        if auto_scroll and self._total_frames > 0:
            window = self.window_spin.value()
            max_start = max(0, self._total_frames - 1 - window + 1)
            if frame_idx < self.scroll_slider.value():
                new_start = max(0, frame_idx - window // 4)
                self._set_scroll_start(min(new_start, max_start))
            elif frame_idx > self.scroll_slider.value() + window - 1:
                new_start = max(0, frame_idx - window + window // 4)
                self._set_scroll_start(min(new_start, max_start))
        self.canvas.set_current_frame(frame_idx)

    def set_enabled_controls(self, enabled: bool) -> None:
        has_plot = enabled and self._has_plot_content()
        self.window_spin.setEnabled(has_plot)
        self.scroll_slider.setEnabled(has_plot)
        if enabled:
            self._update_inference_scope_controls()
        else:
            self.only_low_iou_cb.setEnabled(False)
            self.threshold_spin.setEnabled(False)
            self.range_start_spin.setEnabled(False)
            self.range_end_spin.setEnabled(False)

    def _update_scroll_range(self) -> None:
        last_frame = max(0, self._total_frames - 1)
        window = self.window_spin.value()
        max_start = max(0, last_frame - window + 1)
        self._updating_scroll = True
        self.scroll_slider.setMaximum(max_start)
        self.scroll_slider.setValue(min(self.scroll_slider.value(), max_start))
        self._updating_scroll = False
        self.canvas.set_scroll_start(self.scroll_slider.value())

    def _set_scroll_start(self, start: int) -> None:
        self._updating_scroll = True
        self.scroll_slider.setValue(start)
        self._updating_scroll = False
        self.canvas.set_scroll_start(start)

    def _on_window_changed(self, value: int) -> None:
        self.canvas.set_window_frames(value)
        self._update_scroll_range()

    def _on_threshold_changed(self, value: float) -> None:
        self.canvas.set_threshold(value)

    def _on_scroll_changed(self, value: int) -> None:
        if self._updating_scroll:
            return
        self.canvas.set_scroll_start(value)
