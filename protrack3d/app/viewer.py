"""Viewer: scroll through z and time with the segmentation mask or the tracked object IDs drawn over the image."""
import csv
import os

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSettings, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QSlider, QVBoxLayout, QWidget

from .. import stages
from ..segmentation import image_base, read_image_info, segmentation_folder
from . import icons
from .contrast import ContrastPanel, Levels
from .pages import IMAGE_FILTERS, Page
from .palette import id_colours
from .widgets import Card, PathField, button, text


VIEWER_FILTERS = 'Images (*.tif *.tiff *.nii *.czi);;All files (*)'


def _tif_axes(path):
    import tifffile
    with tifffile.TiffFile(path) as tif:
        return tif.series[0].axes


class Stack:
    """Lazy access to an image and, when they exist next to it, its mask and tracked labels.

    Movies (.tif, .nii) have one channel and are read from disk as they are shown. Files with several channels
    (.czi, or a .tif with a channel axis) are held in memory and shown one channel at a time.
    """

    def __init__(self, path):
        self.path, self.channel, self.channel_names, self.notes = path, 0, [''], []
        self._nii = self._tif = self._multi = None
        low = path.lower()
        if low.endswith('.czi') or (low.endswith(('.tif', '.tiff')) and 'C' in _tif_axes(path)):
            from ..czi import read_czi, read_multichannel_tif
            self._multi, self.channel_names, self.notes = (read_czi if low.endswith('.czi') else read_multichannel_tif)(path)
            t, c, z, y, x = self._multi.shape
            self.info = {'path': path, 'x': x, 'y': y, 'z': z, 't': t}
        else:
            self.info = read_image_info(path)
            if low.endswith('.nii'):
                import nibabel as nib
                self._nii = nib.load(path).dataobj
            else:
                import tifffile
                try:
                    data = tifffile.memmap(path, mode='r')
                except Exception:
                    data = tifffile.imread(path)
                self._tif = data if data.ndim == 4 else data[None]
        self.seg_start, self.mask, self.mask_tif, self.tracks = 1, None, None, None
        seg = segmentation_folder(path)
        try:
            with open(os.path.join(seg, 'segmentation_parameters.csv')) as f:
                self.seg_start = int(next(csv.DictReader(f))['startTime'])
        except (OSError, StopIteration, KeyError, ValueError):
            pass
        import nibabel as nib
        mask = os.path.join(seg, 'CombinedSO', 'CombinedSO.nii')
        if os.path.isfile(mask):
            self.mask = nib.load(mask).dataobj
        else:  # a mask saved as a .tif next to the image: <name>_SEG.tif, or <name>_ARM_SEG.tif from the Armadillo workflow
            import tifffile
            stem = os.path.splitext(path)[0]
            for candidate in (stem + '_SEG.tif', stem + '_ARM_SEG.tif'):
                if os.path.isfile(candidate):
                    data = tifffile.imread(candidate)
                    data = data if data.ndim == 4 else data[None]
                    if data.shape[2:] == (self.info['y'], self.info['x']):
                        self.mask_tif, self.mask_name = data, os.path.basename(candidate)
                        break
        self.tracking_folder = stages.tracking_folder(path, seg)
        tracked = os.path.join(self.tracking_folder, 'TrackedCombined.nii')
        if os.path.isfile(tracked):
            self.tracks = nib.load(tracked).dataobj
        self._levels = {}

    @property
    def channels(self):
        return len(self.channel_names)

    def levels(self):
        """Display range of the current channel. It starts fitted to the middle plane and is the user's to change."""
        if self.channel not in self._levels:
            t, z = self.info['t'], self.info['z']
            mid = self.plane(t // 2, z // 2)
            whole = np.issubdtype(mid.dtype, np.integer)
            if mid.dtype == np.uint8:
                data_min, data_max = 0, 255
            else:  # the range of the data, from the first, middle and last timepoints; render() widens it if needed
                planes = [self.plane(i, k) for i in sorted({0, t // 2, t - 1}) for k in range(z)]
                data_min, data_max = min(float(p.min()) for p in planes), max(float(p.max()) for p in planes)
            low = float(np.percentile(mid, 1))
            self._levels[self.channel] = Levels(data_min, data_max, low, max(float(np.percentile(mid, 99.9)), low + 1), whole)
        return self._levels[self.channel]

    def plane(self, t, z):
        """Image plane as (y, x)."""
        if self._multi is not None:
            return self._multi[t, self.channel, z]
        if self._tif is not None:
            return np.asarray(self._tif[t, z])
        vol = self._nii
        return np.asarray(vol[:, :, z, t] if len(vol.shape) == 4 else vol[:, :, z]).T

    def time_of(self, tracked_timepoint):
        """Position on the time slider of a timepoint counted within the tracked range (1 is the first)."""
        return self.seg_start - 1 + tracked_timepoint - 1

    def best_slice(self, obj, t):
        """The z slice where an object has most voxels at slider time t, or None when it is not there."""
        i = t - (self.seg_start - 1)
        if self.tracks is None or len(self.tracks.shape) != 4 or not 0 <= i < self.tracks.shape[3]:
            return None
        counts = (np.asarray(self.tracks[..., i]) == obj).sum(axis=(0, 1))
        return int(counts.argmax()) if counts.any() else None

    def overlay(self, kind, t, z):
        """Mask or label plane as (y, x), or None when that timepoint has none."""
        if kind == 'mask' and self.mask is None and self.mask_tif is not None:
            m = self.mask_tif  # (t, z, y, x); the Armadillo mask covers only the first slices
            return np.asarray(m[t, z]) if t < m.shape[0] and z < m.shape[1] else None
        data = self.mask if kind == 'mask' else self.tracks
        i = t - (self.seg_start - 1)
        if data is None or i < 0:
            return None
        if len(data.shape) == 3:
            return np.asarray(data[:, :, z]).T if i == 0 else None
        if i >= data.shape[3]:
            return None
        return np.asarray(data[:, :, z, i]).T


class Canvas(QWidget):
    hovered = Signal(int, int)
    clicked = Signal(int, int)
    scrolled = Signal(int)

    def __init__(self):
        super().__init__()
        self.setObjectName('viewer')
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.setMinimumHeight(320)
        self.pixmap, self.target, self.marker = None, QRectF(), None  # marker: (x, y, radius) in image pixels

    def show_rgb(self, rgb):
        h, w, _ = rgb.shape
        self.pixmap = QPixmap.fromImage(QImage(np.ascontiguousarray(rgb).data, w, h, 3 * w, QImage.Format_RGB888).copy())
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.pixmap:
            return
        painter = QPainter(self)
        margin = 12
        scale = min((self.width() - 2 * margin) / self.pixmap.width(), (self.height() - 2 * margin) / self.pixmap.height())
        w, h = self.pixmap.width() * scale, self.pixmap.height() * scale
        self.target = QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, scale < 1)
        painter.drawPixmap(self.target, self.pixmap, QRectF(self.pixmap.rect()))
        if self.marker:  # a ring around the picked object, so a few voxels are easy to find
            x, y, r = self.marker
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QPen(QColor(255, 255, 255, 230), 1.6))
            painter.drawEllipse(QPointF(self.target.x() + (x + 0.5) * scale, self.target.y() + (y + 0.5) * scale), r * scale, r * scale)

    def wheelEvent(self, event):
        self.scrolled.emit(1 if event.angleDelta().y() > 0 else -1)

    def _pixel(self, pos):
        if not self.pixmap or not self.target.contains(pos):
            return None
        return (int((pos.x() - self.target.x()) / self.target.width() * self.pixmap.width()),
                int((pos.y() - self.target.y()) / self.target.height() * self.pixmap.height()))

    def mouseMoveEvent(self, event):
        at = self._pixel(event.position())
        if at:
            self.hovered.emit(*at)

    def mouseReleaseEvent(self, event):
        at = self._pixel(event.position())
        if at:
            self.clicked.emit(*at)


class ViewerPage(Page):
    OVERLAYS = [('Image only', None), ('Segmentation mask', 'mask'), ('Tracked objects', 'tracks')]

    def __init__(self, ctx):
        super().__init__(ctx, 'Viewer', None)
        self.stack, self.colours = None, id_colours()
        self.highlight, self.lineage, self.focus = set(), None, None
        self.source = PathField(ctx.theme, 'file', 'Drop an image here (.tif, .nii or .czi), or browse. Its mask and tracks are picked up when they are next to it.', VIEWER_FILTERS)
        self.source.changed.connect(self.open)
        self.body.addWidget(self.source)
        self.canvas = Canvas()
        self.canvas.hovered.connect(self.hover)
        self.canvas.clicked.connect(self.click)
        self.canvas.scrolled.connect(lambda step: self.z_slider.setValue(self.z_slider.value() + step))
        self.contrast = ContrastPanel(ctx.theme)
        self.contrast.changed.connect(self.render)
        middle = QHBoxLayout()
        middle.setSpacing(16)
        middle.addWidget(self.canvas, 1)
        middle.addWidget(self.contrast)
        self.body.addLayout(middle, 1)

        card = Card(margins=14, spacing=10)
        self.t_slider, self.t_label = self._slider('Timepoint')
        self.z_slider, self.z_label = self._slider('Slice')
        self.play = button('', tip='Play through time')
        self.play.setCheckable(True)
        self.play.toggled.connect(self.toggle_play)
        self.overlay = QComboBox()
        for label, _ in self.OVERLAYS:
            self.overlay.addItem(label)
        self.overlay.currentIndexChanged.connect(self.render)
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(10, 100)
        self.opacity.setValue(55)
        self.opacity.setFixedWidth(110)
        self.opacity.valueChanged.connect(self.render)
        for label, slider, extra in ((self.t_label, self.t_slider, [self.play]), (self.z_label, self.z_slider, [])):
            row = QHBoxLayout()
            label.setFixedWidth(150)
            row.addWidget(label)
            row.addWidget(slider, 1)
            for w in extra:
                row.addWidget(w)
            card.box.addLayout(row)
        row = QHBoxLayout()
        self.channel_label = text('Channel', 'label')
        self.channel_box = QComboBox()
        self.channel_box.currentIndexChanged.connect(self.set_channel)
        for wdg in (self.channel_label, self.channel_box):
            wdg.hide()
            row.addWidget(wdg)
        row.addWidget(text('Show', 'label'))
        row.addWidget(self.overlay)
        row.addWidget(text('Opacity', 'label'))
        row.addWidget(self.opacity)
        self.focus_label = text('', 'label')
        self.lineage_button = button('Show in lineage', 'ghost')
        self.lineage_button.clicked.connect(self.to_lineage)
        self.clear_button = button('Show all objects', 'ghost')
        self.clear_button.clicked.connect(lambda: self.set_highlight(None))
        for w in (self.focus_label, self.lineage_button, self.clear_button):
            w.hide()
            row.addWidget(w)
        row.addStretch(1)
        self.readout = text('', 'hint')
        row.addWidget(self.readout)
        self.contrast_button = button('Brightness && contrast', 'ghost', tip='Show or hide the brightness and contrast panel')
        self.contrast_button.setCheckable(True)
        self.contrast_button.setChecked(QSettings().value('viewer_contrast', 'true') == 'true')
        self.contrast_button.toggled.connect(self.show_contrast)
        self.contrast.setVisible(self.contrast_button.isChecked())
        row.addWidget(self.contrast_button)
        card.box.addLayout(row)
        self.body.addWidget(card)
        self.timer = QTimer(self)
        self.timer.setInterval(120)
        self.timer.timeout.connect(self.advance)
        ctx.theme.listeners.append(self.restyle)
        self.restyle()

    def _slider(self, name):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 0)
        slider.valueChanged.connect(self.render)
        return slider, text(name, 'label')

    def show_contrast(self, shown):
        QSettings().setValue('viewer_contrast', 'true' if shown else 'false')
        self.contrast.setVisible(shown)

    def restyle(self):
        self.play.setIcon(icons.icon('pause' if self.play.isChecked() else 'play', self.ctx.theme.colors['text'], 16))
        self.render()

    def open(self, path):
        self.stack, self.lineage = None, None
        self.set_highlight(None)
        self.timer.stop()
        self.play.setChecked(False)
        if path:
            try:
                self.stack = Stack(path)
            except Exception as e:
                self.source.set_detail(str(e))
        if not self.stack:
            self.canvas.pixmap = None
            self.canvas.update()
            self.contrast.display(None, None, None)
            return
        info, s = self.stack.info, self.stack
        found = [n for n, d in (('mask', s.mask if s.mask is not None else s.mask_tif), ('tracks', s.tracks)) if d is not None]
        parts = [f"{info['x']} × {info['y']} × {info['z']} voxels", f"{info['t']} timepoint{'s' if info['t'] != 1 else ''}"]
        if s.channels > 1:
            parts.append(f'{s.channels} channels')
        parts.append('found ' + ' and '.join(found) if found else 'no mask or tracks next to this image')
        self.source.set_detail('  ·  '.join(parts + s.notes))
        self.channel_box.blockSignals(True)
        self.channel_box.clear()
        for i, name in enumerate(s.channel_names):
            self.channel_box.addItem(f'{i + 1}  ·  {name}' if name else str(i + 1))
        # the Armadillo mask belongs to the second channel, so start there when that mask was found
        s.channel = 1 if s.channels > 1 and s.mask is None and s.mask_tif is not None and s.mask_name.endswith('_ARM_SEG.tif') else 0
        self.channel_box.setCurrentIndex(s.channel)
        self.channel_box.blockSignals(False)
        for wdg in (self.channel_label, self.channel_box):
            wdg.setVisible(s.channels > 1)
        for slider, n in ((self.t_slider, info['t']), (self.z_slider, info['z'])):
            slider.blockSignals(True)
            slider.setRange(0, n - 1)
            slider.blockSignals(False)
        self.t_slider.setValue(s.seg_start - 1)
        self.z_slider.setValue(info['z'] // 2)
        self.overlay.setCurrentIndex(2 if s.tracks is not None else 1 if (s.mask is not None or s.mask_tif is not None) else 0)
        self.render()

    def set_channel(self, index):
        if self.stack and 0 <= index < self.stack.channels:
            self.stack.channel = index
            self.render()

    def show_image(self, path):
        self.source.set_path(path)

    def toggle_play(self, playing):
        self.timer.start() if playing and self.stack else self.timer.stop()
        self.restyle()

    def advance(self):
        self.t_slider.setValue((self.t_slider.value() + 1) % (self.t_slider.maximum() + 1))

    def render(self):
        if not self.stack:
            return
        t, z, s = self.t_slider.value(), self.z_slider.value(), self.stack
        self.t_label.setText(f"Timepoint {t + 1} of {s.info['t']}")
        self.z_label.setText(f"Slice {z + 1} of {s.info['z']}")
        plane = s.plane(t, z).astype(np.float32)
        levels = s.levels()
        levels.include(plane.min(), plane.max())
        self.contrast.display(levels, plane, (id(s), s.channel, t, z, levels.data_min, levels.data_max))
        rgb = np.repeat((levels.scale(plane) * 255).astype(np.uint8)[..., None], 3, axis=2)
        kind = self.OVERLAYS[self.overlay.currentIndex()][1]
        self.labels = s.overlay(kind, t, z) if kind else None
        if self.labels is not None and self.labels.shape == plane.shape:
            on = self.labels > 0
            a = self.opacity.value() / 100
            if kind == 'mask':
                accent = self.ctx.theme.colors['accent'].lstrip('#')
                colour = np.array([int(accent[i:i + 2], 16) for i in (0, 2, 4)], np.float32)
                rgb[on] = (rgb[on] * (1 - a) + colour * a).astype(np.uint8)
            else:
                ids = self.labels[on].astype(np.int64)
                colour = self.colours[ids % len(self.colours)].astype(np.float32)
                # with a family picked out, its members are drawn solid and everything else is faint
                alpha = np.where(np.isin(ids, list(self.highlight)), 0.9, 0.18)[:, None] if self.highlight else a
                rgb[on] = (rgb[on] * (1 - alpha) + colour * alpha).astype(np.uint8)
        self.plane_shown = plane
        self.canvas.marker = None
        if self.focus is not None and kind == 'tracks' and self.labels is not None:
            ys, xs = np.nonzero(self.labels == self.focus)
            if len(xs):
                radius = max(7.0, 1.6 * max(np.ptp(xs), np.ptp(ys)) / 2 + 4)
                self.canvas.marker = (float(xs.mean()), float(ys.mean()), radius)
        self.canvas.show_rgb(rgb)

    def hover(self, x, y):
        if not self.stack or not (0 <= y < self.plane_shown.shape[0] and 0 <= x < self.plane_shown.shape[1]):
            return
        parts = [f'x {x}, y {y}', f'intensity {int(self.plane_shown[y, x])}']
        if getattr(self, 'labels', None) is not None and self.labels.shape == self.plane_shown.shape and self.labels[y, x]:
            kind = self.OVERLAYS[self.overlay.currentIndex()][1]
            parts.append(f'object {int(self.labels[y, x])}' if kind == 'tracks' else 'inside the mask')
        self.readout.setText('   ·   '.join(parts))

    # --- picking objects ---
    def tracking_lineage(self):
        if self.lineage is None and self.stack and self.stack.tracks is not None:
            from ..lineage import Lineage
            try:
                self.lineage = Lineage(self.stack.tracking_folder)
            except Exception:
                self.lineage = False
        return self.lineage or None

    def set_highlight(self, obj):
        lineage = self.tracking_lineage() if obj is not None else None
        self.focus = obj if lineage and obj in lineage.objects else None
        self.highlight = set(lineage.members(obj)) if self.focus is not None else set()
        if self.focus is not None:
            n = len(self.highlight)
            self.focus_label.setText(f'Object {obj}' + (f' and its family ({n} objects)' if n > 1 else ', no relatives'))
        for w in (self.focus_label, self.lineage_button, self.clear_button):
            w.setVisible(self.focus is not None)
        self.render()

    def click(self, x, y):
        """Clicking a tracked object picks out its family; clicking the background shows everything again."""
        if self.stack is None or self.OVERLAYS[self.overlay.currentIndex()][1] != 'tracks' or getattr(self, 'labels', None) is None:
            return
        if 0 <= y < self.labels.shape[0] and 0 <= x < self.labels.shape[1]:
            obj = int(self.labels[y, x])
            self.set_highlight(obj if obj else None)

    def show_object(self, image, lineage, obj):
        """Open `image` at the first timepoint of `obj`, on the slice where it is largest, with its family picked out."""
        if os.path.normpath(image) != os.path.normpath(self.source.path or '.'):
            self.source.set_path(image)
        if not self.stack:
            return
        self.lineage = lineage
        t = self.stack.time_of(lineage.objects[obj][0])
        self.overlay.setCurrentIndex(2 if self.stack.tracks is not None else self.overlay.currentIndex())
        self.t_slider.setValue(t)
        z = self.stack.best_slice(obj, t)
        if z is not None:
            self.z_slider.setValue(z)
        self.set_highlight(obj)

    def to_lineage(self):
        if self.focus is not None and self.stack:
            self.ctx.show_lineage(self.stack.tracking_folder, self.focus)
