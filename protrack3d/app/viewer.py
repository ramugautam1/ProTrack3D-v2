"""Viewer: scroll through z and time with the segmentation mask or the tracked object IDs drawn over the image."""
import csv
import gc
import os

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, QSettings, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QSlider, QVBoxLayout, QWidget

from .. import stages
from ..segmentation import image_base, read_image_info, segmentation_folder
from . import icons
from .contrast import ContrastPanel, Levels, SaveDialog
from .jobs import Job
from .pages import IMAGE_FILTERS, Page
from .palette import id_colours
from .widgets import Card, ElidedLabel, FlowLayout, PathField, button, text


def rename_active(widget, active):
    """Switch a widget's 'active' look (a file is being dragged over it) on or off."""
    widget.setProperty('active', 'true' if active else 'false')
    widget.style().unpolish(widget)
    widget.style().polish(widget)


VIEWER_FILTERS = 'Images (*.tif *.tiff *.nii *.czi);;All files (*)'


def _tif_axes(path):
    import tifffile
    with tifffile.TiffFile(path) as tif:
        return tif.series[0].axes


class Stack:
    """Lazy access to an image and, when they exist next to it, its mask and tracked labels.

    Movies (.tif, .nii) have one channel and are read from disk as they are shown. Files with several channels
    (.czi, or a .tif with a channel axis) are held in memory; any of their channels can be shown, alone or together.
    `channel` is the one the brightness and contrast panel adjusts.
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
        self.luts = {}  # channel -> colour table name; Auto when not chosen

    @property
    def channels(self):
        return len(self.channel_names)

    def all_levels(self):
        """The display range of every channel, as each is (or would be) shown."""
        return [self.levels(c) for c in range(self.channels)]

    def levels(self, channel=None):
        """Display range of a channel (default: the one being adjusted). It starts fitted to the middle plane and is
        the user's to change."""
        channel = self.channel if channel is None else channel
        if channel not in self._levels:
            t, z = self.info['t'], self.info['z']
            mid = self.plane(t // 2, z // 2, channel)
            whole = np.issubdtype(mid.dtype, np.integer)
            if mid.dtype == np.uint8:
                data_min, data_max = 0, 255
            else:  # the range of the data, from the first, middle and last timepoints; render() widens it if needed
                planes = [self.plane(i, k, channel) for i in sorted({0, t // 2, t - 1}) for k in range(z)]
                data_min, data_max = min(float(p.min()) for p in planes), max(float(p.max()) for p in planes)
            low = float(np.percentile(mid, 1))
            self._levels[channel] = Levels(data_min, data_max, low, max(float(np.percentile(mid, 99.9)), low + 1), whole)
        return self._levels[channel]

    def plane(self, t, z, channel=None):
        """Image plane as (y, x), of `channel` (default: the one being adjusted)."""
        if self._multi is not None:
            return self._multi[t, self.channel if channel is None else channel, z]
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
    """Shows one plane. Ctrl-scroll or pinch zooms around the pointer, dragging pans, a double-click fits the image
    again; a plain scroll is passed on (it moves through the slices)."""
    hovered = Signal(int, int)
    clicked = Signal(int, int)
    scrolled = Signal(int)
    MARGIN, MOST = 12, 40  # MOST: screen pixels per image pixel at the closest zoom

    def __init__(self, theme):
        super().__init__()
        self.theme = theme
        self.setObjectName('viewer')
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.setMinimumHeight(320)
        self.pixmap, self.target, self.marker = None, QRectF(), None  # marker: (x, y, radius) in image pixels
        self.zoom, self.centre, self._drag = 1.0, None, None  # zoom 1 fits the image; centre is in image pixels
        self.prompt = ''
        self.buttons = []
        for label, tip, action in (('+', 'Zoom in (Ctrl-scroll or pinch)', lambda: self.zoom_by(1.5)),
                                   ('−', 'Zoom out', lambda: self.zoom_by(1 / 1.5)),
                                   ('Fit', 'Show the whole image (double-click)', self.fit)):
            b = button(label, 'viewerButton', tip=tip)
            b.setParent(self)
            b.setFixedSize(44 if label == 'Fit' else 30, 28)
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(action)
            b.hide()
            self.buttons.append(b)
        self.level = text('', 'viewerLevel')
        self.level.setParent(self)
        self.level.hide()

    # --- the view ---
    def _fit_scale(self):
        m = 2 * self.MARGIN
        return max(1e-6, min((self.width() - m) / self.pixmap.width(), (self.height() - m) / self.pixmap.height()))

    def _scale(self):
        return self._fit_scale() * self.zoom

    def _place(self):
        """Where the whole image lies in the widget at the current zoom; the view is kept inside the image."""
        scale, view = self._scale(), self.view()
        w, h = self.pixmap.width() * scale, self.pixmap.height() * scale
        cx, cy = self.centre if self.centre else (self.pixmap.width() / 2, self.pixmap.height() / 2)
        if w <= view.width():
            cx = self.pixmap.width() / 2
        else:
            cx = min(max(cx, view.width() / 2 / scale), self.pixmap.width() - view.width() / 2 / scale)
        if h <= view.height():
            cy = self.pixmap.height() / 2
        else:
            cy = min(max(cy, view.height() / 2 / scale), self.pixmap.height() - view.height() / 2 / scale)
        self.centre = (cx, cy)
        self.target = QRectF(view.center().x() - cx * scale, view.center().y() - cy * scale, w, h)

    def view(self):
        return QRectF(self.rect()).adjusted(self.MARGIN, self.MARGIN, -self.MARGIN, -self.MARGIN)

    def fit(self):
        self.zoom, self.centre = 1.0, None
        self._changed()

    def zoom_by(self, factor, at=None):
        """Zoom, keeping the image point under `at` (a widget position; default: the middle) where it is."""
        if not self.pixmap:
            return
        self._place()
        at = at or self.view().center()
        old = self._scale()
        most = max(1.0, self.MOST / self._fit_scale())
        self.zoom = min(max(self.zoom * factor, 1.0), most)
        new = self._scale()
        px, py = (at.x() - self.target.x()) / old, (at.y() - self.target.y()) / old  # image point under the pointer
        centre = self.view().center()
        self.centre = (px + (centre.x() - at.x()) / new, py + (centre.y() - at.y()) / new)
        self._changed()

    def centre_on(self, x, y):
        self.centre = (x, y)
        self._changed()

    def shows(self, x, y):
        """Is this image point inside what is on screen?"""
        self._place()
        scale = self._scale()
        return self.view().contains(QPointF(self.target.x() + x * scale, self.target.y() + y * scale))

    def _changed(self):
        zoomed = self.zoom > 1.001
        self.level.setText(f'{self._scale() * 100:.0f}%' if self.pixmap else '')
        self.level.adjustSize()
        self.setCursor(Qt.OpenHandCursor if zoomed else Qt.ArrowCursor)
        self._arrange()
        self.update()

    def _arrange(self):
        x = self.width() - self.MARGIN - 8
        for b in reversed(self.buttons):
            x -= b.width()
            b.move(x, self.MARGIN + 8)
            b.setVisible(self.pixmap is not None)
            x -= 4
        self.level.move(self.MARGIN + 10, self.height() - self.MARGIN - 8 - self.level.height())
        self.level.setVisible(self.pixmap is not None)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._arrange()

    def clear(self):
        self.pixmap, self.marker = None, None
        self.fit()

    def show_rgb(self, rgb):
        h, w, _ = rgb.shape
        fresh = self.pixmap is None or (self.pixmap.width(), self.pixmap.height()) != (w, h)
        self.pixmap = QPixmap.fromImage(QImage(np.ascontiguousarray(rgb).data, w, h, 3 * w, QImage.Format_RGB888).copy())
        if fresh:  # another image: start from the whole picture
            self.zoom, self.centre = 1.0, None
        self._changed()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        if not self.pixmap:
            painter.setPen(QColor(self.theme.colors['faint']))
            painter.drawText(self.rect(), Qt.AlignCenter, self.prompt)
            return
        self._place()
        scale, view = self._scale(), self.view()
        painter.setClipRect(view)
        # only the part of the image that is on screen is drawn, in whole image pixels
        x0, y0 = max(0, int((view.left() - self.target.x()) / scale)), max(0, int((view.top() - self.target.y()) / scale))
        x1 = min(self.pixmap.width(), int((view.right() - self.target.x()) / scale) + 1)
        y1 = min(self.pixmap.height(), int((view.bottom() - self.target.y()) / scale) + 1)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, scale < 1)
        painter.drawPixmap(QRectF(self.target.x() + x0 * scale, self.target.y() + y0 * scale, (x1 - x0) * scale, (y1 - y0) * scale),
                           self.pixmap, QRectF(x0, y0, x1 - x0, y1 - y0))
        if self.marker:  # a ring around the picked object, so a few voxels are easy to find
            x, y, r = self.marker
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QPen(QColor(255, 255, 255, 230), 1.6))
            painter.drawEllipse(QPointF(self.target.x() + (x + 0.5) * scale, self.target.y() + (y + 0.5) * scale), r * scale, r * scale)

    # --- the mouse ---
    def wheelEvent(self, event):
        if event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
            self.zoom_by(1.2 if event.angleDelta().y() > 0 else 1 / 1.2, event.position())
        elif event.angleDelta().y():
            self.scrolled.emit(1 if event.angleDelta().y() > 0 else -1)

    def event(self, event):
        if event.type() == QEvent.NativeGesture and event.gestureType() == Qt.ZoomNativeGesture:  # trackpad pinch
            self.zoom_by(1 + event.value(), event.position())
            return True
        return super().event(event)

    def _pixel(self, pos):
        if not self.pixmap or not self.target.contains(pos) or not self.view().contains(pos):
            return None
        return (min(self.pixmap.width() - 1, int((pos.x() - self.target.x()) / self.target.width() * self.pixmap.width())),
                min(self.pixmap.height() - 1, int((pos.y() - self.target.y()) / self.target.height() * self.pixmap.height())))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.pixmap:
            self._drag = (event.position(), self.centre, False)  # where it began, the centre then, whether it moved

    def mouseMoveEvent(self, event):
        if self._drag:
            began, centre, moved = self._drag
            delta = event.position() - began
            if moved or abs(delta.x()) + abs(delta.y()) > 4:
                self._drag = (began, centre, True)
                if self.zoom > 1.001 and centre:
                    self.setCursor(Qt.ClosedHandCursor)
                    self.centre = (centre[0] - delta.x() / self._scale(), centre[1] - delta.y() / self._scale())
                    self.update()
                return
        at = self._pixel(event.position())
        if at:
            self.hovered.emit(*at)

    def mouseReleaseEvent(self, event):
        drag, self._drag = self._drag, None
        self.setCursor(Qt.OpenHandCursor if self.zoom > 1.001 else Qt.ArrowCursor)
        if drag and drag[2]:
            return  # that was a drag, not a click
        at = self._pixel(event.position())
        if at and event.button() == Qt.LeftButton:
            self.clicked.emit(*at)

    def mouseDoubleClickEvent(self, event):
        self.fit()


class ViewerPage(Page):
    NAME_LENGTH = 16  # channel names are shortened on their checkboxes so a row of them fits; the tooltip has the rest

    def __init__(self, ctx):
        super().__init__(ctx, 'Viewer', None)
        self.stack, self.colours = None, id_colours()
        self.highlight, self.lineage, self.focus = set(), None, None
        self.source = PathField(ctx.theme, 'file', 'Drop an image here (.tif, .nii or .czi), or browse. Its mask and tracks are picked up when they are next to it.', VIEWER_FILTERS)
        self.source.changed.connect(self.open)
        self.body.addWidget(self.source)
        self.canvas = Canvas(ctx.theme)
        self.canvas.prompt = 'Drop an image here'
        self.setAcceptDrops(True)  # anywhere on the page, the picture included
        self.canvas.hovered.connect(self.hover)
        self.canvas.clicked.connect(self.click)
        self.canvas.scrolled.connect(lambda step: self.z_slider.setValue(self.z_slider.value() + step))
        self.contrast = ContrastPanel(ctx.theme)
        self.contrast.changed.connect(self.render)
        self.contrast.save_requested.connect(self.save_adjusted)
        self.contrast.channel_chosen.connect(self.adjust_channel)
        self.contrast.lut_chosen.connect(self.choose_lut)
        self.reopen = None  # (job, image) while the image on screen is being replaced
        ctx.jobs.finished.connect(self.save_finished)
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
        self.layers = QWidget()  # a checkbox per channel, and for the mask and the tracks when they were found
        self.layer_row = FlowLayout(self.layers, spacing=18)  # wraps onto a second line when there are many
        self.layers.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.channel_boxes, self.mask_box, self.tracks_box = [], None, None
        row = QHBoxLayout()
        show = text('Show', 'label')
        show.setFixedWidth(150)
        row.addWidget(show, 0, Qt.AlignTop)
        row.addWidget(self.layers, 1)
        card.box.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(text('Overlay opacity', 'label'))
        row.addWidget(self.opacity)
        self.readout = ElidedLabel('', 'hint')  # takes the room that is left, so a long readout cannot widen the page
        row.addWidget(self.readout, 1)
        self.contrast_button = button('Brightness && contrast', 'ghost', tip='Show or hide the brightness and contrast panel')
        self.contrast_button.setCheckable(True)
        self.contrast_button.setChecked(QSettings().value('viewer_contrast', 'true') == 'true')
        self.contrast_button.toggled.connect(self.show_contrast)
        self.contrast.setVisible(self.contrast_button.isChecked())
        row.addWidget(self.contrast_button)
        card.box.addLayout(row)
        self.picked = QWidget()  # a line of its own, shown while an object is picked out
        row = QHBoxLayout(self.picked)
        row.setContentsMargins(0, 0, 0, 0)
        self.focus_label = text('', 'label')
        self.lineage_button = button('Show in lineage', 'ghost')
        self.lineage_button.clicked.connect(self.to_lineage)
        self.clear_button = button('Show all objects', 'ghost')
        self.clear_button.clicked.connect(lambda: self.set_highlight(None))
        for w in (self.focus_label, self.lineage_button, self.clear_button):
            row.addWidget(w)
        row.addStretch(1)
        self.picked.hide()
        card.box.addWidget(self.picked)
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

    def save_adjusted(self):
        """Save the image with its brightness and contrast applied, as a copy or in place of the original."""
        if not self.stack or self.reopen:
            return
        image, levels = self.stack.path, self.stack.all_levels()
        whole = self.stack.plane(0, 0).dtype
        top = float(np.iinfo(whole).max) if np.issubdtype(whole, np.integer) else 1.0
        ranges = [(lv.low, lv.high) for lv in levels]
        problem = None
        if image.lower().endswith('.czi'):
            from ..rescale import czi_problem
            problem = czi_problem(image)
        # parented to the window, not this page: the page's scroll-area style would strip the dialog's borders
        dialog = SaveDialog(self.window(), image, ranges, top, problem)
        if not dialog.exec():
            return
        args = ['rescale', image, '--out', dialog.target]
        for low, high in ranges:
            args += ['--range', repr(float(low)), repr(float(high))]
        job = Job(f'Save image  ·  {os.path.basename(dialog.target)}', args, ['saving image'], folder=os.path.dirname(dialog.target))
        if dialog.target == image:  # the file must not be open here while it is replaced
            self.reopen = (job, image)
            self.stack = None
            gc.collect()
            self.canvas.clear()
            self.contrast.display(None, None, None)
            self.source.set_detail('Being replaced…')
        self.ctx.submit(job, announce=False)
        self.ctx.toast('Saving ' + os.path.basename(dialog.target))

    def save_finished(self, job):
        if not self.reopen or job is not self.reopen[0]:
            return
        image, self.reopen = self.reopen[1], None
        if os.path.normpath(self.source.path or '.') != os.path.normpath(image):
            return  # another image was opened in the meantime
        self.open(image)
        if job.state == 'done' and self.stack:  # the saved values already hold the contrast: show them as they are
            for levels in self.stack.all_levels():
                levels.reset()
            self.render()

    def _dropped(self, event):
        urls = event.mimeData().urls()
        path = urls[0].toLocalFile() if len(urls) == 1 and urls[0].isLocalFile() else ''
        return path if os.path.isfile(path) else None

    def dragEnterEvent(self, event):
        if self._dropped(event):
            event.acceptProposedAction()
            rename_active(self.canvas, True)

    def dragLeaveEvent(self, event):
        rename_active(self.canvas, False)

    def dropEvent(self, event):
        path = self._dropped(event)
        rename_active(self.canvas, False)
        if path:
            event.acceptProposedAction()
            self.source.set_path(path)

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
            self.canvas.clear()
            self.contrast.display(None, None, None)
            self.contrast.set_channels([], [], 0)
            while self.layer_row.count():
                item = self.layer_row.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.hide()
                    widget.setParent(None)
                    widget.deleteLater()
            self.channel_boxes, self.mask_box, self.tracks_box = [], None, None
            return
        info, s = self.stack.info, self.stack
        found = [n for n, d in (('mask', s.mask if s.mask is not None else s.mask_tif), ('tracks', s.tracks)) if d is not None]
        parts = [f"{info['x']} × {info['y']} × {info['z']} voxels", f"{info['t']} timepoint{'s' if info['t'] != 1 else ''}"]
        if s.channels > 1:
            parts.append(f'{s.channels} channels')
        parts.append('found ' + ' and '.join(found) if found else 'no mask or tracks next to this image')
        self.source.set_detail('  ·  '.join(parts + s.notes))
        # the Armadillo mask belongs to the second channel, so start with that channel alone when that mask was found
        arm = s.channels > 1 and s.mask is None and s.mask_tif is not None and s.mask_name.endswith('_ARM_SEG.tif')
        s.channel = 1 if arm else 0
        self.build_layers([s.channel] if arm else range(s.channels))
        for slider, n in ((self.t_slider, info['t']), (self.z_slider, info['z'])):
            slider.blockSignals(True)
            slider.setRange(0, n - 1)
            slider.blockSignals(False)
        self.t_slider.setValue(s.seg_start - 1)
        self.z_slider.setValue(info['z'] // 2)
        self.render()

    def build_layers(self, shown):
        """Checkboxes for what can be drawn: each channel (or the image, if it has one), the mask and the tracks."""
        from PySide6.QtGui import QIcon, QPixmap
        from PySide6.QtWidgets import QCheckBox
        from .palette import channel_colour
        s = self.stack
        while self.layer_row.count():
            item = self.layer_row.takeAt(0)
            widget = item.widget()
            if widget:  # off screen now: deleteLater alone leaves it drawn until the event loop gets to it
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()

        def box(label, tip, checked, colour=None):
            b = QCheckBox(label)
            b.setToolTip(tip)
            b.setChecked(checked)
            if colour is not None:  # a swatch of the colour this layer is drawn in
                swatch = QPixmap(10, 10)
                swatch.fill(QColor(*colour))
                b.setIcon(QIcon(swatch))
            b.toggled.connect(self.layers_changed)
            self.layer_row.addWidget(b)
            return b

        self.colours_by_channel = [channel_colour(name, i) for i, name in enumerate(s.channel_names)]
        self.channel_boxes = []
        for i, name in enumerate(s.channel_names):
            if s.channels == 1:
                label = 'Image'
            else:
                short = name if len(name) <= self.NAME_LENGTH else name[:self.NAME_LENGTH - 1] + '…'
                label = f'C{i + 1}: {short}' if name else f'C{i + 1}'
            tip = f'Channel {i + 1}' + (f': {name}' if name else '') if s.channels > 1 else 'The image itself'
            self.channel_boxes.append(box(label, tip, i in shown, self.swatch_colour(i) if s.channels > 1 else None))
        has_mask = s.mask is not None or s.mask_tif is not None
        self.mask_box = box('Mask', 'The segmentation mask', has_mask and s.tracks is None) if has_mask else None
        if has_mask:  # the checkbox and its colour button move together when the row wraps
            pair = QWidget()
            line = QHBoxLayout(pair)
            line.setContentsMargins(0, 0, 0, 0)
            line.setSpacing(2)
            self.layer_row.removeWidget(self.mask_box)
            line.addWidget(self.mask_box)
            self.mask_colour_button = button('', 'ghost', tip='Colour of the mask')
            self.mask_colour_button.setMenu(self.mask_menu())
            self.paint_mask_button()
            line.addWidget(self.mask_colour_button)
            self.layer_row.addWidget(pair)
        self.tracks_box = box('Tracks', 'Tracked objects, each in its own colour. Click one to pick out its family.',
                              True) if s.tracks is not None else None
        self.layer_row.invalidate()
        self.layers.updateGeometry()
        self.contrast.set_channels(s.channel_names, self.colours_by_channel, s.channel)
        self.contrast.show_lut(s.luts.get(s.channel, 'Auto'), self.auto_colour(s.channel))

    MASK_COLOURS = [('Teal', '#35C6AE'), ('Red', '#FF3B30'), ('Green', '#34C759'), ('Blue', '#2F7BFF'),
                    ('Yellow', '#FFD60A'), ('Cyan', '#32D7FF'), ('Magenta', '#FF2DCB'), ('Orange', '#FF9500'),
                    ('White', '#FFFFFF')]

    def mask_colour(self):
        return QColor(QSettings().value('mask_colour', self.MASK_COLOURS[0][1]))

    def mask_menu(self):
        from PySide6.QtGui import QIcon, QPixmap
        from PySide6.QtWidgets import QColorDialog, QMenu
        menu = QMenu(self)
        for name, value in self.MASK_COLOURS:
            swatch = QPixmap(12, 12)
            swatch.fill(QColor(value))
            menu.addAction(QIcon(swatch), name, lambda v=value: self.set_mask_colour(QColor(v)))
        menu.addSeparator()

        def other():
            colour = QColorDialog.getColor(self.mask_colour(), self.window(), 'Mask colour')
            if colour.isValid():
                self.set_mask_colour(colour)
        menu.addAction('Other…', other)
        return menu

    def set_mask_colour(self, colour):
        QSettings().setValue('mask_colour', colour.name())
        self.paint_mask_button()
        if self.mask_box is not None and not self.mask_box.isChecked():
            self.mask_box.setChecked(True)  # choosing a colour means wanting to see it
        self.render()

    def paint_mask_button(self):
        from PySide6.QtGui import QIcon, QPixmap
        swatch = QPixmap(14, 14)
        swatch.fill(self.mask_colour())
        self.mask_colour_button.setIcon(QIcon(swatch))

    def auto_colour(self, channel):
        """What Auto looks like for a channel on the Colours strip: grey for an image with one channel."""
        return (255, 255, 255) if self.stack.channels == 1 else self.colours_by_channel[channel]

    def swatch_colour(self, channel):
        from .palette import AUTO, lut_end_colour
        return lut_end_colour(self.stack.luts.get(channel, AUTO), self.colours_by_channel[channel])

    def choose_lut(self, name):
        """A colour table for the channel being adjusted."""
        from PySide6.QtGui import QIcon, QPixmap
        if not self.stack:
            return
        c = self.stack.channel
        self.stack.luts[c] = name
        if self.stack.channels > 1:
            swatch = QPixmap(10, 10)
            swatch.fill(QColor(*self.swatch_colour(c)))
            self.channel_boxes[c].setIcon(QIcon(swatch))
            self.contrast.picker_buttons[c].setIcon(QIcon(swatch))
        self.render()

    def shown_channels(self):
        return [i for i, b in enumerate(self.channel_boxes) if b.isChecked()]

    def overlay_kind(self):
        """What a click picks from: the tracks when they are shown, else the mask when it is."""
        if self.tracks_box is not None and self.tracks_box.isChecked():
            return 'tracks'
        if self.mask_box is not None and self.mask_box.isChecked():
            return 'mask'
        return None

    def layers_changed(self, *_):
        if not self.stack:
            return
        shown = self.shown_channels()
        sender = self.sender()
        if sender in self.channel_boxes and sender.isChecked():  # a channel just turned on is the one to adjust
            self.adjust_channel(self.channel_boxes.index(sender))
        elif shown and self.stack.channel not in shown:
            self.adjust_channel(shown[0])
        else:
            self.render()

    def adjust_channel(self, index):
        """Point the brightness and contrast panel at a channel."""
        if self.stack and 0 <= index < self.stack.channels:
            from .palette import AUTO
            self.stack.channel = index
            self.contrast.set_active(index)
            self.contrast.show_lut(self.stack.luts.get(index, AUTO), self.auto_colour(index))
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
        # the channel being adjusted feeds the brightness and contrast panel, shown or not
        active = s.plane(t, z).astype(np.float32)
        levels = s.levels()
        levels.include(active.min(), active.max())
        self.contrast.display(levels, active, (id(s), s.channel, t, z, levels.data_min, levels.data_max))
        shown = self.shown_channels()
        self.planes_shown = {c: (active if c == s.channel else s.plane(t, z, c).astype(np.float32)) for c in shown}
        from .palette import AUTO, channel_lut
        # each channel through its colour table (Auto: grey alone, its own colour mixed); several are added up
        mix = np.zeros(active.shape + (3,), np.float32)
        for c in shown:
            plane = self.planes_shown[c]
            s.levels(c).include(plane.min(), plane.max())
            table = channel_lut(s.luts.get(c, AUTO), self.colours_by_channel[c], alone=len(shown) == 1)
            mix += table[(s.levels(c).scale(plane) * 255).astype(np.uint8)]
        rgb = np.clip(mix, 0, 255).astype(np.uint8)
        a = self.opacity.value() / 100
        self.labels = None
        for kind, b in (('mask', self.mask_box), ('tracks', self.tracks_box)):
            if b is None or not b.isChecked():
                continue
            labels = s.overlay(kind, t, z)
            if labels is None or labels.shape != active.shape:
                continue
            on = labels > 0
            if kind == 'mask':
                m = self.mask_colour()
                colour = np.array([m.red(), m.green(), m.blue()], np.float32)
                rgb[on] = (rgb[on] * (1 - a) + colour * a).astype(np.uint8)
            else:
                ids = labels[on].astype(np.int64)
                colour = self.colours[ids % len(self.colours)].astype(np.float32)
                # with a family picked out, its members are drawn solid and everything else is faint
                alpha = np.where(np.isin(ids, list(self.highlight)), 0.9, 0.18)[:, None] if self.highlight else a
                rgb[on] = (rgb[on] * (1 - alpha) + colour * alpha).astype(np.uint8)
            if kind == self.overlay_kind():
                self.labels = labels
        self.plane_shown = active
        kind = self.overlay_kind()
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
        parts = [f'x {x}, y {y}']
        planes = getattr(self, 'planes_shown', {})
        if self.stack.channels == 1 or not planes:
            parts.append(f'intensity {int(self.plane_shown[y, x])}')
        else:
            parts.append('  '.join(f'C{c + 1} {int(p[y, x])}' for c, p in sorted(planes.items())))
        if getattr(self, 'labels', None) is not None and self.labels.shape == self.plane_shown.shape and self.labels[y, x]:
            parts.append(f'object {int(self.labels[y, x])}' if self.overlay_kind() == 'tracks' else 'inside the mask')
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
        self.picked.setVisible(self.focus is not None)
        self.render()

    def click(self, x, y):
        """Clicking a tracked object picks out its family; clicking the background shows everything again."""
        if self.stack is None or self.overlay_kind() != 'tracks' or getattr(self, 'labels', None) is None:
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
        if self.tracks_box is not None:
            self.tracks_box.setChecked(True)
        self.t_slider.setValue(t)
        z = self.stack.best_slice(obj, t)
        if z is not None:
            self.z_slider.setValue(z)
        self.set_highlight(obj)
        if self.canvas.marker and not self.canvas.shows(*self.canvas.marker[:2]):  # zoomed in somewhere else
            self.canvas.centre_on(*self.canvas.marker[:2])

    def to_lineage(self):
        if self.focus is not None and self.stack:
            self.ctx.show_lineage(self.stack.tracking_folder, self.focus)
