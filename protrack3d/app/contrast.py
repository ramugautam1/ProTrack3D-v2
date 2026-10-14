"""Brightness and contrast for the viewer, after ImageJ's B&C window: a histogram with the display range drawn
over it, Minimum / Maximum / Brightness / Contrast sliders, Auto and Reset. Only the display changes, never the data."""
import os

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QDoubleValidator, QPainter, QPen
from PySide6.QtWidgets import (QButtonGroup, QDialog, QFileDialog, QGridLayout, QHBoxLayout, QLineEdit, QMessageBox,
                               QRadioButton, QSlider, QVBoxLayout, QWidget)

from .widgets import Card, ElidedLabel, button, text

STEPS = 1000  # slider positions
BINS = 256


class Levels:
    """The display range (low..high) of one channel inside the range of its data, with ImageJ's arithmetic."""

    def __init__(self, data_min, data_max, low=None, high=None, whole=True):
        self.data_min, self.data_max = float(data_min), float(max(data_max, data_min + 1))
        self.low = self.data_min if low is None else float(low)
        self.high = self.data_max if high is None else float(high)
        self.whole = whole  # the data are whole numbers
        self.auto_threshold = 0

    @property
    def span(self):
        return self.data_max - self.data_min

    def include(self, low, high):
        """Widen the data range when a plane goes beyond what was sampled."""
        self.data_min, self.data_max = min(self.data_min, float(low)), max(self.data_max, float(high))

    def scale(self, plane):
        """The plane as 0..1 under the current display range."""
        return np.clip((plane - self.low) / max(self.high - self.low, 1e-9), 0, 1)

    def set_range(self, low, high):
        self.low, self.high = float(min(low, high)), float(max(low, high))

    def set_minimum(self, fraction):
        self.low = self.data_min + fraction * self.span
        self.high = max(self.high, self.low)

    def set_maximum(self, fraction):
        self.high = self.data_min + fraction * self.span
        self.low = min(self.low, self.high)

    def set_brightness(self, fraction):
        """Slide the window without changing its width; further right is brighter."""
        centre, width = self.data_min + self.span * (1 - fraction), self.high - self.low
        self.low, self.high = centre - width / 2, centre + width / 2

    def set_contrast(self, fraction):
        """Narrow or widen the window around its centre; further right is more contrast."""
        centre = (self.low + self.high) / 2
        slope = fraction / 0.5 if fraction <= 0.5 else 0.5 / max(1 - fraction, 1e-6)
        if slope > 0:
            self.low, self.high = centre - 0.5 * self.span / slope, centre + 0.5 * self.span / slope

    def sliders(self):
        """Where the four sliders stand for the current range, each 0..1."""
        width = max(self.high - self.low, 1e-9)
        contrast = self.span / width * 0.5
        if contrast > 0.5:
            contrast = 1 - width / self.span * 0.5
        values = ((self.low - self.data_min) / self.span, (self.high - self.data_min) / self.span,
                  1 - ((self.low + self.high) / 2 - self.data_min) / self.span, contrast)
        return [min(1.0, max(0.0, v)) for v in values]

    def reset(self):
        self.low, self.high, self.auto_threshold = self.data_min, self.data_max, 0

    def auto(self, plane):
        """ImageJ's Auto: cut the thin tails off the histogram of this plane. Pressing again cuts deeper."""
        plane = np.asarray(plane)
        lo, hi = float(plane.min()), float(plane.max())
        if hi <= lo:
            return self.reset()
        counts = np.histogram(plane, BINS, (lo, hi))[0]
        self.auto_threshold = 5000 if self.auto_threshold < 10 else self.auto_threshold // 2
        counts = np.where(counts > plane.size // 10, 0, counts)  # a huge peak (the background) does not count
        above = np.nonzero(counts > plane.size // self.auto_threshold)[0]
        if len(above) and above[-1] > above[0]:
            step = (hi - lo) / BINS
            self.low, self.high = lo + above[0] * step, lo + above[-1] * step
        else:
            self.low, self.high = lo, hi


class Histogram(QWidget):
    """The histogram of the plane on screen (and, fainter, its log), with the display range as a line from black
    (bottom) to white (top)."""

    def __init__(self, theme):
        super().__init__()
        self.theme, self.counts, self.levels = theme, None, None
        self.setFixedHeight(96)

    def paintEvent(self, event):
        c = self.theme.colors
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        box = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(QColor(c['border']), 1))
        painter.setBrush(QColor(c['surface2']))
        painter.drawRoundedRect(box, 6, 6)
        if self.counts is None or self.levels is None:
            return
        inner = box.adjusted(4, 4, -4, -4)
        ordered = np.sort(self.counts)
        top = max(1.0, min(float(ordered[-1]), 1.5 * float(ordered[-2])))  # one tall bin must not flatten the rest
        painter.setPen(Qt.NoPen)
        width = inner.width() / len(self.counts)
        # behind: the same counts on a log scale, so the thin bright tail of a fluorescence image can be seen
        log_top = np.log1p(float(ordered[-1])) or 1.0
        for colour, heights in ((c['border'], np.log1p(self.counts) / log_top), (c['faint'], np.minimum(1.0, self.counts / top))):
            painter.setBrush(QColor(colour))
            for i, h in enumerate(heights):
                if h > 0:
                    painter.drawRect(QRectF(inner.x() + i * width, inner.bottom() - h * inner.height(), width + 0.5, h * inner.height()))
        lv = self.levels
        x1 = inner.x() + (lv.low - lv.data_min) / lv.span * inner.width()
        x2 = inner.x() + (lv.high - lv.data_min) / lv.span * inner.width()
        painter.setClipRect(inner)
        painter.setPen(QPen(QColor(c['accent']), 1.8))
        painter.drawLine(QPointF(inner.x() - 1, inner.bottom()), QPointF(x1, inner.bottom()))
        painter.drawLine(QPointF(x1, inner.bottom()), QPointF(x2, inner.top()))
        painter.drawLine(QPointF(x2, inner.top()), QPointF(inner.right() + 1, inner.top()))


class ContrastPanel(Card):
    changed = Signal()  # the display range was changed here
    save_requested = Signal()

    SLIDERS = [('Minimum', 'set_minimum', 'Everything at or below this value is black'),
               ('Maximum', 'set_maximum', 'Everything at or above this value is white'),
               ('Brightness', 'set_brightness', 'Slides the range up or down without changing its width'),
               ('Contrast', 'set_contrast', 'Narrows or widens the range around its centre')]

    def __init__(self, theme):
        super().__init__('Brightness & contrast', margins=14, spacing=10)
        self.setFixedWidth(280)
        self.levels, self.plane, self.key = None, None, None
        self.histogram = self.add(Histogram(theme))
        fields = QHBoxLayout()
        self.low_field, self.high_field = self._field('Black at this value. Type a number and press Return.'), \
            self._field('White at this value. Type a number and press Return.')
        self.high_field.setAlignment(Qt.AlignRight)
        fields.addWidget(self.low_field)
        fields.addStretch(1)
        fields.addWidget(self.high_field)
        self.box.addLayout(fields)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        self.sliders = []
        for row, (label, method, tip) in enumerate(self.SLIDERS):
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, STEPS)
            slider.setToolTip(tip)
            slider.valueChanged.connect(lambda value, m=method: self._slid(m, value))
            grid.addWidget(text(label, 'label'), row, 0)
            grid.addWidget(slider, row, 1)
            self.sliders.append(slider)
        self.box.addLayout(grid)
        buttons = QHBoxLayout()
        auto = button('Auto', tip='Fit the range to this plane. Press again to cut deeper into the tails.')
        auto.clicked.connect(self._auto)
        reset = button('Reset', tip='Show the whole range of the data')
        reset.clicked.connect(self._reset)
        save = button('Save image…', tip='Save the image with this brightness and contrast, as a copy or in place of the original')
        save.clicked.connect(self.save_requested)
        buttons.addWidget(auto)
        buttons.addWidget(reset)
        buttons.addStretch(1)
        self.box.addLayout(buttons)
        self.box.addWidget(save)
        self.box.addStretch(1)
        self.setEnabled(False)

    def _field(self, tip):
        field = QLineEdit()
        field.setFixedWidth(92)
        field.setToolTip(tip)
        field.setValidator(QDoubleValidator())
        field.editingFinished.connect(self._typed)
        return field

    def display(self, levels, plane, key):
        """Show the range of `levels` over the histogram of `plane`. `key` tells planes apart, so the histogram is counted once."""
        self.levels, self.plane = levels, plane
        if levels is None:
            self.histogram.counts = self.histogram.levels = None
            self.setEnabled(False)
            return self.histogram.update()
        self.setEnabled(True)
        if key != self.key or self.histogram.counts is None:
            self.key = key
            self.histogram.counts = np.histogram(plane, BINS, (levels.data_min, levels.data_max))[0]
        self.histogram.levels = levels
        self.histogram.update()
        for slider, value in zip(self.sliders, levels.sliders()):
            slider.blockSignals(True)
            slider.setValue(round(value * STEPS))
            slider.blockSignals(False)
        for field, value in ((self.low_field, levels.low), (self.high_field, levels.high)):
            if not field.hasFocus():
                field.setText(self._words(value))

    def _words(self, value):
        return str(int(round(value))) if self.levels.whole else f'{value:.4g}'

    def _slid(self, method, value):
        if self.levels is not None:
            getattr(self.levels, method)(value / STEPS)
            self.changed.emit()

    def _typed(self):
        if self.levels is None:
            return
        try:
            low, high = float(self.low_field.text().replace(',', '.')), float(self.high_field.text().replace(',', '.'))
        except ValueError:
            return
        if (self.low_field.text(), self.high_field.text()) != (self._words(self.levels.low), self._words(self.levels.high)):
            self.low_field.clearFocus()
            self.high_field.clearFocus()
            self.levels.set_range(low, high)
            self.changed.emit()

    def _auto(self):
        if self.levels is not None:
            self.levels.auto(self.plane)
            self.changed.emit()

    def _reset(self):
        if self.levels is not None:
            self.levels.reset()
            self.changed.emit()


class SaveDialog(QDialog):
    """Asks how to save an image with its brightness and contrast applied: as a copy under a new name, or in place
    of the original. `target` holds the chosen file when the dialog is accepted."""

    def __init__(self, parent, image, ranges, top, czi_problem=None):
        super().__init__(parent)
        self.setWindowTitle('Save image')
        self.setFixedWidth(560)
        self.image, self.target = image, None
        stem, extension = os.path.splitext(os.path.basename(image))
        czi = extension.lower() == '.czi'
        self.fixed = czi and czi_problem is not None  # a .czi that cannot be rewritten: only a copy, as .tif
        self.extension = '.nii' if extension.lower() == '.nii' else '.czi' if czi and not self.fixed else '.tif'
        self.allowed = {'.nii': ('.nii',), '.czi': ('.czi', '.tif', '.tiff')}.get(self.extension, ('.tif', '.tiff'))
        self.folder = os.path.dirname(image)
        box = QVBoxLayout(self)
        box.setContentsMargins(22, 20, 22, 18)
        box.setSpacing(12)
        box.addWidget(text('Save with this brightness and contrast', 'h2'))
        if len(ranges) == 1:
            what = f'{ranges[0][0]:g} becomes 0 and {ranges[0][1]:g} becomes {top:g}'
        else:
            what = 'in each channel the range it is shown with becomes 0 to ' + f'{top:g}'
        box.addWidget(text(f'The pixel values are rescaled: {what}. Everything outside is clipped. '
                           'All timepoints and slices are saved.', 'sub', wrap=True))
        self.copy, self.replace = QRadioButton('Save a copy'), QRadioButton('Replace the original')
        group = QButtonGroup(self)
        for choice in (self.copy, self.replace):
            group.addButton(choice)
            box.addWidget(choice)
        self.copy.setChecked(True)
        if self.fixed:
            self.replace.setEnabled(False)
            self.replace.setToolTip(f'This .czi cannot be rewritten: {czi_problem}. The copy is saved as a .tif.')
        line = QHBoxLayout()
        line.addWidget(text('Name', 'label'))
        self.name = QLineEdit(stem + '_BC' + self.extension)
        line.addWidget(self.name, 1)
        box.addLayout(line)
        if czi:
            box.addWidget(text('A .czi copy keeps all the metadata of the original; only the pixel values change. '
                               'End the name with .tif to save a .tif instead.' if not self.fixed else
                               f'This .czi cannot be rewritten ({czi_problem}), so the copy is a .tif.', 'hint', wrap=True))
        line = QHBoxLayout()
        self.where = ElidedLabel('', 'hint')
        folder = button('Change folder…', 'ghost')
        folder.clicked.connect(self.choose_folder)
        self.folder_button = folder
        line.addWidget(self.where, 1)
        line.addWidget(folder)
        box.addLayout(line)
        self.warning = text('Replacing cannot be undone. The measured intensities in this file are lost, and intensity '
                            'measurements made from it afterwards will not match the raw data.', 'error', wrap=True)
        box.addWidget(self.warning)
        self.problem = text('', 'error', wrap=True)
        box.addWidget(self.problem)
        line = QHBoxLayout()
        line.addStretch(1)
        cancel = button('Cancel')
        cancel.clicked.connect(self.reject)
        self.save = button('Save', 'primary')
        self.save.clicked.connect(self.confirm)
        self.save.setDefault(True)
        line.addWidget(cancel)
        line.addWidget(self.save)
        box.addLayout(line)
        self.replace.toggled.connect(self.choice_changed)
        self.choice_changed()
        self.name.setFocus()  # ready to type over the suggested name, keeping the extension
        self.name.setSelection(0, len(stem) + 3)

    def choice_changed(self, *_):
        replacing = self.replace.isChecked()
        if replacing:
            self.typed = self.name.text()
            self.name.setText(os.path.basename(self.image))
        elif getattr(self, 'typed', None) is not None:
            self.name.setText(self.typed)
        self.name.setEnabled(not replacing)
        self.folder_button.setVisible(not replacing)
        self.warning.setVisible(replacing)
        self.save.setText('Replace' if replacing else 'Save')
        self.problem.setText('')
        self.where.setText('in ' + (os.path.dirname(self.image) if replacing else self.folder))
        self.adjustSize()  # the warning comes and goes

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Choose where to save the copy', self.folder)
        if folder:
            self.folder = folder
            self.choice_changed()

    def confirm(self):
        if self.replace.isChecked():
            self.target = self.image
            return self.accept()
        name = self.name.text().strip()
        if not name or any(mark in name for mark in '/\\:*?"<>|'):
            return self.problem.setText('Give the copy a name without any of  / \\ : * ? " < > |')
        if not name.lower().endswith(self.allowed):
            name += self.extension
            self.name.setText(name)
        target = os.path.join(self.folder, name)
        if os.path.normcase(os.path.abspath(target)) == os.path.normcase(os.path.abspath(self.image)):
            return self.problem.setText('That is the name of the original. Choose "Replace the original" to overwrite it.')
        if os.path.exists(target) and QMessageBox.question(
                self, 'Save image', f'{name} already exists in that folder. Replace it?') != QMessageBox.Yes:
            return None
        self.target = target
        self.accept()
