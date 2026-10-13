"""Reusable widgets: headings, cards, drop-in path fields, animated progress, page transitions and toasts."""
import os

from PySide6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, QParallelAnimationGroup, Qt, QTimer, Signal
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QLineEdit, QProgressBar,
                               QPushButton, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget)

from . import icons


def text(value, name='', wrap=False):
    label = QLabel(value)
    if name:
        label.setObjectName(name)
    label.setWordWrap(wrap)
    return label


def button(label, name='', icon_name=None, color=None, tip=None):
    b = QPushButton(label)
    if name:
        b.setObjectName(name)
    if icon_name:
        b.setIcon(icons.icon(icon_name, color or '#888888', 16))
    if tip:
        b.setToolTip(tip)
    b.setCursor(Qt.PointingHandCursor)
    return b


class ElidedLabel(QLabel):
    """A one-line label that shortens long text (paths) in the middle instead of widening the window."""

    def __init__(self, value='', name=''):
        super().__init__(value)
        if name:
            self.setObjectName(name)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(40)

    def setText(self, value):
        super().setText(value)
        self.setToolTip(value)

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter
        painter = QPainter(self)
        painter.setPen(self.palette().color(self.foregroundRole()))
        painter.drawText(self.rect(), Qt.AlignVCenter | Qt.AlignLeft, self.fontMetrics().elidedText(self.text(), Qt.ElideMiddle, self.width()))


class Card(QFrame):
    def __init__(self, title=None, margins=18, spacing=12):
        super().__init__()
        self.setObjectName('card')
        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(margins, margins, margins, margins)
        self.box.setSpacing(spacing)
        if title:
            self.box.addWidget(text(title, 'h2'))

    def add(self, widget, stretch=0):
        self.box.addWidget(widget, stretch)
        return widget

    def row(self, label, widget, hint=None):
        """A labelled form row."""
        line = QHBoxLayout()
        line.setSpacing(12)
        name = text(label, 'label')
        name.setFixedWidth(150)
        line.addWidget(name, 0, Qt.AlignTop)
        column = QVBoxLayout()
        column.setSpacing(4)
        column.addWidget(widget)
        if hint:
            column.addWidget(text(hint, 'hint', wrap=True))
        line.addLayout(column, 1)
        self.box.addLayout(line)
        return widget


class PathField(QFrame):
    """Pick a file or folder by dropping it here or browsing. Emits `changed(path)`."""
    changed = Signal(str)

    def __init__(self, theme, mode='file', prompt='Drop a file here, or browse', filters='', settings_key=None):
        super().__init__()
        self.theme, self.mode, self.prompt, self.filters, self.key = theme, mode, prompt, filters, settings_key
        self.path = ''
        self.setObjectName('drop')
        self.setAcceptDrops(True)
        self.setMinimumHeight(58)
        row = QHBoxLayout(self)
        row.setContentsMargins(14, 10, 10, 10)
        row.setSpacing(12)
        self.icon = QLabel()
        row.addWidget(self.icon)
        column = QVBoxLayout()
        column.setSpacing(2)
        self.title = ElidedLabel(prompt, 'sub')
        self.detail = ElidedLabel('', 'hint')
        self.detail.hide()
        column.addWidget(self.title)
        column.addWidget(self.detail)
        row.addLayout(column, 1)
        self.clear_button = button('', 'ghost', tip='Clear')
        self.clear_button.clicked.connect(lambda: self.set_path(''))
        self.clear_button.hide()
        row.addWidget(self.clear_button)
        self.browse_button = button('Browse…')
        self.browse_button.clicked.connect(self.browse)
        row.addWidget(self.browse_button)
        theme.listeners.append(self.restyle)
        self.restyle()

    def restyle(self):
        c = self.theme.colors
        self.icon.setPixmap(icons.pixmap('folder' if self.mode == 'folder' else 'image', c['accent'] if self.path else c['faint'], 22))
        self.clear_button.setIcon(icons.icon('x', c['muted'], 14))

    def browse(self):
        from PySide6.QtCore import QSettings
        settings = QSettings()
        start = os.path.dirname(self.path) if self.path else settings.value('last_dir', os.path.expanduser('~'))
        if self.mode == 'folder':
            path = QFileDialog.getExistingDirectory(self, 'Choose a folder', start)
        else:
            path, _ = QFileDialog.getOpenFileName(self, 'Choose a file', start, self.filters)
        if path:
            settings.setValue('last_dir', os.path.dirname(path))
            self.set_path(path)

    def set_path(self, path, quiet=False):
        self.path = path or ''
        self.title.setText(os.path.basename(self.path.rstrip('/')) if self.path else self.prompt)
        self.title.setObjectName('' if self.path else 'sub')
        self.set_detail(os.path.dirname(self.path) if self.path else '')
        self.clear_button.setVisible(bool(self.path))
        self.setProperty('filled', 'true' if self.path else 'false')
        self._repolish()
        self.restyle()
        if not quiet:
            self.changed.emit(self.path)

    def set_detail(self, value):
        self.detail.setText(value)
        self.detail.setVisible(bool(value))

    def _repolish(self):
        for w in (self, self.title):
            w.style().unpolish(w)
            w.style().polish(w)

    def _accepts(self, event):
        urls = event.mimeData().urls()
        if len(urls) != 1 or not urls[0].isLocalFile():
            return None
        path = urls[0].toLocalFile()
        return path if os.path.isdir(path) == (self.mode == 'folder') else None

    def dragEnterEvent(self, event):
        if self._accepts(event):
            event.acceptProposedAction()
            self.setProperty('active', 'true')
            self._repolish()

    def dragLeaveEvent(self, event):
        self.setProperty('active', 'false')
        self._repolish()

    def dropEvent(self, event):
        path = self._accepts(event)
        self.setProperty('active', 'false')
        if path:
            self.set_path(path)
        self._repolish()


class NumberField(QLineEdit):
    """A whole-number field. Out-of-range entries are pulled back into range when editing ends."""
    valueChanged = Signal(int)

    def __init__(self, low=1, high=99999, value=1, width=90):
        super().__init__()
        self.low, self.high = low, high
        self.setFixedWidth(width)
        self.setValidator(QIntValidator(0, 9999999, self))
        self.setText(str(value))
        self.textChanged.connect(lambda _: self.valueChanged.emit(self.value()))
        self.editingFinished.connect(lambda: self.setValue(self.value()))

    def value(self):
        try:
            return max(self.low, min(self.high, int(self.text())))
        except ValueError:
            return self.low

    def setValue(self, value):
        value = str(max(self.low, min(self.high, int(value))))
        if value != self.text():
            self.setText(value)

    def setRange(self, low, high):
        self.low, self.high = low, high
        self.setValue(self.value())


class ModelBox(QComboBox):
    """The installed models. The last one used is remembered between sessions."""

    def __init__(self):
        super().__init__()
        self.reload()
        from PySide6.QtCore import QSettings
        last = self.findData(QSettings().value('last_model', ''))
        if last >= 0:
            self.setCurrentIndex(last)
        self.activated.connect(lambda _: QSettings().setValue('last_model', self.currentData()))

    def reload(self):
        from ..models import list_models
        from ..paths import models_dir
        current = self.currentData()
        self.clear()
        for m in list_models(models_dir()):
            iou = f"  ·  validation IoU {m['best_val_iou']:.2f}" if m.get('best_val_iou') else ''
            self.addItem(m['name'] + iou, m['path'])
        if current:
            self.setCurrentIndex(max(0, self.findData(current)))
        if not self.count():
            self.setPlaceholderText('No models installed')


class SmoothBar(QProgressBar):
    """A progress bar that glides to its new value; `set_fraction(None)` shows activity without a percentage."""

    def __init__(self):
        super().__init__()
        self.setTextVisible(False)
        self.setRange(0, 1000)
        self._anim = QPropertyAnimation(self, b'value', self)
        self._anim.setDuration(350)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

    def set_fraction(self, fraction):
        if fraction is None:
            self._anim.stop()
            self.setRange(0, 0)
            return
        if self.maximum() == 0:
            self.setRange(0, 1000)
        self._anim.stop()
        self._anim.setStartValue(self.value())
        self._anim.setEndValue(int(max(0, min(1, fraction)) * 1000))
        self._anim.start()


class FadeStack(QStackedWidget):
    """A stacked widget whose pages fade and slide in."""

    def show_page(self, index):
        if index == self.currentIndex():
            return
        self.setCurrentIndex(index)
        page = self.currentWidget()
        effect = QGraphicsOpacityEffect(page)
        page.setGraphicsEffect(effect)
        fade = QPropertyAnimation(effect, b'opacity', self)
        fade.setDuration(180)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        slide = QPropertyAnimation(page, b'pos', self)
        slide.setDuration(220)
        slide.setEasingCurve(QEasingCurve.OutCubic)
        slide.setStartValue(page.pos() + QPoint(0, 14))
        slide.setEndValue(page.pos())
        group = QParallelAnimationGroup(self)
        group.addAnimation(fade)
        group.addAnimation(slide)
        group.finished.connect(lambda: page.setGraphicsEffect(None))
        group.start()
        self._group = group


class Toast(QLabel):
    """A short message that slides up from the bottom of the window and fades away."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName('toast')
        self.hide()
        self._effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._effect)
        self._fade = QPropertyAnimation(self._effect, b'opacity', self)
        self._fade.setDuration(220)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._leave)

    def announce(self, message, seconds=4):
        self.setText(message)
        self.adjustSize()
        parent = self.parentWidget()
        self.move(parent.width() - self.width() - 24, parent.height() - self.height() - 24)
        self.show()
        self.raise_()
        self._fade.stop()
        self._fade.setStartValue(self._effect.opacity() if self.isVisible() else 0.0)
        self._fade.setEndValue(1.0)
        self._fade.start()
        self._timer.start(int(seconds * 1000))

    def _leave(self):
        self._fade.stop()
        self._fade.setStartValue(1.0)
        self._fade.setEndValue(0.0)
        self._fade.start()
        QTimer.singleShot(240, self.hide)


def spacer():
    w = QWidget()
    w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return w
