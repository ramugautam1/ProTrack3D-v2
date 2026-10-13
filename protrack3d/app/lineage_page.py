"""Lineage: every tracked object as a line through time. Click one to see its family; double-click to open it in the viewer."""
import os

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QScrollBar, QWidget

from ..lineage import Lineage
from .jobs import Job
from .pages import Page, spin
from .palette import id_colours
from .widgets import PathField, button, text

HEADER, GUTTER, RIGHT = 26, 58, 46  # pixels: time axis, family labels, room for ids at the line ends
MIN_ROW, MAX_ROW = 0.4, 30.0


class LineageCanvas(QWidget):
    hovered = Signal(object)   # object id or None
    selected = Signal(object)
    opened = Signal(int)       # double-click
    scrolled = Signal()

    def __init__(self, theme):
        super().__init__()
        self.theme, self.lineage, self.order, self.row = theme, None, [], {}
        self.colours = id_colours()
        self.row_h, self.top = 6.0, 0.0  # pixels per object; first visible row
        self.selection, self.family, self.hover = None, set(), None
        self._drag = None
        self.setMouseTracking(True)
        self.setMinimumHeight(300)
        self.setFocusPolicy(Qt.StrongFocus)

    # --- data ---
    def set_data(self, lineage, order):
        self.lineage, self.order = lineage, order
        self.row = {obj: i for i, obj in enumerate(order)}
        if self.selection not in self.row:
            self.selection, self.family = None, set()
        self.fit()

    def fit(self):
        rows = max(1, len(self.order))
        self.row_h = max(MIN_ROW, min(18.0, (self.height() - HEADER - 8) / rows))
        self.set_top(0)

    def visible_rows(self):
        return max(1.0, (self.height() - HEADER) / self.row_h)

    def set_top(self, top):
        self.top = max(0.0, min(top, max(0.0, len(self.order) - self.visible_rows() + 2)))
        self.scrolled.emit()
        self.update()

    def zoom(self, factor, at_y=None):
        at_y = self.height() / 2 if at_y is None else at_y
        anchor = self.top + (at_y - HEADER) / self.row_h
        self.row_h = max(MIN_ROW, min(MAX_ROW, self.row_h * factor))
        self.set_top(anchor - (at_y - HEADER) / self.row_h)

    def select(self, obj, reveal=False):
        self.selection = obj if obj in self.row else None
        self.family = set(self.lineage.members(obj)) if self.selection is not None else set()
        if reveal and self.selection is not None:
            if self.row_h < 9:
                self.row_h = 12.0
            self.set_top(self.row[obj] - self.visible_rows() / 2)
        self.selected.emit(self.selection)
        self.update()

    # --- geometry ---
    def _px(self):
        span = self.lineage.last - self.lineage.first + 1
        return max(1.0, (self.width() - GUTTER - RIGHT) / span)

    def _x(self, t):
        return GUTTER + (t - self.lineage.first + 0.5) * self._px()

    def _y(self, row):
        return HEADER + (row - self.top + 0.5) * self.row_h

    def object_at(self, pos):
        if not self.lineage or pos.y() < HEADER:
            return None
        row = int(self.top + (pos.y() - HEADER) / self.row_h)
        if not 0 <= row < len(self.order):
            return None
        obj = self.order[row]
        start, end, _ = self.lineage.objects[obj]
        slack = max(0.5 * self._px(), 4)
        return obj if self._x(start) - slack <= pos.x() <= self._x(end) + slack else None

    # --- drawing ---
    def paintEvent(self, event):
        c = self.theme.colors
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(c['surface']))
        if not self.lineage:
            p.setPen(QColor(c['faint']))
            p.drawText(self.rect(), Qt.AlignCenter, 'Choose a tracking results folder to see its lineage.')
            return
        L, px, rh = self.lineage, self._px(), self.row_h
        first_row = max(0, int(self.top))
        last_row = min(len(self.order), int(self.top + self.visible_rows()) + 2)
        dim = self.selection is not None
        picked = L.family[self.selection] if dim else None
        p.setRenderHint(QPainter.Antialiasing, rh >= 3)

        # time grid
        step = next(s for s in (1, 2, 5, 10, 20, 50, 100) if s * px >= 34)
        p.setFont(QFont(self.font().family(), 9))
        for t in range(L.first, L.last + 1):
            if t % step == 0 or t == L.first:
                x = self._x(t)
                p.setPen(QPen(QColor(c['border']), 1))
                p.drawLine(QPointF(x, HEADER), QPointF(x, self.height()))
                p.setPen(QColor(c['muted']))
                p.drawText(QRectF(x - 20, 4, 40, HEADER - 8), Qt.AlignCenter, str(t))

        # family bands and labels
        p.setClipRect(QRectF(0, HEADER, self.width(), self.height() - HEADER))
        row = first_row
        while row < last_row:
            fam = L.family[self.order[row]]
            end = row
            while end + 1 < len(self.order) and L.family[self.order[end + 1]] == fam:
                end += 1
            begin = row
            while begin > 0 and L.family[self.order[begin - 1]] == fam:
                begin -= 1
            if L.family_size[fam] > 1:
                top, bottom = self._y(begin) - rh / 2, self._y(end) + rh / 2
                if fam == picked:
                    p.fillRect(QRectF(0, top, self.width(), bottom - top), QColor(c['accent_soft']))
                if rh >= 4:
                    p.setPen(QPen(QColor(c['border']), 1))
                    p.drawLine(QPointF(0, top), QPointF(self.width(), top))
                if bottom - top >= 11:
                    p.setPen(QColor(c['muted']))
                    label_y = max(top, HEADER) if bottom - max(top, HEADER) >= 11 else top
                    p.drawText(QRectF(4, label_y, GUTTER - 10, min(bottom - label_y, 16)), Qt.AlignLeft | Qt.AlignVCenter, str(fam))
            row = end + 1

        # With room to spare the lineage is drawn like the family-tree pictures: a dot per timepoint an object exists,
        # joined by its life line, with slanted links for splits and merges. Zoomed out, dots would run together, so
        # each object becomes one solid line and events become ticks.
        nodes = rh >= 9 and px >= 9
        radius = min(rh, px, 16) * 0.3
        width = 1.4 if nodes else max(1.0, min(rh * 0.55, 9.0))
        label = rh >= 11
        p.setFont(QFont(self.font().family(), max(7, min(10, int(rh * 0.7)))))
        reach = 0 if nodes else 0.4 * px
        for r in range(first_row, last_row):
            obj = self.order[r]
            start, end, _ = L.objects[obj]
            related = L.family_size[L.family[obj]] > 1
            colour = QColor(*[int(v) for v in self.colours[obj % len(self.colours)]]) if related else QColor(c['faint'])
            if dim and obj not in self.family:
                colour.setAlpha(55)
            y = self._y(r)
            p.setPen(QPen(colour, width * (1.5 if obj == self.selection else 1), Qt.SolidLine, Qt.RoundCap if rh >= 5 else Qt.FlatCap))
            p.drawLine(QPointF(self._x(start) - reach, y), QPointF(self._x(end) + reach, y))
            if nodes:
                p.setPen(Qt.NoPen)
                p.setBrush(colour)
                for t in range(start, end + 1):
                    p.drawEllipse(QPointF(self._x(t), y), radius, radius)
                p.setBrush(Qt.NoBrush)
            edge = max(reach, radius)
            if obj == self.selection or obj == self.hover:
                p.setPen(QPen(QColor(c['text']), 1))
                p.drawRoundedRect(QRectF(self._x(start) - edge - 3, y - max(rh, 8) / 2, (end - start) * px + 2 * edge + 6, max(rh, 8)), 3, 3)
            if label or obj == self.selection:
                p.setPen(QColor(c['text'] if not dim or obj in self.family else c['faint']))
                p.drawText(QPointF(self._x(end) + edge + 5, y + 3.5), str(obj))

        # splits and merges
        for links, colour, style in ((L.splits, '#2B4FD8', Qt.SolidLine), (L.merges, '#E0372E', Qt.DashLine if nodes else Qt.SolidLine),
                                     (L.partial if L.partial_merges else [], '#E08A1B', Qt.DotLine)):
            for a, b, t in links:
                ra, rb = self.row.get(a), self.row.get(b)
                if ra is None or rb is None or max(ra, rb) < first_row or min(ra, rb) > last_row:
                    continue
                col = QColor(colour)
                if dim and a not in self.family:
                    col.setAlpha(45)
                p.setPen(QPen(col, 1.4 if rh >= 4 else 1, style))
                if nodes:  # from where the first object was, one timepoint earlier, to the other object now
                    p.drawLine(QPointF(self._x(t - 1), self._y(ra)), QPointF(self._x(t), self._y(rb)))
                else:
                    x = self._x(t) - 0.5 * px
                    p.drawLine(QPointF(x, self._y(ra)), QPointF(x, self._y(rb)))
        p.setClipping(False)
        p.setPen(QPen(QColor(c['border']), 1))
        p.drawLine(QPointF(0, HEADER), QPointF(self.width(), HEADER))

    # --- interaction ---
    def wheelEvent(self, event):
        if event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
            self.zoom(1.15 if event.angleDelta().y() > 0 else 1 / 1.15, event.position().y())
        else:
            pixels = event.pixelDelta().y() or event.angleDelta().y() / 2
            self.set_top(self.top - pixels / self.row_h)

    def event(self, event):
        if event.type() == QEvent.NativeGesture and event.gestureType() == Qt.ZoomNativeGesture:  # trackpad pinch
            self.zoom(1 + event.value(), event.position().y())
            return True
        return super().event(event)

    def mousePressEvent(self, event):
        self._drag = (event.position().y(), self.top, False)

    def mouseMoveEvent(self, event):
        if self._drag and event.buttons() & Qt.LeftButton:
            y0, top0, _ = self._drag
            if abs(event.position().y() - y0) > 4:
                self._drag = (y0, top0, True)
                self.set_top(top0 - (event.position().y() - y0) / self.row_h)
            return
        obj = self.object_at(event.position())
        if obj != self.hover:
            self.hover = obj
            self.setCursor(Qt.PointingHandCursor if obj is not None else Qt.ArrowCursor)
            self.hovered.emit(obj)
            self.update()

    def mouseReleaseEvent(self, event):
        moved = self._drag and self._drag[2]
        self._drag = None
        if not moved and self.lineage:
            self.select(self.object_at(event.position()))

    def mouseDoubleClickEvent(self, event):
        obj = self.object_at(event.position())
        if obj is not None:
            self.opened.emit(obj)

    def leaveEvent(self, event):
        if self.hover is not None:
            self.hover = None
            self.hovered.emit(None)
            self.update()

    def resizeEvent(self, event):
        self.set_top(self.top)


class LineagePage(Page):
    def __init__(self, ctx):
        super().__init__(ctx, 'Lineage', None)
        self.lineage = None
        self.source = PathField(ctx.theme, 'folder', 'Drop a …_TrackingOutput folder here, or browse')
        self.source.changed.connect(self.open)
        self.body.addWidget(self.source)

        tools = QHBoxLayout()
        tools.setSpacing(10)
        self.find = QLineEdit()
        self.find.setPlaceholderText('Find object ID')
        self.find.setFixedWidth(130)
        self.find.returnPressed.connect(self.find_object)
        tools.addWidget(self.find)
        tools.addWidget(text('Hide unrelated objects shorter than', 'label'))
        self.min_life = spin(1, 9999, 1, 56)
        self.min_life.editingFinished.connect(self.reorder)
        tools.addWidget(self.min_life)
        tools.addWidget(text('timepoints', 'label'))
        tools.addStretch(1)
        for label, tip, action in (('−', 'Thinner rows', lambda: self.canvas.zoom(1 / 1.4)), ('+', 'Thicker rows', lambda: self.canvas.zoom(1.4)),
                                   ('Fit', 'Show everything', lambda: self.canvas.fit())):
            b = button(label, tip=tip)
            b.clicked.connect(action)
            tools.addWidget(b)
        self.body.addLayout(tools)

        area = QHBoxLayout()
        area.setSpacing(0)
        self.canvas = LineageCanvas(ctx.theme)
        self.bar = QScrollBar(Qt.Vertical)
        area.addWidget(self.canvas, 1)
        area.addWidget(self.bar)
        self.body.addLayout(area, 1)
        self.canvas.scrolled.connect(self.sync_bar)
        self.bar.valueChanged.connect(lambda v: self.canvas.set_top(v / 10) if abs(v / 10 - self.canvas.top) > 0.05 else None)
        self.canvas.hovered.connect(self.show_info)
        self.canvas.selected.connect(self.selection_changed)
        self.canvas.opened.connect(self.open_in_viewer)

        foot = QHBoxLayout()
        self.info = text('', 'sub', wrap=True)
        foot.addWidget(self.info, 1)
        self.view_button = button('Show in viewer', 'primary')
        self.view_button.clicked.connect(lambda: self.open_in_viewer(self.canvas.selection))
        self.tree_button = button('Draw this family')
        self.tree_button.clicked.connect(self.draw_family)
        for b in (self.tree_button, self.view_button):
            b.setEnabled(False)
            foot.addWidget(b)
        self.body.addLayout(foot)
        ctx.theme.listeners.append(self.canvas.update)
        self.show_info(None)

    def open(self, folder):
        self.lineage = None
        if folder:
            try:
                self.lineage = Lineage(folder)
            except Exception as e:
                self.source.set_detail(str(e))
        if self.lineage:
            L = self.lineage
            families = sum(v > 1 for v in L.family_size.values())
            self.source.set_detail(f'{len(L.objects)} objects over timepoints {L.first}–{L.last}  ·  {families} families of two or more')
        self.reorder()

    def reorder(self):
        self.canvas.set_data(self.lineage, self.lineage.order(self.min_life.value()) if self.lineage else [])
        self.selection_changed(self.canvas.selection)

    def sync_bar(self):
        self.bar.blockSignals(True)
        self.bar.setRange(0, int(max(0, len(self.canvas.order) - self.canvas.visible_rows() + 2) * 10))
        self.bar.setPageStep(int(self.canvas.visible_rows() * 10))
        self.bar.setValue(int(self.canvas.top * 10))
        self.bar.blockSignals(False)

    def show_info(self, obj):
        obj = obj if obj is not None else self.canvas.selection
        if self.lineage and obj is not None:
            self.info.setText(self.lineage.describe(obj))
        else:
            self.info.setText('Click an object to highlight its family. Double-click to open it in the viewer. '
                              'Scroll to move, pinch or Ctrl-scroll to change the row height.' if self.lineage else '')

    def selection_changed(self, obj):
        for b in (self.tree_button, self.view_button):
            b.setEnabled(obj is not None)
        self.show_info(obj)

    def find_object(self):
        try:
            obj = int(self.find.text().strip())
        except ValueError:
            return
        if not self.lineage or obj not in self.lineage.objects:
            self.ctx.toast(f'There is no object {obj} in this movie')
        elif obj not in self.canvas.row:
            self.ctx.toast(f'Object {obj} is hidden by the lifetime filter')
        else:
            self.canvas.select(obj, reveal=True)

    def show_object(self, folder, obj=None):
        if os.path.normpath(folder) != os.path.normpath(self.source.path or '.'):
            self.source.set_path(folder)
        if obj is not None and self.lineage and obj in self.canvas.row:
            self.canvas.select(obj, reveal=True)

    def open_in_viewer(self, obj):
        if obj is None or not self.lineage:
            return
        self.canvas.select(obj)
        self.ctx.view_object(self.lineage, obj)

    def draw_family(self):
        obj = self.canvas.selection
        if obj is None:
            return
        fam = self.lineage.family[obj]
        self.ctx.submit(Job(f'Family tree  ·  family {fam}', ['tree', self.lineage.folder, '--ids', str(obj), '--complete'],
                            ['family tree'], folder=self.lineage.folder))
