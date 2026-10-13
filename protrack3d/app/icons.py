"""Line icons drawn from inline SVG, tinted to the current theme."""
from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round">{body}</svg>')

PATHS = {
    'home': '<path d="M4 11l8-7 8 7"/><path d="M6 10v9h12v-9"/><path d="M10 19v-5h4v5"/>',
    'pipeline': '<circle cx="5" cy="12" r="2.2"/><circle cx="12" cy="12" r="2.2"/><circle cx="19" cy="12" r="2.2"/><path d="M7.2 12h2.6M14.2 12h2.6"/>',
    'segment': '<path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3"/><circle cx="10" cy="10.5" r="2"/><circle cx="14.5" cy="14" r="1.5"/>',
    'track': '<circle cx="5" cy="18" r="2"/><circle cx="19" cy="6" r="2"/><path d="M7 18h4a3 3 0 0 0 3-3V9a3 3 0 0 1 3-3"/>',
    'tree': '<circle cx="6" cy="5" r="2"/><circle cx="6" cy="19" r="2"/><circle cx="18" cy="9" r="2"/><path d="M6 7v10M18 11c0 3-6 2-12 5"/>',
    'lineage': '<path d="M4 6h9M4 12h16M9 18h11"/><path d="M13 6v6M9 12v6"/>',
    'terminal': '<rect x="3" y="4.5" width="18" height="15" rx="2.5"/><path d="M7 9.5l3 2.5-3 2.5M12.5 14.5H17"/>',
    'train': '<path d="M4 18l5-6 4 3 7-9"/><path d="M15 6h5v5"/>',
    'armdia': '<circle cx="9" cy="12" r="5.5"/><circle cx="15" cy="12" r="5.5"/>',
    'analysis': '<path d="M4 20V4M4 20h16"/><path d="M8 16v-4M12 16V8M16 16v-6"/>',
    'viewer': '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="2.8"/>',
    'models': '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z"/><path d="M4 7.5l8 4.5 8-4.5M12 12v9"/>',
    'jobs': '<path d="M9 6h11M9 12h11M9 18h11"/><path d="M4 6l1 1 1.5-2M4 12l1 1 1.5-2M4 18l1 1 1.5-2"/>',
    'folder': '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z"/>',
    'play': '<path d="M7 5l12 7-12 7V5z"/>',
    'pause': '<path d="M8 5v14M16 5v14"/>',
    'stop': '<rect x="6" y="6" width="12" height="12" rx="2"/>',
    'check': '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    'x': '<path d="M6 6l12 12M18 6L6 18"/>',
    'moon': '<path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z"/>',
    'sun': '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.4 1.4M17.6 17.6L19 19M5 19l1.4-1.4M17.6 6.4L19 5"/>',
    'image': '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="1.6"/><path d="M4 18l5-5 4 4 3-3 4 4"/>',
    'chevron': '<path d="M9 6l6 6-6 6"/>',
    'clock': '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
    'log': '<path d="M6 3h9l4 4v14H6V3z"/><path d="M9 11h7M9 15h7M9 7h3"/>',
}


def pixmap(name, color, size=18):
    scale = 2
    pm = QPixmap(QSize(size * scale, size * scale))
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    QSvgRenderer(QByteArray(_SVG.format(color=color, body=PATHS[name]).encode())).render(painter, QRectF(0, 0, size * scale, size * scale))
    painter.end()
    pm.setDevicePixelRatio(scale)
    return pm


def icon(name, color, size=18):
    return QIcon(pixmap(name, color, size))
