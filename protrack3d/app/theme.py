"""Colours and the application style sheet. One palette per colour scheme; the style sheet is built from it."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QPalette

LIGHT = dict(
    bg='#F4F6F8', surface='#FFFFFF', surface2='#EEF1F4', sidebar='#E9EDF1', border='#D9DEE4',
    text='#16202A', muted='#5F6C7B', faint='#8B97A5',
    accent='#0E8F7E', accent_hover='#0B7C6D', accent_soft='#DDF3EF', on_accent='#FFFFFF',
    ok='#1E8E4E', warn='#B26A00', bad='#C4322B', info='#2563C9', viewer='#0C1116',
)
DARK = dict(
    bg='#12171D', surface='#1A2129', surface2='#222B35', sidebar='#151B22', border='#2C3742',
    text='#E7ECF1', muted='#9BA8B6', faint='#6E7B89',
    accent='#35C6AE', accent_hover='#4AD6BF', accent_soft='#16332F', on_accent='#06211D',
    ok='#4CC38A', warn='#E0A341', bad='#F0716A', info='#6BA4F8', viewer='#07090C',
)

QSS = """
* {{ font-size: 13px; color: {text}; outline: 0; }}
QMainWindow, #page, #content {{ background: {bg}; }}
QToolTip {{ background: {surface2}; color: {text}; border: 1px solid {border}; padding: 4px 6px; }}

#sidebar {{ background: {sidebar}; border-right: 1px solid {border}; }}
#brand {{ font-size: 18px; font-weight: 700; }}
#brandSub {{ color: {faint}; font-size: 11px; }}
#nav {{ text-align: left; padding: 9px 12px; border: none; border-radius: 8px; color: {muted}; font-weight: 500; background: transparent; }}
#nav:hover {{ background: {surface2}; color: {text}; }}
#nav:checked {{ background: {accent_soft}; color: {accent}; font-weight: 600; }}
#badge {{ background: {accent}; color: {on_accent}; border-radius: 9px; padding: 1px 7px; font-size: 11px; font-weight: 700; }}
#chip {{ background: {surface2}; color: {muted}; border-radius: 10px; padding: 3px 9px; font-size: 11px; }}
QPushButton#notice {{ text-align: left; padding: 8px 12px; border: none; border-radius: 8px; background: {accent_soft}; color: {accent}; font-weight: 600; font-size: 12px; }}
QPushButton#notice:hover {{ background: {surface2}; }}

#h1 {{ font-size: 24px; font-weight: 700; }}
#h2 {{ font-size: 15px; font-weight: 600; }}
#sub {{ color: {muted}; }}
#hint {{ color: {faint}; font-size: 12px; }}
#label {{ color: {muted}; font-weight: 500; }}
#error {{ color: {bad}; }}

#card {{ background: {surface}; border: 1px solid {border}; border-radius: 12px; }}
#action {{ background: {surface}; border: 1px solid {border}; border-radius: 12px; text-align: left; padding: 16px; }}
#action:hover {{ border-color: {accent}; background: {accent_soft}; }}
#drop {{ background: {surface2}; border: 1.5px dashed {border}; border-radius: 10px; }}
#drop[active="true"] {{ border-color: {accent}; background: {accent_soft}; }}
#drop[filled="true"] {{ border-style: solid; background: {surface}; }}

QPushButton {{ background: {surface}; border: 1px solid {border}; border-radius: 8px; padding: 7px 14px; font-weight: 500; }}
QPushButton:hover {{ border-color: {accent}; }}
QPushButton:pressed {{ background: {surface2}; }}
QPushButton:disabled {{ color: {faint}; border-color: {border}; background: {surface2}; }}
QPushButton#primary {{ background: {accent}; color: {on_accent}; border: none; padding: 9px 22px; font-weight: 600; }}
QPushButton#primary:hover {{ background: {accent_hover}; }}
QPushButton#primary:disabled {{ background: {border}; color: {faint}; }}
QPushButton#ghost {{ border: none; background: transparent; color: {muted}; padding: 5px 8px; }}
QPushButton#ghost:hover {{ color: {accent}; background: {surface2}; }}
QPushButton#ghost:checked {{ color: {accent}; }}
QPushButton#danger:hover {{ border-color: {bad}; color: {bad}; }}

QLineEdit, QComboBox, QPlainTextEdit {{ background: {surface}; border: 1px solid {border}; border-radius: 8px;
    padding: 6px 9px; selection-background-color: {accent}; selection-color: {on_accent}; }}
QLineEdit:focus, QComboBox:focus {{ border-color: {accent}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {surface}; border: 1px solid {border}; selection-background-color: {accent_soft};
    selection-color: {text}; padding: 4px; }}
QPlainTextEdit#log {{ font-family: Menlo, Consolas, 'DejaVu Sans Mono', monospace; font-size: 12px; background: {surface2}; }}
QRadioButton {{ spacing: 8px; }}
QRadioButton::indicator {{ width: 14px; height: 14px; border-radius: 8px; border: 1px solid {border}; background: {surface}; }}
QRadioButton::indicator:checked {{ width: 6px; height: 6px; border: 5px solid {accent}; background: {on_accent}; }}
QRadioButton::indicator:disabled {{ background: {surface2}; }}
QRadioButton:disabled {{ color: {faint}; }}
QDialog {{ background: {bg}; }}
QCheckBox {{ spacing: 7px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid {border}; background: {surface}; }}
QCheckBox::indicator:checked {{ background: {accent}; border-color: {accent}; image: url({check}); }}

QProgressBar {{ background: {surface2}; border: none; border-radius: 4px; max-height: 8px; min-height: 8px; }}
QProgressBar::chunk {{ background: {accent}; border-radius: 4px; }}
QSlider::groove:horizontal {{ height: 4px; background: {border}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {accent}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; background: {accent}; }}

QTableWidget {{ background: {surface}; border: 1px solid {border}; border-radius: 10px; gridline-color: {border}; }}
QTableWidget::item {{ padding: 4px 8px; }}
QTableWidget::item:selected {{ background: {accent_soft}; color: {text}; }}
QHeaderView::section {{ background: {surface2}; color: {muted}; border: none; border-bottom: 1px solid {border};
    padding: 7px 8px; font-weight: 600; font-size: 12px; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {border}; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {faint}; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{ height: 0; background: none; }}
QScrollBar:horizontal {{ height: 0; }}

#viewer {{ background: {viewer}; border-radius: 12px; }}
#viewer[active="true"] {{ border: 2px dashed {accent}; }}
QPushButton#viewerButton {{ background: rgba(255, 255, 255, 0.14); color: #FFFFFF; border: none; border-radius: 7px; padding: 0; font-weight: 600; }}
QPushButton#viewerButton:hover {{ background: rgba(255, 255, 255, 0.28); }}
#viewerLevel {{ color: rgba(255, 255, 255, 0.75); background: rgba(0, 0, 0, 0.45); border-radius: 6px; padding: 2px 7px; font-size: 11px; }}
#console {{ background: #0E1318; border-top: 1px solid {border}; }}
#consoleTitle {{ color: #E7ECF1; font-weight: 600; font-size: 12px; }}
#consoleStatus {{ color: #35C6AE; font-size: 12px; }}
#consolePrompt {{ color: #35C6AE; font-family: Menlo, Consolas, 'DejaVu Sans Mono', monospace; font-size: 12px; }}
QPushButton#consoleButton {{ background: transparent; border: none; color: #9BA8B6; padding: 3px 8px; font-size: 12px; }}
QPushButton#consoleButton:hover {{ color: #FFFFFF; background: #1F2933; }}
QPlainTextEdit#terminal {{ background: #0E1318; color: #D7E0E8; border: none; border-radius: 0; padding: 4px 10px;
    font-family: Menlo, Consolas, 'DejaVu Sans Mono', monospace; font-size: 12px; selection-background-color: #2A5C55; }}
QLineEdit#consoleInput {{ background: #0E1318; color: #E7ECF1; border: none; border-radius: 0; padding: 7px 2px;
    font-family: Menlo, Consolas, 'DejaVu Sans Mono', monospace; font-size: 12px; }}
QSplitter::handle:vertical {{ background: {border}; height: 1px; }}
#toast {{ background: {text}; color: {bg}; border-radius: 10px; padding: 10px 16px; font-weight: 500; }}
#state_done {{ color: {ok}; font-weight: 600; }}
#state_failed {{ color: {bad}; font-weight: 600; }}
#state_running {{ color: {accent}; font-weight: 600; }}
#state_queued, #state_cancelled {{ color: {faint}; font-weight: 600; }}
"""


class Theme:
    def __init__(self, app):
        self.app = app
        self.dark = QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark
        self.listeners = []

    @property
    def colors(self):
        return DARK if self.dark else LIGHT

    def apply(self):
        c = self.colors
        self.app.setStyle('Fusion')
        palette = QPalette()
        for role, key in ((QPalette.Window, 'bg'), (QPalette.Base, 'surface'), (QPalette.AlternateBase, 'surface2'),
                          (QPalette.Text, 'text'), (QPalette.WindowText, 'text'), (QPalette.ButtonText, 'text'),
                          (QPalette.Button, 'surface'), (QPalette.Highlight, 'accent'),
                          (QPalette.HighlightedText, 'on_accent'), (QPalette.PlaceholderText, 'faint')):
            palette.setColor(role, QColor(c[key]))
        self.app.setPalette(palette)
        self.app.setStyleSheet(QSS.format(check=self._check_mark(c['on_accent']), **c))
        for listener in self.listeners:
            listener()

    def _check_mark(self, color):
        """Style sheets can only take images from files, so the tick is drawn once into the temp folder."""
        import os
        import tempfile
        from . import icons
        path = os.path.join(tempfile.gettempdir(), f'protrack3d_check_{color.lstrip("#")}.png')
        if not os.path.isfile(path):
            icons.pixmap('check', color, 14).save(path)
        return path.replace('\\', '/')

    def toggle(self):
        self.dark = not self.dark
        self.apply()
