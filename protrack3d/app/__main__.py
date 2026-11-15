import os
import sys

ICON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icon.png')


def main():
    # The name and icon in the Dock come from the launcher, which starts Python through a small app bundle on macOS.
    if sys.platform == 'win32':
        import ctypes
        try:  # so the taskbar shows ProTrack3D's icon rather than Python's
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('ProTrack3D.App')
        except Exception:
            pass
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication
    from .main_window import MainWindow
    from .theme import Theme
    app = QApplication(sys.argv)
    app.setOrganizationName('ProTrack3D')
    app.setApplicationName('ProTrack3D')
    app.setDesktopFileName('protrack3d')  # Linux: ties the window to the menu entry the launcher writes, and its icon
    if os.path.isfile(ICON):
        app.setWindowIcon(QIcon(ICON))  # the Dock icon on macOS, the window and taskbar icon elsewhere
    theme = Theme(app)
    theme.apply()
    window = MainWindow(theme)
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
