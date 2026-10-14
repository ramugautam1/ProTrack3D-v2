import os
import sys

ICON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icon.png')


def _mac_name(name):
    """Run from source, the app is Python as far as macOS knows, and the Dock calls it that. Launch Services takes
    another name for a running process; this is how Chromium names its helper processes."""
    import ctypes
    import ctypes.util
    try:
        ls = ctypes.cdll.LoadLibrary('/System/Library/Frameworks/CoreServices.framework/CoreServices')
        cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library('CoreFoundation'))
        cf.CFStringCreateWithCString.restype = ctypes.c_void_p
        cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
        ls._LSGetCurrentApplicationASN.restype = ctypes.c_void_p
        ls._LSSetApplicationInformationItem.restype = ctypes.c_int32
        ls._LSSetApplicationInformationItem.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        key = ctypes.c_void_p.in_dll(ls, '_kLSDisplayNameKey').value
        asn = ls._LSGetCurrentApplicationASN()
        if asn and key:
            ls._LSSetApplicationInformationItem(-2, asn, key, cf.CFStringCreateWithCString(None, name.encode(), 0x08000100), None)
    except (OSError, AttributeError, ValueError):
        pass


def main():
    if '--engine' in sys.argv:  # a packaged app runs its own engine through this switch
        from ..cli import main as engine
        return engine(sys.argv[sys.argv.index('--engine') + 1:])
    frozen = getattr(sys, 'frozen', False)  # the packaged app has its own name and icon
    if sys.platform == 'win32' and not frozen:
        import ctypes
        try:  # so the taskbar shows ProTrack3D's icon rather than Python's
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('ProTrack3D.App')
        except Exception:
            pass
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication
    from .main_window import MainWindow
    from .theme import Theme
    app = QApplication(sys.argv)
    app.setOrganizationName('ProTrack3D')
    app.setApplicationName('ProTrack3D')
    if os.path.isfile(ICON):
        app.setWindowIcon(QIcon(ICON))  # the Dock icon on macOS, the window and taskbar icon elsewhere
    theme = Theme(app)
    theme.apply()
    window = MainWindow(theme)
    window.show()
    if sys.platform == 'darwin' and not frozen:
        QTimer.singleShot(0, lambda: _mac_name('ProTrack3D'))  # once macOS has registered the app
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
