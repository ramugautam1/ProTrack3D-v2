import sys


def main():
    if '--engine' in sys.argv:  # a packaged app runs its own engine through this switch
        from ..cli import main as engine
        return engine(sys.argv[sys.argv.index('--engine') + 1:])
    from PySide6.QtWidgets import QApplication
    from .main_window import MainWindow
    from .theme import Theme
    app = QApplication(sys.argv)
    app.setOrganizationName('ProTrack3D')
    app.setApplicationName('ProTrack3D')
    theme = Theme(app)
    theme.apply()
    window = MainWindow(theme)
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
