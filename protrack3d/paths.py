import os
import sys

FROZEN = getattr(sys, 'frozen', False)  # True inside the packaged app
REPO_ROOT = getattr(sys, '_MEIPASS', None) or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def add_repo_to_path():
    """The tracking, family-tree and analysis modules live at the repository root."""
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)


def user_data_dir():
    """Where the packaged app keeps what the user adds to it (models)."""
    if sys.platform == 'darwin':
        base = os.path.expanduser('~/Library/Application Support')
    elif sys.platform == 'win32':
        base = os.environ.get('APPDATA') or os.path.expanduser('~/AppData/Roaming')
    else:
        base = os.environ.get('XDG_DATA_HOME') or os.path.expanduser('~/.local/share')
    return os.path.join(base, 'ProTrack3D')


def models_dir():
    """Models sit in the repository when running from source, and in the user's data folder for the packaged app."""
    return os.environ.get('PROTRACK3D_MODELS') or os.path.join(user_data_dir() if FROZEN else REPO_ROOT, 'models')
