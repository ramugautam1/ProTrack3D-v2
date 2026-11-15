import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def add_repo_to_path():
    """The tracking, family-tree and analysis modules live at the repository root."""
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)


def models_dir():
    """Models sit in the repository's models folder, unless PROTRACK3D_MODELS names another."""
    return os.environ.get('PROTRACK3D_MODELS') or os.path.join(REPO_ROOT, 'models')
