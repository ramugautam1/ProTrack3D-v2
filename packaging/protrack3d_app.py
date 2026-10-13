"""Entry point of the packaged app. `ProTrack3D --engine <command>` runs the engine; otherwise the window opens."""
import sys

from protrack3d.app.__main__ import main

if __name__ == '__main__':
    sys.exit(main())
