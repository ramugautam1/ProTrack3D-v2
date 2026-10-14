#!/bin/bash
# Starts the ProTrack3D app on macOS and Linux. The first run creates the conda environment (several minutes).
cd "$(dirname "$0")" || exit 1
ENV_NAME="${PROTRACK3D_ENV:-protrack3d-torch}"

if ! command -v conda >/dev/null 2>&1; then
    for d in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" /opt/homebrew/Caskroom/miniforge/base /opt/miniforge3 /opt/miniconda3 /opt/anaconda3; do
        if [ -f "$d/etc/profile.d/conda.sh" ]; then
            source "$d/etc/profile.d/conda.sh"
            break
        fi
    done
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found. Install Miniforge first: https://github.com/conda-forge/miniforge"
    exit 1
fi

if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "First run: setting up ProTrack3D. This takes several minutes and happens only once."
    conda create -y -n "$ENV_NAME" python=3.11 pip || exit 1
    conda run -n "$ENV_NAME" --no-capture-output pip install -r requirements.txt || { conda env remove -y -n "$ENV_NAME"; exit 1; }
fi

# Start the app on its own, so this window can be closed.
PYTHON="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"

# On macOS, Python is started through a small app bundle, so the Dock shows ProTrack3D and its icon, not Python.
if [ "$(uname)" = "Darwin" ]; then
    BUNDLE="$PWD/.launcher/ProTrack3D.app"
    rm -rf "$PWD/.launcher"
    mkdir -p "$BUNDLE/Contents/MacOS" "$BUNDLE/Contents/Resources"
    cp packaging/ProTrack3D.icns "$BUNDLE/Contents/Resources/"
    ln -s "$PYTHON" "$BUNDLE/Contents/MacOS/ProTrack3D"
    cat > "$BUNDLE/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
    <key>CFBundleName</key><string>ProTrack3D</string>
    <key>CFBundleDisplayName</key><string>ProTrack3D</string>
    <key>CFBundleExecutable</key><string>ProTrack3D</string>
    <key>CFBundleIconFile</key><string>ProTrack3D.icns</string>
    <key>CFBundleIdentifier</key><string>edu.unlv.protrack3d.source</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
    PYTHON="$BUNDLE/Contents/MacOS/ProTrack3D"
fi
LOG="${TMPDIR:-/tmp}/protrack3d-launch.log"
nohup "$PYTHON" -W ignore -m protrack3d.app > "$LOG" 2>&1 &
APP=$!
sleep 4
if kill -0 "$APP" 2>/dev/null; then
    echo "ProTrack3D is running. You can close this window."
else
    echo "ProTrack3D did not start:"
    cat "$LOG"
    exit 1
fi
