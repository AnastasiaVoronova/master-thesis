#!/bin/bash
set -e

REPO_DIR="$HOME/eNPS_App"
ENV_DIR="$REPO_DIR/runtime"
VENV_DIR="$ENV_DIR/venv"

echo "============================================================================"
echo "Setting up eNPS_App in: $REPO_DIR"
echo "============================================================================"

mkdir -p "$ENV_DIR"

if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python3 not found."
    echo "Install it from https://www.python.org or run: brew install python3"
    exit 1
fi

echo "Using $(python3 --version)"

if [ -d "$VENV_DIR" ]; then
    rm -rf "$VENV_DIR"
fi

echo "Creating virtual environment..."
python3 -m venv "$VENV_DIR"

source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "Installing dependencies..."
if [ -f "$REPO_DIR/requirements.txt" ]; then
    pip install -r "$REPO_DIR/requirements.txt"
else
    echo "WARNING: requirements.txt not found in $REPO_DIR"
fi

echo "Downloading models from HuggingFace..."
mkdir -p "$REPO_DIR/final_model"
curl -L --progress-bar "https://huggingface.co/AnastasiaVoronova/enps_binary/resolve/main/binary.pt" -o "$REPO_DIR/final_model/binary.pt"
curl -L --progress-bar "https://huggingface.co/AnastasiaVoronova/enps_multiclass/resolve/main/multiclass.pt" -o "$REPO_DIR/final_model/multiclass.pt"

SHORTCUT_PATH="$HOME/Desktop/eNPS_App.command"
cat > "$SHORTCUT_PATH" << SHORTCUT_EOF
#!/bin/bash
bash "$HOME/eNPS_App/run.command"
SHORTCUT_EOF
chmod +x "$SHORTCUT_PATH"

if [ -f "$SHORTCUT_PATH" ]; then
    echo "Link created: $SHORTCUT_PATH"
else
    echo "Link NOT created. Run app with eNPS_App/run.command"
fi

ICON_ICO="$REPO_DIR/media/logo.ico"

if [ -f "$ICON_ICO" ]; then
    osascript << ASEOF
use framework "AppKit"
set theImage to current application's NSImage's alloc()'s initWithContentsOfFile_("$ICON_ICO")
current application's NSWorkspace's sharedWorkspace()'s setIcon:theImage forFile:"$SHORTCUT_PATH" options:0
ASEOF
fi

echo "============================================================================"
echo "Set Up Finished: $REPO_DIR"
echo "============================================================================"
