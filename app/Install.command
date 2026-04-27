#!/bin/bash
set -e

ROOT="$HOME/eNPS_App"
ZIP_URL="https://github.com/AnastasiaVoronova/master-thesis/archive/refs/heads/main.zip"
ZIP_FILE="$ROOT/archive.zip"

echo "============================================================================"
echo "Downloading eNPS App from GitHub"
echo "Target folder: $ROOT"
echo "============================================================================"

mkdir -p "$ROOT"

if [ -z "$(ls -A "$ROOT")" ]; then
    echo "Downloading archive..."
    curl -L "$ZIP_URL" -o "$ZIP_FILE"

    echo "Extracting archive..."
    unzip -q "$ZIP_FILE" -d "$ROOT"
    rm -f "$ZIP_FILE"

    for DIR in "$ROOT"/*-main "$ROOT"/*-master; do
        if [ -d "$DIR" ]; then
            echo "Moving contents from $(basename "$DIR")..."
            cp -r "$DIR/app/." "$ROOT/"
            rm -rf "$DIR"
        fi
    done

    echo "============================================================================"
    echo "Repository ready in: $ROOT"
    echo "============================================================================"

    echo "Removing Windows-only files..."
    rm -f "$ROOT"/*.bat "$ROOT"/*.vbs

    if [ -f "$ROOT/SetUp.sh" ]; then
        echo "Running SetUp.sh..."
        bash "$ROOT/SetUp.sh"
    fi
else
    echo "Folder is not empty. Uninstall eNPS_App first."
fi
