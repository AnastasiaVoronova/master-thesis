#!/bin/bash

ROOT="$HOME/eNPS_App"
SHORTCUT="$HOME/Desktop/eNPS_App.command"

echo "============================================================================"
echo "Deleting eNPS_App..."
echo "============================================================================"

if [ -d "$ROOT" ]; then
    echo "Deleting project folder..."
    rm -rf "$ROOT"
else
    echo "Project folder not found: skipping."
fi

if [ -f "$SHORTCUT" ]; then
    echo "Deleting link from Desktop..."
    rm -f "$SHORTCUT"
else
    echo "Link not found: skipping."
fi

echo "============================================================================"
echo "eNPS_App successfully deleted"
echo "============================================================================"
