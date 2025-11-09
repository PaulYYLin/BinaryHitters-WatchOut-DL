#!/bin/bash
# Fall Detection System - Quick Run Script

set -e

echo "=================================="
echo "Fall Detection System"
echo "=================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed"
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "Please edit .env with your configuration"
    echo ""
fi

# Sync dependencies
echo "Installing dependencies..."
uv sync
echo ""

# Run the application
echo "Starting Fall Detection System..."
echo "Press Ctrl+C to stop"
echo ""
uv run python main.py
