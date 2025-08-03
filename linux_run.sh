#!/bin/bash

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "[setup] Virtual environment does not exist, creating..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# First-time setup
if [ ! -f ".setup_done" ]; then
    echo "[setup] Installing dependencies..."
    pip install -r requirements.txt
    touch .setup_done
    echo "[setup] Complete."
fi

# Run app
echo "[Iris] Do not close this window to keep the bot running."
sleep 3
python3 main.py
