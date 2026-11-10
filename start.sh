#!/usr/bin/env bash
# ==============================================================================
# CardioPulse AI - One-Click Launcher & Verification Script
# ==============================================================================

set -e

echo "========================================================================"
echo "  CARDIOPULSE AI - CLINICAL CARDIOVASCULAR PLATFORM LAUNCHER"
echo "========================================================================"

# Step 1: Check Python installation
if command -v python3 &>/dev/null; then
    PY_BIN="python3"
elif command -v python &>/dev/null; then
    PY_BIN="python"
else
    echo "[ERROR] Python 3 is required but could not be found."
    exit 1
fi

PY_VER=$($PY_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo "[INFO] Detected Python Version: $PY_VER"

# Step 2: Run Automated Unit Tests
echo "[INFO] Running automated unit test suite (test_pipeline.py)..."
$PY_BIN test_pipeline.py

# Step 3: Launch Local Web Server
PORT=${PORT:-8000}
echo "------------------------------------------------------------------------"
echo "[SUCCESS] Unit tests verified! Starting local application server..."
echo "          Open your browser to: http://localhost:$PORT"
echo "          Press Ctrl+C to terminate the server."
echo "------------------------------------------------------------------------"

$PY_BIN app.py

