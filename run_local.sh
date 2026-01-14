#!/bin/bash
# Run pipeline directly in Cloud Shell (No Docker)

echo "=========================================="
echo "Setup & Run Pipeline Locally"
echo "=========================================="

# 1. Fix Permissions first
echo "Checking permissions..."
chmod +x fix_permissions.sh
./fix_permissions.sh

# 2. Setup Virtual Environment
echo "Setting up Python environment..."
if [ -d "venv" ]; then
    echo "Existing venv found."
    source venv/bin/activate
else
    echo "Creating new venv..."
    python3 -m venv venv
    source venv/bin/activate
fi

# 3. Install Dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run Pipeline
echo "Running pipeline..."
python3 run_pipeline.py
