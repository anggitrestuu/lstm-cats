#!/bin/bash

# Create a virtual environment
echo "Creating virtual environment..."
python -m venv .venv

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies from requirements.txt
echo "Installing dependencies..."
pip install -r requirements.txt

# Verify installation
echo "Verifying installation..."
pip list

echo "Setup complete! To activate the virtual environment, run: source venv/bin/activate"

# run python clean_dataset.py
echo "Running clean_dataset.py..."
python clean_dataset.py