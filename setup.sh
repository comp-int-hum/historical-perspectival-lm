#!/usr/bin/env bash
#
# Automatic setup script for Perspectival Language Models
# Usage: ./setup.sh
#
# Exit immediately if a command exits with a non-zero status
set -e

# -----------------------------------------------------------------------------
# (Optional) Create and activate a virtual environment
# Uncomment the lines below if you want an isolated Python environment
# -----------------------------------------------------------------------------
# echo "Creating a new virtual environment (.venv)..."
# python -m venv .venv
# source .venv/bin/activate

# -----------------------------------------------------------------------------
# Install main dependencies
# -----------------------------------------------------------------------------
echo "Installing main dependencies from requirements.txt..."
pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Clone and install the evaluation harness
# -----------------------------------------------------------------------------
if [ ! -d "evaluation-pipeline-2024" ]; then
  echo "Cloning evaluation harness repository..."
  git clone -b historical-minimal-pairs https://github.com/sabrinaxinli/evaluation-pipeline-2024.git
else
  echo "Evaluation harness repository already exists. Skipping clone."
fi

echo "Installing the evaluation harness..."
cd evaluation-pipeline-2024
pip install -e .
pip install minicons
pip install --upgrade accelerate
cd ..

# -----------------------------------------------------------------------------
# Wrap-up
# -----------------------------------------------------------------------------
echo "Setup completed successfully!"
