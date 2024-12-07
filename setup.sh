#!/bin/bash

# Exit on error
set -e

echo "Starting MDLM setup..."

# Skip Miniconda installation if already exists
if [ ! -d "$HOME/miniconda3" ]; then
    # Download Miniconda installer
    echo "Downloading Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

    # Install Miniconda (with -b flag for batch mode, -p for prefix)
    echo "Installing Miniconda..."
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
else
    echo "Miniconda installation found, skipping download and install..."
fi

# Initialize conda for bash
echo "Initializing conda..."
/root/miniconda3/bin/conda init bash

# Source bashrc to get conda working in current shell
echo "Sourcing .bashrc..."
source ~/.bashrc

# Create conda environment if it doesn't exist
if ! /root/miniconda3/bin/conda env list | grep -q "^mdlm "; then
    echo "Creating conda environment..."
    /root/miniconda3/bin/conda env create -f requirements.yaml
else
    echo "Conda environment 'mdlm' already exists, skipping creation..."
fi

# Activate environment and install pip dependencies
echo "Installing pip dependencies..."
source /root/miniconda3/bin/activate mdlm
pip install -r requirements-pip.txt

# Clean up installer
echo "Cleaning up..."
rm Miniconda3-latest-Linux-x86_64.sh

echo "Setup complete! Please run 'conda activate mdlm' to start using the environment."