#!/bin/bash

# Function to check if a Python package is installed
package_installed() {
    python -c "import $1" &> /dev/null
}

# Check if packages are already installed
if package_installed humanfriendly && package_installed jsonpickle && package_installed ultralytics; then
    echo "Packages already installed. Skipping installation."
else
    # Install required Python packages
    echo "Installing required Python packages..."
    pip install humanfriendly jsonpickle ultralytics
fi

# Get the directory of the script
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Define paths and URLs
MODEL_URL="https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt"
MEGADETECTOR_REPO="https://github.com/agentmorris/MegaDetector"
YOLV5_REPO="https://github.com/ultralytics/yolov5"
RELATIVE_DOWNLOAD_DIR="$SCRIPT_DIR/../downloads"
DOWNLOAD_DIR=$(realpath "$RELATIVE_DOWNLOAD_DIR")
echo DOWNLOAD_DIR $DOWNLOAD_DIR
MODEL_PATH="$DOWNLOAD_DIR/md_v5a.0.0.pt"
MEGADETECTOR_DIR="$DOWNLOAD_DIR/MegaDetector"
YOLV5_DIR="$DOWNLOAD_DIR/yolov5"

# Ensure the download directory exists
mkdir -p "$DOWNLOAD_DIR"

# Function to check if a file or directory exists
file_exists() {
    if [ -e "$1" ]; then
        return 0
    else
        return 1
    fi
}

# Download model weights if they don't exist
if ! file_exists "$MODEL_PATH"; then
    echo "Downloading model weights..."
    wget -O "$MODEL_PATH" "$MODEL_URL"
else
    echo "Model weights already exist. Skipping download."
fi

# Clone or fetch MegaDetector repo
if ! file_exists "$MEGADETECTOR_DIR"; then
    echo "Cloning MegaDetector repo..."
    git clone "$MEGADETECTOR_REPO" "$MEGADETECTOR_DIR"
else
    echo "MegaDetector repo already exists, fetching latest..."
    (cd "$MEGADETECTOR_DIR" && git fetch)
fi

# Clone or fetch YOLOv5 repo
if ! file_exists "$YOLV5_DIR"; then
    echo "Cloning YOLOv5 repo..."
    git clone "$YOLV5_REPO" "$YOLV5_DIR"
else
    echo "YOLOv5 repo already exists, fetching latest..."
    (cd "$YOLV5_DIR" && git fetch)
fi

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$MEGADETECTOR_DIR:$YOLV5_DIR
echo "$MEGADETECTOR_DIR:$YOLV5_DIR"
echo "export PYTHONPATH=\$PYTHONPATH:$MEGADETECTOR_DIR:$YOLV5_DIR" >> ~/.bashrc

echo "Setup completed. PYTHONPATH updated."
