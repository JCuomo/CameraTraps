#!/bin/bash

# Function to check if a Python package is installed
package_installed() {
    python -c "import $1" &> /dev/null
}

# Check if packages are already installed
if package_installed transformers && package_installed scikit-learn; then
    echo "Packages already installed. Skipping installation."
else
    # Install required Python packages
    echo "Installing required Python packages..."
    pip install transformers scikit-learn
fi