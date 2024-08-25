#!/bin/bash

# Function to check if a package is installed
package_installed() {
    dpkg -s "$1" &> /dev/null
}

# Check if libimage-exiftool-perl is already installed
if package_installed libimage-exiftool-perl; then
    echo "libimage-exiftool-perl is already installed. Skipping installation."
else
    # Update package lists and install libimage-exiftool-perl
    echo "Updating package lists..."
    sudo apt-get update

    echo "Installing libimage-exiftool-perl..."
    sudo apt-get install -y libimage-exiftool-perl

    echo "Installation complete."
fi
