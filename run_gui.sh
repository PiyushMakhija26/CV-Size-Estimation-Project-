#!/bin/bash

# Vision-Fit macOS/Linux Launcher Script
# Run this file to launch the Vision-Fit GUI application

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         Vision-Fit: AI-Based Body Measurement              ║"
echo "║                  GUI Application                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    echo ""
    echo "   Install with:"
    echo "   - macOS (Homebrew): brew install python3"
    echo "   - Ubuntu/Debian: sudo apt-get install python3 python3-pip"
    echo "   - Fedora: sudo dnf install python3 python3-pip"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

echo "✓ Python detected: $(python3 --version)"
echo ""

# Check if required packages are installed
echo "Checking dependencies..."
python3 -m pip show opencv-python &> /dev/null

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  Missing dependencies"
    echo "Installing required packages..."
    echo ""
    
    pip3 install opencv-python mediapipe numpy pillow
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Failed to install dependencies"
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

echo "✓ Dependencies verified"
echo ""

# Check for Tkinter (may need separate installation on some systems)
python3 -c "import tkinter" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "⚠️  Tkinter not found"
    echo ""
    echo "Installing Tkinter..."
    
    if [[ "$OSTYPE" == "apple-darwin"* ]]; then
        # macOS
        echo "macOS detected - Tkinter should be included with Python"
        echo "If still missing, reinstall Python with: brew install python3"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        distro=$(lsb_release -si 2>/dev/null)
        if [[ "$distro" == "Ubuntu" ]] || [[ "$distro" == "Debian" ]]; then
            sudo apt-get install python3-tk
        elif [[ "$distro" == "Fedora" ]]; then
            sudo dnf install python3-tkinter
        else
            echo "Please install python3-tk for your distribution"
        fi
    fi
fi

# Launch the GUI
echo "Launching Vision-Fit GUI..."
echo ""

python3 app_gui.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Error launching GUI"
    echo "Please check the error message above"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

exit 0