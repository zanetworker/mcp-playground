#!/bin/bash

# Run the Streamlit app
cd "$(dirname "$0")"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the app
echo "Starting MCP Streamlit App..."
streamlit run app.py
