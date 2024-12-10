#!/bin/bash

# Ensure the script exits immediately if a command fails
set -e

# Generate the requirements.txt file
echo "Generating requirements.txt using pipreqs..."
pipreqs "/Users/joshualum/Documents/pythonprojects/project 2: quiz" --force

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r "/Users/joshualum/Documents/pythonprojects/project 2: quiz/requirements.txt"

# Run the Streamlit app
echo "Running the Streamlit app..."
streamlit run "/Users/joshualum/Documents/pythonprojects/project 2: quiz/streamlit_app.py"
