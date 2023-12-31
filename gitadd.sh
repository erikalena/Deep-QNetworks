#!/bin/bash

# Get the folder name from the command line argument
folder_name="$1"

# Check if the folder exists
if [ -d "results/cnn/$folder_name" ]; then
  # add to git all files in the folder
    git add "results/cnn/$folder_name/*"
else
  echo "Folder '$folder_name' does not exist."
fi