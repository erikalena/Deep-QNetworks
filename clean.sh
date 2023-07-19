#!/bin/bash

# Get the folder name from the command line argument
folder_name="$1"

# Check if the folder exists
if [ -d "results/cnn/$folder_name" ]; then
  # Remove the folder
  rm -r "checkpoint/cnn/$folder_name"
  rm "my_job_$folder_name.out"
  rm "my_job_$folder_name.err"
  echo "Folder '$folder_name' removed successfully."
else
  echo "Folder '$folder_name' does not exist."
fi