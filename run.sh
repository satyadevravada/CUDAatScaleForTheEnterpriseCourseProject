#!/bin/bash

# Ensure output directory exists
mkdir -p output

# Clean previous log
> output/output.log

# Define all filters
filters=("canny" "sobel" "gauss" "sharpen" "box" "laplacian")

# Loop through all .bmp files in ./data
for input in ./data/*.bmp; do
    filename=$(basename "$input" .bmp)
    for filter in "${filters[@]}"; do
        output_file="./output/${filename}_${filter}.bmp"
        echo "Running $filter on $filename.bmp..." | tee -a output/output.log
        ./nppFilters "$input" "$filter" "$output_file" >> output/output.log 2>&1
    done
done

echo "All filters applied. Output saved in ./output and log in ./output/output.log"
