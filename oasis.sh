#!/bin/bash

# Define the base directory and output directory
BASE_DIR="./OASIS"
OUTPUT_DIR="./OASIS/oasis_merged"

# Create the output directory if it does not exist
mkdir -p "$OUTPUT_DIR"

# Loop through disc1 to disc12
for disc in {1..12}; do
    # Find all .hdr files in the specific folder structure
    find "$BASE_DIR" -type f -path "*/disc${disc}/*/PROCESSED/MPRAGE/SUBJ_111/*.hdr" | while read -r hdr_file; do
        # Get the corresponding .img file
        img_file="${hdr_file%.hdr}.img"

        # Check if the corresponding .img file exists
        if [ -f "$img_file" ]; then
            # Extract the base name without extension
            base_name=$(basename "$hdr_file" .hdr)

            # Define the output file path
            output_file="$OUTPUT_DIR/${base_name}.nii.gz"

            # Run the conversion using fslchfiletype
            fslchfiletype NIFTI_GZ "$hdr_file" "$output_file"

            # Log the operation
            echo "Converted: $hdr_file to $output_file"
        else
            echo "Missing .img file for: $hdr_file" >&2
        fi
    done
done

# Notify completion
echo "Conversion completed. All .nii.gz files are saved in $OUTPUT_DIR."
