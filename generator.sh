#!/bin/bash

# Check if arguments were provided
if [ $# -lt 2 ]; then
  echo "Usage: $0 <number_of_lines> <line_length>"
  echo "Line length must be a multiple of 8"
  exit 1
fi

NUM_LINES=$1
LINE_LENGTH=$2

# Check if the line length is a multiple of 8
if [ $((LINE_LENGTH % 8)) -ne 0 ]; then
  echo "Error: Line length ($LINE_LENGTH) must be a multiple of 8"
  exit 1
fi

# Create output file
OUTPUT_FILE="file.vals"
> "$OUTPUT_FILE"  # Clear file if it exists

# Generate random hex values
for ((i=1; i<=NUM_LINES; i++)); do
  # Calculate number of bytes needed (each byte produces 2 hex chars)
  bytes_needed=$((LINE_LENGTH / 2))
  
  # Generate random hex characters per line
  hex_line=$(head -c "$bytes_needed" /dev/urandom | xxd -p | tr -d '\n')
  
  # Ensure we have exactly the right length
  hex_line="${hex_line:0:$LINE_LENGTH}"
  
  echo "$hex_line" >> "$OUTPUT_FILE"
done

echo "Created $OUTPUT_FILE with $NUM_LINES lines of $LINE_LENGTH hex characters each"