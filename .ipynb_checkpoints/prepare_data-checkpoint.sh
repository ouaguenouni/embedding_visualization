#!/bin/bash
# Script to prepare embedding data for visualization

# Check if model file name is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_file.model> [num_words] [perplexity]"
    echo "Example: $0 my_embedding.model 300 30"
    exit 1
fi

MODEL_FILE=$1
NUM_WORDS=${2:-500}  # Default to 500 words if not specified
PERPLEXITY=${3:-30}  # Default to perplexity of 30 if not specified

# Output file name
OUTPUT_FILE="embedding_data.json"

echo "Converting model $MODEL_FILE to JSON data for web visualization..."
echo "Will include $NUM_WORDS most frequent words with perplexity $PERPLEXITY"

# Run the Python conversion script
python3 prepare_embedding_data.py "$MODEL_FILE" "$OUTPUT_FILE" --num_words "$NUM_WORDS" --perplexity "$PERPLEXITY"

if [ $? -eq 0 ]; then
    echo "Conversion complete! The data is ready for web visualization."
    echo "You can now deploy the files to GitHub Pages."
else
    echo "Error occurred during conversion. Please check the error message above."
fi