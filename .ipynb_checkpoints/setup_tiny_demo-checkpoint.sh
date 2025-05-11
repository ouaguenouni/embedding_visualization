#!/bin/bash
# Setup script for extremely tiny word embedding models
# Creates the smallest possible models for GitHub Pages demonstration

set -e  # Exit on error

# Default values
OUTPUT_DIR="github_pages"
VECTOR_SIZE=10
NUM_WORDS=50
PERPLEXITY=5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --vector-size)
      VECTOR_SIZE="$2"
      shift 2
      ;;
    --num-words)
      NUM_WORDS="$2"
      shift 2
      ;;
    --perplexity)
      PERPLEXITY="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --output-dir DIR    Directory to save prepared files (default: github_pages)"
      echo "  --vector-size NUM   Size of embedding vectors (default: 10)"
      echo "  --num-words NUM     Number of words to include (default: 50)"
      echo "  --perplexity NUM    t-SNE perplexity parameter (default: 5)"
      echo "  --help              Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help to see available options"
      exit 1
      ;;
  esac
done

echo "===== Tiny Word Embedding Explorer Setup ====="
echo "Output directory:     $OUTPUT_DIR"
echo "Vector size:          $VECTOR_SIZE"
echo "Words per model:      $NUM_WORDS"
echo "t-SNE perplexity:     $PERPLEXITY"
echo ""

# Check if required files exist
if [ ! -f "prepare_embedding_data.py" ]; then
  echo "Error: prepare_embedding_data.py not found."
  echo "Make sure you're running this script from the directory containing prepare_embedding_data.py."
  exit 1
fi

if [ ! -f "create_tiny_models.py" ]; then
  echo "Error: create_tiny_models.py not found."
  echo "Make sure you're running this script from the directory containing create_tiny_models.py."
  exit 1
fi

# Check if Python and required packages are installed
echo "Checking Python dependencies..."
python3 -c "import gensim, numpy, pandas, sklearn, yaml" || {
  echo "Installing required Python packages..."
  pip install gensim numpy pandas scikit-learn pyyaml
}

# Ensure index.html exists
if [ ! -f "index.html" ]; then
  echo "Error: index.html not found."
  echo "Make sure you have the index.html file in the current directory."
  exit 1
fi

# Create incredibly tiny models
echo "Creating extremely small word embedding models..."
python3 create_tiny_models.py --output_dir "$OUTPUT_DIR" --vector_size "$VECTOR_SIZE" --num_words "$NUM_WORDS" --perplexity "$PERPLEXITY" --clean

echo ""
echo "===== Setup Complete ====="
echo "All files are now in $OUTPUT_DIR"

# Provide GitHub Pages instructions
echo ""
echo "To publish to GitHub Pages:"
echo "1. Create a new repository on GitHub (e.g., word-embedding-explorer)"
echo "2. Run these commands:"
echo "   cd $OUTPUT_DIR"
echo "   git init"
echo "   git add ."
echo "   git commit -m \"Initial commit with tiny word embedding visualization\""
echo "   git remote add origin https://github.com/yourusername/word-embedding-explorer.git"
echo "   git push -u origin main"
echo "3. Go to repository settings on GitHub and enable GitHub Pages"
echo ""
echo "Your visualization will be available at: https://yourusername.github.io/word-embedding-explorer/"