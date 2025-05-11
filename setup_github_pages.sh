#!/bin/bash
# Setup script for GitHub Pages with word embedding visualization
# Creates tiny models and keeps all files in the root directory

set -e  # Exit on error

# Default values
VECTOR_SIZE=10
NUM_WORDS=50
PERPLEXITY=5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
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

echo "===== GitHub Pages Word Embedding Setup ====="
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

# Create initial config.yaml if it doesn't exist
if [ ! -f "config.yaml" ]; then
  echo "Creating config.yaml..."
  cat > config.yaml << 'EOL'
# Word Embedding Visualization Configuration

# Words to ignore in the visualization (will be filtered out)
ignore:
  # Common stopwords
  - a
  - an
  - the
  - and
  - or
  - but
  - if
  - because
  - as
  - what
  - which
  - who
  - whom
  - this
  - that
  - these
  - those
  - am
  - is
  - are
  - was
  - were
  - be
  - been
  - being
  - have
  - has
  - had
  - do
  - does
  - did
  - to
  - at
  - by
  - for
  - with
  - about
  - against
  - between
  - into
  - through

# Available embedding models
models: []
EOL
fi

# Create incredibly tiny models in the current directory
echo "Creating extremely small word embedding models..."
python3 create_tiny_models.py --vector_size "$VECTOR_SIZE" --num_words "$NUM_WORDS" --perplexity "$PERPLEXITY"

echo ""
echo "===== Setup Complete ====="
echo "All files are ready in the current directory."

# Check if this is a git repository
if [ -d ".git" ]; then
  echo ""
  echo "This appears to be a git repository. To push changes:"
  echo "  git add ."
  echo "  git commit -m \"Update word embedding visualization\""
  echo "  git push origin main"
  echo ""
  echo "To enable GitHub Pages:"
  echo "1. Go to your repository on GitHub"
  echo "2. Click on Settings"
  echo "3. Scroll down to GitHub Pages"
  echo "4. Under Source, select your main branch"
  echo "5. Click Save"
  echo ""
  echo "Your visualization will be available at: https://yourusername.github.io/yourrepository/"
else
  echo ""
  echo "This is not a git repository. To create one and push to GitHub:"
  echo "  git init"
  echo "  git add ."
  echo "  git commit -m \"Initial commit with word embedding visualization\""
  echo "  git remote add origin https://github.com/yourusername/yourrepository.git"
  echo "  git push -u origin main"
  echo ""
  echo "Then follow the GitHub Pages setup steps above."
fi

echo ""
echo "To test locally, just open index.html in your browser."