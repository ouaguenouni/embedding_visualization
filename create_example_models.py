"""
Create and prepare various word embedding models for visualization.

This script downloads pre-trained models and also trains simple models 
on sample text to provide a range of embedding examples.
"""

import os
import argparse
import gensim
import gensim.downloader as api
from gensim.models import Word2Vec, FastText, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.corpora import WikiCorpus
import numpy as np
import wget
import zipfile
import tarfile
import tempfile
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import gutenberg, brown
import subprocess
import shutil

# Configure logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

# Make sure required NLTK data is downloaded
try:
    nltk.data.find('corpora/gutenberg')
    nltk.data.find('corpora/brown')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('gutenberg')
    nltk.download('brown')
    nltk.download('punkt')

def ensure_dir(directory):
    """Make sure a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_sample_corpus():
    """Get a sample corpus for training models."""
    # Combine some texts from NLTK's included corpora
    texts = []
    
    # Add some texts from Gutenberg
    for fileid in gutenberg.fileids():
        words = gutenberg.words(fileid)
        texts.append([word.lower() for word in words if word.isalpha()])
    
    # Add some texts from Brown corpus
    for fileid in brown.fileids():
        words = brown.words(fileid)
        texts.append([word.lower() for word in words if word.isalpha()])
    
    print(f"Created a sample corpus with {len(texts)} documents")
    return texts

def create_word2vec_example(output_dir, sample_corpus):
    """Create a Word2Vec example model and prepare it for visualization."""
    model_path = os.path.join(output_dir, "word2vec_sample.model")
    json_path = os.path.join(output_dir, "word2vec_sample.json")
    
    print("Training Word2Vec model on sample corpus...")
    model = Word2Vec(
        sentences=sample_corpus,
        vector_size=100,
        window=5,
        min_count=5,
        workers=4,
        epochs=5
    )
    
    print(f"Saving Word2Vec model to {model_path}")
    model.save(model_path)
    
    print(f"Preparing visualization data for Word2Vec model...")
    cmd = [
        "python", "prepare_embedding_data.py",
        model_path, json_path,
        "--num_words", "300",
        "--model_name", "Word2Vec (Sample)",
        "--description", "Word2Vec model trained on sample text corpus"
    ]
    subprocess.run(cmd)
    
    return model_path, json_path

def create_fasttext_example(output_dir, sample_corpus):
    """Create a FastText example model and prepare it for visualization."""
    model_path = os.path.join(output_dir, "fasttext_sample.model")
    json_path = os.path.join(output_dir, "fasttext_sample.json")
    
    print("Training FastText model on sample corpus...")
    model = FastText(
        sentences=sample_corpus,
        vector_size=100,
        window=5,
        min_count=5,
        workers=4,
        epochs=5
    )
    
    print(f"Saving FastText model to {model_path}")
    model.save(model_path)
    
    print(f"Preparing visualization data for FastText model...")
    cmd = [
        "python", "prepare_embedding_data.py",
        model_path, json_path,
        "--num_words", "300",
        "--model_name", "FastText (Sample)",
        "--description", "FastText model trained on sample text corpus"
    ]
    subprocess.run(cmd)
    
    return model_path, json_path

def get_glove_example(output_dir):
    """Download a pre-trained GloVe model and prepare it for visualization."""
    # We'll use a smaller GloVe model for demonstration
    glove_model_path = os.path.join(output_dir, "glove_twitter_25d.txt")
    word2vec_model_path = os.path.join(output_dir, "glove_twitter_25d.word2vec")
    keyvectors_path = os.path.join(output_dir, "glove_twitter_25d.kv")
    json_path = os.path.join(output_dir, "glove_twitter_25d.json")
    
    # Check if model already exists
    if not os.path.exists(glove_model_path):
        print("Downloading pre-trained GloVe Twitter model...")
        glove_model = api.load("glove-twitter-25")
        glove_model.save_word2vec_format(glove_model_path)
    
    # Convert to word2vec format if not already done
    if not os.path.exists(word2vec_model_path):
        print("Converting GloVe to word2vec format...")
        glove2word2vec(glove_model_path, word2vec_model_path)
    
    # Load and save as KeyVectors for consistency
    if not os.path.exists(keyvectors_path):
        print("Loading GloVe model into KeyVectors...")
        model = KeyedVectors.load_word2vec_format(word2vec_model_path)
        model.save(keyvectors_path)
    
    print(f"Preparing visualization data for GloVe model...")
    cmd = [
        "python", "prepare_embedding_data.py",
        keyvectors_path, json_path,
        "--num_words", "300",
        "--model_name", "GloVe Twitter (25d)",
        "--description", "Pre-trained GloVe model on Twitter data (25 dimensions)"
    ]
    subprocess.run(cmd)
    
    return keyvectors_path, json_path

def get_word2vec_google_news(output_dir):
    """Download the pre-trained Word2Vec Google News model and prepare it for visualization."""
    keyvectors_path = os.path.join(output_dir, "word2vec_google_news.kv")
    json_path = os.path.join(output_dir, "word2vec_google_news.json")
    
    # Try to download a smaller subset of the model to save space and time
    if not os.path.exists(keyvectors_path):
        print("Downloading pre-trained Word2Vec Google News model (this may take a while)...")
        try:
            # Try to download a smaller version from Gensim's API
            model = api.load("word2vec-google-news-300")
            model.save(keyvectors_path)
        except Exception as e:
            print(f"Error downloading model from API: {e}")
            print("Trying alternative method...")
            
            # If that fails, try to download directly (only top 100k words)
            model = api.load("word2vec-google-news-300", limit=100000)
            model.save(keyvectors_path)
    
    print(f"Preparing visualization data for Word2Vec Google News model...")
    cmd = [
        "python", "prepare_embedding_data.py",
        keyvectors_path, json_path,
        "--num_words", "300",
        "--model_name", "Word2Vec (Google News)",
        "--description", "Pre-trained Word2Vec model on Google News corpus (300 dimensions)"
    ]
    subprocess.run(cmd)
    
    return keyvectors_path, json_path

def main():
    parser = argparse.ArgumentParser(description="Create example word embedding models for visualization")
    parser.add_argument("--output_dir", default="models", help="Directory to save models")
    parser.add_argument("--include_google", action="store_true", help="Include Google News model (large download)")
    args = parser.parse_args()
    
    # Create output directory
    ensure_dir(args.output_dir)
    
    # Get sample corpus for training models
    sample_corpus = get_sample_corpus()
    
    # Create and prepare models
    models_created = []
    
    # 1. Word2Vec example
    try:
        model_path, json_path = create_word2vec_example(args.output_dir, sample_corpus)
        models_created.append(("Word2Vec (Sample)", model_path, json_path))
    except Exception as e:
        print(f"Error creating Word2Vec example: {e}")
    
    # 2. FastText example
    try:
        model_path, json_path = create_fasttext_example(args.output_dir, sample_corpus)
        models_created.append(("FastText (Sample)", model_path, json_path))
    except Exception as e:
        print(f"Error creating FastText example: {e}")
    
    # 3. GloVe example
    try:
        model_path, json_path = get_glove_example(args.output_dir)
        models_created.append(("GloVe Twitter", model_path, json_path))
    except Exception as e:
        print(f"Error creating GloVe example: {e}")
    
    # 4. Word2Vec Google News (optional, as it's large)
    if args.include_google:
        try:
            model_path, json_path = get_word2vec_google_news(args.output_dir)
            models_created.append(("Word2Vec Google News", model_path, json_path))
        except Exception as e:
            print(f"Error creating Word2Vec Google News example: {e}")
    
    # Print summary
    print("\n=== Models Created ===")
    for name, model_path, json_path in models_created:
        print(f"{name}:")
        print(f"  - Model: {model_path}")
        print(f"  - JSON for visualization: {json_path}")
    
    print("\nAll example models have been created and prepared for visualization.")
    print("You can now launch the web interface to explore these embeddings.")

if __name__ == "__main__":
    main()