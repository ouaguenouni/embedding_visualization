#!/usr/bin/env python3
"""
Create Extremely Small Word Embedding Models for Demonstration

This script creates minimal word embedding models for demonstration
purposes only. The models are intentionally kept extremely small
to minimize file size for GitHub Pages deployment.

Modified to work in the root directory without removing existing files.
"""

import os
import argparse
import logging
import gensim
from gensim.models import Word2Vec, FastText, KeyedVectors
import numpy as np
import random
import subprocess
import shutil
import yaml

# Configure logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

def ensure_dir(directory):
    """Make sure a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_tiny_corpus():
    """Create a tiny text corpus for training word embeddings."""
    # Very small set of sentences focused on a few topics
    corpus = [
        ["word", "embeddings", "are", "vector", "representations", "of", "words"],
        ["vectors", "capture", "semantic", "relationships", "between", "words"],
        ["similar", "words", "appear", "close", "together", "in", "vector", "space"],
        ["king", "minus", "man", "plus", "woman", "equals", "queen"],
        ["paris", "is", "the", "capital", "of", "france"],
        ["berlin", "is", "the", "capital", "of", "germany"],
        ["rome", "is", "the", "capital", "of", "italy"],
        ["madrid", "is", "the", "capital", "of", "spain"],
        ["dog", "cat", "fish", "bird", "are", "animals"],
        ["apple", "orange", "banana", "grape", "are", "fruits"],
        ["red", "blue", "green", "yellow", "are", "colors"],
        ["computer", "keyboard", "mouse", "monitor", "are", "technology"],
    ]
    return corpus

def create_tiny_word2vec(output_dir, vector_size=10):
    """Create a tiny Word2Vec model."""
    corpus = create_tiny_corpus()
    
    logging.info(f"Creating tiny Word2Vec model with dimension {vector_size}...")
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,  # Extremely small dimension
        window=3,
        min_count=1,              # Include all words
        workers=4,
        epochs=20                 # Train more on small data
    )
    
    model_path = os.path.join(output_dir, "tiny_word2vec.model")
    model.save(model_path)
    
    # Also save as KeyVectors for smaller size
    kv_path = os.path.join(output_dir, "tiny_word2vec.kv")
    model.wv.save(kv_path)
    
    logging.info(f"Created tiny Word2Vec model with {len(model.wv)} words and {vector_size} dimensions")
    logging.info(f"Saved to {kv_path}")
    
    return kv_path

def create_tiny_fasttext(output_dir, vector_size=10):
    """Create a tiny FastText model."""
    corpus = create_tiny_corpus()
    
    logging.info(f"Creating tiny FastText model with dimension {vector_size}...")
    model = FastText(
        sentences=corpus,
        vector_size=vector_size,  # Extremely small dimension
        window=3,
        min_count=1,              # Include all words
        workers=4,
        epochs=20                 # Train more on small data
    )
    
    model_path = os.path.join(output_dir, "tiny_fasttext.model")
    model.save(model_path)
    
    # Also save as KeyVectors for smaller size
    kv_path = os.path.join(output_dir, "tiny_fasttext.kv")
    model.wv.save(kv_path)
    
    logging.info(f"Created tiny FastText model with {len(model.wv)} words and {vector_size} dimensions")
    logging.info(f"Saved to {kv_path}")
    
    return kv_path

def create_custom_embedding(output_dir, vector_size=10, vocab_size=100):
    """Create a custom embedding with predefined vectors."""
    words = []
    vectors = []
    
    # Create some meaningful clusters for visualization
    word_clusters = [
        # Numbers
        ["one", "two", "three", "four", "five"],
        # Animals
        ["dog", "cat", "bird", "fish", "horse"],
        # Colors
        ["red", "blue", "green", "yellow", "purple"],
        # Countries
        ["usa", "canada", "france", "germany", "japan"],
        # Fruits
        ["apple", "banana", "orange", "grape", "cherry"],
        # Tech
        ["computer", "internet", "software", "hardware", "data"],
        # Common verbs
        ["run", "walk", "jump", "swim", "fly"],
    ]
    
    # Create each cluster with similar vectors plus some noise
    for i, cluster in enumerate(word_clusters):
        # Create a base vector for this cluster
        base = np.zeros(vector_size)
        base[i % vector_size] = 1.0  # Set one dimension high
        
        for word in cluster:
            # Add the word to our vocabulary
            words.append(word)
            
            # Create a vector similar to the base with some noise
            noise = np.random.normal(0, 0.1, vector_size)
            vec = base + noise
            vec = vec / np.linalg.norm(vec)  # Normalize
            vectors.append(vec)
    
    # Create the model directly
    logging.info(f"Creating custom embedding with {len(words)} words and {vector_size} dimensions...")
    wv = KeyedVectors(vector_size)
    wv.add_vectors(words, vectors)
    
    kv_path = os.path.join(output_dir, "tiny_custom.kv")
    wv.save(kv_path)
    
    logging.info(f"Created custom embedding with clusters for better visualization")
    logging.info(f"Saved to {kv_path}")
    
    return kv_path

def prepare_model_for_visualization(model_path, output_dir, num_words, perplexity):
    """Prepare a model for visualization by running the prepare_embedding_data script."""
    # Get base filename and model name
    model_filename = os.path.basename(model_path)
    model_name_base = os.path.splitext(model_filename)[0]
    
    # Define output JSON path
    json_path = os.path.join(output_dir, f"{model_name_base}.json")
    
    # Prepare model name and description
    model_name = model_name_base.replace('_', ' ').title()
    model_description = f"Tiny demo model ({model_name})"
    
    logging.info(f"Preparing {model_name} for visualization...")
    
    # Run the prepare_embedding_data.py script
    cmd = [
        "python3", "prepare_embedding_data.py",
        model_path, json_path,
        "--num_words", str(num_words),
        "--perplexity", str(perplexity),
        "--model_name", model_name,
        "--description", model_description
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logging.info(f"Successfully prepared {model_name} for visualization: {json_path}")
        return True, json_path, model_name
    except subprocess.CalledProcessError as e:
        logging.error(f"Error preparing model for visualization: {e}")
        return False, None, None

def load_config(config_path):
    """Load the config file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if config is None:
                config = {}
            if 'models' not in config:
                config['models'] = []
        return config
    except FileNotFoundError:
        # Create default config
        return {
            'ignore': [
                'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as',
                'what', 'which', 'who', 'this', 'that', 'these', 'those',
                'is', 'are', 'was', 'were', 'be', 'have', 'has', 'had',
                'do', 'does', 'did', 'to', 'at', 'by', 'for', 'with'
            ],
            'models': []
        }

def update_config(config_path, models):
    """Update config.yaml with models information without removing existing models."""
    # Load existing config
    config = load_config(config_path)
    
    # Keep track of which model files are already in the config
    existing_files = [model.get('file', '') for model in config.get('models', [])]
    
    # Add each new model to the config, avoiding duplicates
    for json_path, name in models:
        if json_path and name:
            filename = os.path.basename(json_path)
            
            # Check if this file is already in the config
            if filename not in existing_files:
                config['models'].append({
                    'name': name,
                    'file': filename,
                    'description': f"Tiny demo model for visualization ({name})"
                })
                logging.info(f"Added {name} to config.yaml")
            else:
                logging.info(f"Model {filename} already in config.yaml")
    
    # Save the config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logging.info(f"Updated config file with models")

def main():
    parser = argparse.ArgumentParser(description="Create tiny word embedding models for demonstration")
    parser.add_argument("--output_dir", default=".", help="Directory to save models")
    parser.add_argument("--vector_size", type=int, default=10, help="Size of embedding vectors (smaller = smaller files)")
    parser.add_argument("--num_words", type=int, default=50, help="Number of words to include in visualization")
    parser.add_argument("--perplexity", type=int, default=5, help="t-SNE perplexity parameter")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    ensure_dir(args.output_dir)
    
    # Create models directory
    models_dir = os.path.join(args.output_dir, "demo_models")
    ensure_dir(models_dir)
    
    # Create three tiny models
    logging.info("Creating extremely small word embedding models...")
    
    model_paths = []
    
    # 1. Tiny Word2Vec
    w2v_path = create_tiny_word2vec(models_dir, vector_size=args.vector_size)
    model_paths.append(w2v_path)
    
    # 2. Tiny FastText
    ft_path = create_tiny_fasttext(models_dir, vector_size=args.vector_size)
    model_paths.append(ft_path)
    
    # 3. Custom embedding with predefined clusters
    custom_path = create_custom_embedding(models_dir, vector_size=args.vector_size)
    model_paths.append(custom_path)
    
    # Prepare models for visualization
    prepared_models = []
    
    for model_path in model_paths:
        success, json_path, model_name = prepare_model_for_visualization(
            model_path, 
            args.output_dir, 
            args.num_words, 
            args.perplexity
        )
        
        if success:
            prepared_models.append((json_path, model_name))
    
    # Update config file
    config_path = os.path.join(args.output_dir, "config.yaml")
    update_config(config_path, prepared_models)
    
    # Print file sizes
    logging.info("\nFile sizes:")
    for model_name in ["tiny_word2vec.json", "tiny_fasttext.json", "tiny_custom.json"]:
        file_path = os.path.join(args.output_dir, model_name)
        if os.path.isfile(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            logging.info(f"  {model_name}: {size_kb:.1f} KB")
    
    logging.info("\nSetup complete! All files are now in: " + os.path.abspath(args.output_dir))
    logging.info("Your visualization is ready. Open index.html in your browser to test it locally.")
    logging.info("For GitHub Pages, push these files to your repository and enable GitHub Pages in the settings.")

if __name__ == "__main__":
    main()