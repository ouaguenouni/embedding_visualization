#!/usr/bin/env python3
"""
Prepare Pre-trained Word Embedding Models for Visualization

This script downloads pre-trained models from Gensim's API,
processes them for visualization, and outputs JSON files.
No models are saved to the repository, only the prepared JSON files.
"""

import os
import argparse
import logging
import gensim
import gensim.downloader as api
from gensim.models import KeyedVectors
import subprocess
import yaml
import tempfile

# Configure logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

def ensure_dir(directory):
    """Make sure a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def list_available_models():
    """List available pre-trained models in Gensim."""
    logging.info("Available pre-trained models in Gensim:")
    available_models = api.info()['models']
    for model_name in available_models:
        model_info = api.info(model_name)
        logging.info(f"- {model_name}: {model_info.get('description', 'No description')}")
    return available_models

def download_and_prepare_model(model_name, output_dir, num_words, perplexity, download_dir=None):
    """Download a pre-trained model and prepare it for visualization."""
    logging.info(f"Downloading pre-trained model: {model_name}")
    
    # Create temp directory if none specified
    if download_dir is None:
        download_dir = tempfile.mkdtemp()
        logging.info(f"Using temporary directory for downloads: {download_dir}")
    
    try:
        # Download the model
        model = api.load(model_name)
        logging.info(f"Successfully loaded {model_name}")
        
        # Save model to a temporary file
        if hasattr(model, 'wv'):
            # Word2Vec style model with word vectors
            wv = model.wv
        else:
            # Already KeyedVectors
            wv = model
            
        temp_model_path = os.path.join(download_dir, f"{model_name}.kv")
        wv.save(temp_model_path)
        logging.info(f"Saved model temporarily to {temp_model_path}")
        
        # Prepare model name and description
        nice_model_name = model_name.replace('-', ' ').replace('_', ' ').title()
        model_description = f"Pre-trained {nice_model_name} from Gensim"
        
        # Define output JSON path
        json_path = os.path.join(output_dir, f"{model_name.replace('-', '_')}.json")
        
        # Run the prepare_embedding_data.py script
        cmd = [
            "python3", "prepare_embedding_data.py",
            temp_model_path, json_path,
            "--num_words", str(num_words),
            "--perplexity", str(perplexity),
            "--model_name", nice_model_name,
            "--description", model_description
        ]
        
        logging.info(f"Preparing {model_name} for visualization...")
        logging.info(f"Command: {' '.join(cmd)}")
        
        subprocess.run(cmd, check=True)
        logging.info(f"Successfully prepared {model_name} for visualization: {json_path}")
        
        # Add model to config
        update_config_with_model(json_path, nice_model_name, model_description)
        
        return True, json_path, nice_model_name
        
    except Exception as e:
        logging.error(f"Error processing model {model_name}: {e}")
        return False, None, None
    finally:
        # Clean up temp files if using a temp directory
        if download_dir == tempfile.mkdtemp():
            logging.info(f"Cleaning up temporary directory: {download_dir}")
            import shutil
            shutil.rmtree(download_dir, ignore_errors=True)

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            if config is None:
                config = {}
            if 'models' not in config:
                config['models'] = []
        return config
    except FileNotFoundError:
        # Return default config if file not found
        return {
            'ignore': [
                'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as',
                'what', 'which', 'who', 'this', 'that', 'these', 'those',
                'is', 'are', 'was', 'were', 'be', 'have', 'has', 'had',
                'do', 'does', 'did', 'to', 'at', 'by', 'for', 'with'
            ],
            'models': []
        }

def update_config_with_model(json_path, model_name, description):
    """Update the config.yaml file with information about the new model"""
    config_path = 'config.yaml'
    config = load_config(config_path)
    
    # Get the filename from the output path
    filename = os.path.basename(json_path)
    
    # Check if model already exists
    for model in config.get('models', []):
        if model.get('file') == filename:
            # Update existing entry
            model['name'] = model_name
            model['description'] = description
            break
    else:
        # Add new model entry
        if 'models' not in config:
            config['models'] = []
        
        config['models'].append({
            'name': model_name,
            'file': filename,
            'description': description
        })
    
    # Write updated config back to file
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logging.info(f"Updated config.yaml with model information: {model_name}")

def main():
    parser = argparse.ArgumentParser(description="Prepare pre-trained embedding models for visualization")
    parser.add_argument("--output_dir", default=".", help="Directory to save prepared JSON files")
    parser.add_argument("--download_dir", default=None, help="Directory to temporarily save downloaded models")
    parser.add_argument("--num_words", type=int, default=300, help="Number of words to include in visualization")
    parser.add_argument("--perplexity", type=int, default=25, help="t-SNE perplexity parameter")
    parser.add_argument("--list", action="store_true", help="List available pre-trained models and exit")
    parser.add_argument("--models", default="glove-twitter-25,conceptnet-numberbatch-17-06-300,fasttext-wiki-news-subwords-300", 
                       help="Comma-separated list of models to download and prepare")
    args = parser.parse_args()
    
    # List available models if requested
    if args.list:
        list_available_models()
        return
    
    # Create output directory if it doesn't exist
    ensure_dir(args.output_dir)
    
    # Create download directory if specified
    if args.download_dir:
        ensure_dir(args.download_dir)
    
    # Process requested models
    model_names = args.models.split(',')
    logging.info(f"Processing {len(model_names)} models: {', '.join(model_names)}")
    
    successful_models = []
    
    for model_name in model_names:
        model_name = model_name.strip()
        success, json_path, model_nice_name = download_and_prepare_model(
            model_name,
            args.output_dir,
            args.num_words,
            args.perplexity,
            args.download_dir
        )
        
        if success:
            successful_models.append((model_name, json_path))
    
    # Report results
    logging.info(f"\n=== Processing Complete ===")
    logging.info(f"Successfully prepared {len(successful_models)} out of {len(model_names)} models")
    
    if successful_models:
        logging.info(f"\nPrepared models:")
        for model_name, json_path in successful_models:
            logging.info(f"- {model_name}: {os.path.basename(json_path)}")
    
    logging.info("\nThe prepared JSON files are ready to use with the visualization.")
    logging.info("No model files were saved to the repository, only the processed JSON data.")

if __name__ == "__main__":
    main()