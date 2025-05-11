#!/usr/bin/env python3
"""
Batch Prepare Word Embedding Models

This script takes a directory of word embedding model files and prepares
them all for visualization, storing the results in a 'prepared_models'
directory. The script will automatically update the config.yaml file with
information about all processed models.
"""

import os
import sys
import argparse
import logging
import glob
import yaml
import shutil
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

def load_config(config_path):
    """Load the config YAML file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        return {
            'ignore': [],
            'models': []
        }

def save_config(config, config_path):
    """Save the config YAML file"""
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        logging.info(f"Updated config file at {config_path}")
    except Exception as e:
        logging.error(f"Error saving config file: {e}")

def prepare_model(model_path, output_dir, num_words, perplexity, config_path):
    """Prepare a single model for visualization"""
    # Get base filename for model
    model_filename = os.path.basename(model_path)
    model_name_base = os.path.splitext(model_filename)[0]
    
    # Define output path for JSON
    output_path = os.path.join(output_dir, f"{model_name_base}.json")
    
    # Create model description based on filename
    model_name = model_name_base.replace('_', ' ').title()
    model_description = f"Word embedding model: {model_name}"
    
    # Run the prepare_embedding_data.py script
    cmd = [
        "python3", "prepare_embedding_data.py",
        model_path, output_path,
        "--num_words", str(num_words),
        "--perplexity", str(perplexity),
        "--config", config_path,
        "--model_name", model_name,
        "--description", model_description
    ]
    
    logging.info(f"Processing model: {model_path}")
    logging.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info(f"Successfully prepared model: {output_path}")
        return True, output_path, model_name, model_description
    except subprocess.CalledProcessError as e:
        logging.error(f"Error preparing model {model_path}: {e}")
        logging.error(f"Error output: {e.stderr.decode('utf-8')}")
        return False, None, None, None

def find_model_files(input_dir, extensions=['.model', '.kv', '.bin', '.vec', '.txt']):
    """Find all potential model files in a directory based on extensions"""
    model_files = []
    
    for ext in extensions:
        pattern = os.path.join(input_dir, f"*{ext}")
        model_files.extend(glob.glob(pattern))
    
    return model_files

def main():
    parser = argparse.ArgumentParser(description="Batch prepare word embedding models for visualization")
    parser.add_argument("input_dir", help="Directory containing model files")
    parser.add_argument("--output_dir", default="prepared_models", help="Directory to save prepared JSON files")
    parser.add_argument("--num_words", type=int, default=300, help="Number of words to include in each visualization")
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity parameter")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--extensions", default=".model,.kv,.bin,.vec,.txt", help="Comma-separated list of file extensions to process")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Copy config file to output directory if it doesn't exist there
    output_config_path = os.path.join(args.output_dir, os.path.basename(args.config))
    if not os.path.exists(output_config_path) and os.path.exists(args.config):
        shutil.copy(args.config, output_config_path)
        logging.info(f"Copied config file to {output_config_path}")
    
    # Use the output directory config from now on
    config_path = output_config_path if os.path.exists(output_config_path) else args.config
    
    # Find model files
    extensions = args.extensions.split(',')
    model_files = find_model_files(args.input_dir, extensions)
    
    if not model_files:
        logging.error(f"No model files found in {args.input_dir} with extensions {extensions}")
        sys.exit(1)
    
    logging.info(f"Found {len(model_files)} model files to process")
    
    # Process each model file
    successful_models = []
    
    for model_path in model_files:
        success, output_path, model_name, model_description = prepare_model(
            model_path, 
            args.output_dir, 
            args.num_words, 
            args.perplexity, 
            config_path
        )
        
        if success:
            successful_models.append({
                'source': model_path,
                'output': output_path,
                'name': model_name,
                'description': model_description
            })
    
    # Report results
    logging.info(f"\n=== Preparation Complete ===")
    logging.info(f"Successfully prepared {len(successful_models)} out of {len(model_files)} models")
    
    if successful_models:
        logging.info(f"\nPrepared models:")
        for model in successful_models:
            logging.info(f"- {model['name']}: {os.path.basename(model['output'])}")
    
    # Copy index.html to output directory if it doesn't exist there
    if os.path.exists('index.html') and not os.path.exists(os.path.join(args.output_dir, 'index.html')):
        shutil.copy('index.html', args.output_dir)
        logging.info(f"Copied index.html to {args.output_dir}")
    
    logging.info(f"\nAll files are now in {args.output_dir}")
    logging.info(f"You can serve this directory with: python -m http.server --directory {args.output_dir} 8000")
    logging.info(f"Then visit http://localhost:8000 in your browser")

if __name__ == "__main__":
    main()