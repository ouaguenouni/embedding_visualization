import json
import yaml
import numpy as np
import pandas as pd
import os
from gensim.models import Word2Vec, KeyedVectors, FastText
from sklearn.manifold import TSNE
import argparse

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        # Return default config if file not found
        return {
            'ignore': ['a', 'an', 'the', 'and', 'or', 'but', 'if'],
            'models': []
        }

def prepare_embedding_data(model_path, output_path, num_words=500, perplexity=30, config_path='config.yaml'):
    """
    Prepares embedding data for visualization on a static website.
    
    Parameters:
    -----------
    model_path : str
        Path to the Gensim .model file
    output_path : str
        Path to save the output JSON file
    num_words : int, default=500
        Number of most frequent words to include in the visualization
    perplexity : int, default=30
        Perplexity parameter for t-SNE
    config_path : str, default='config.yaml'
        Path to the configuration file
    """
    # Load configuration
    config = load_config(config_path)
    words_to_ignore = set(config.get('ignore', []))
    
    print(f"Loading model from {model_path}...")
    print(f"Will ignore {len(words_to_ignore)} stopwords/custom words")
    
    # Try to load the model, handling different possible formats
    try:
        # Try loading as Word2Vec model
        model = Word2Vec.load(model_path)
        wv = model.wv
        print("Loaded as Word2Vec model")
    except:
        try:
            # Try loading as KeyedVectors
            wv = KeyedVectors.load(model_path)
            print("Loaded as KeyedVectors")
        except:
            try:
                # Try loading in word2vec format
                wv = KeyedVectors.load_word2vec_format(model_path, binary=True)
                print("Loaded as word2vec format (binary)")
            except:
                try:
                    # Try loading in word2vec text format
                    wv = KeyedVectors.load_word2vec_format(model_path, binary=False)
                    print("Loaded as word2vec format (text)")
                except:
                    try:
                        # Try loading as FastText
                        model = FastText.load(model_path)
                        wv = model.wv
                        print("Loaded as FastText model")
                    except Exception as e:
                        print(f"Error loading model: {e}")
                        return
    
    # Get words, filtering out words to ignore
    all_words = wv.index_to_key
    filtered_words = [word for word in all_words if word.lower() not in words_to_ignore]
    
    # Limit to num_words
    words = filtered_words[:min(num_words, len(filtered_words))]
    
    print(f"Found {len(all_words)} total words")
    print(f"After filtering: {len(filtered_words)} words")
    print(f"Using {len(words)} most frequent words for visualization")
    
    # Extract vectors for these words
    vectors = np.array([wv[word] for word in words])
    
    # Apply t-SNE to reduce to 2D
    max_perplexity = max(5, min(perplexity, len(words) // 2 - 1))
    tsne_params = {
        'n_components': 2,
        'random_state': 42,
        'perplexity': max_perplexity,
        'n_iter': 1000,
        'init': 'pca'
    }
    
    print(f"Applying t-SNE with perplexity={tsne_params['perplexity']}...")
    tsne = TSNE(**tsne_params)
    vectors_2d = tsne.fit_transform(vectors)
    
    # Get word frequencies if available
    frequencies = []
    if hasattr(wv, 'get_vecattr') and 'count' in wv.expandos:
        frequencies = [int(wv.get_vecattr(word, 'count')) for word in words]
    else:
        # Estimate from vocab rank
        for i, word in enumerate(words):
            # Use Zipf's law: rank ~ 1/frequency
            frequencies.append(num_words - i)
    
    # Find similar words for each word
    similar_words = {}
    for word in words:
        try:
            # Filter out ignored words from similar words results too
            similars = wv.most_similar(word, topn=10)
            filtered_similars = [
                {"word": w, "similarity": float(s)} 
                for w, s in similars 
                if w.lower() not in words_to_ignore
            ][:5]  # Take top 5 after filtering
            similar_words[word] = filtered_similars
        except:
            # Skip if there's an issue finding similar words
            pass
    
    # Prepare data for JSON export
    result = {
        "words": list(words),
        "coordinates": vectors_2d.tolist(),
        "frequencies": [int(f) for f in frequencies],
        "similarWords": similar_words,
        "metadata": {
            "modelType": type(model).__name__ if 'model' in locals() else "Unknown",
            "vectorDimension": wv.vector_size,
            "totalWords": len(all_words),
            "filteredWords": len(filtered_words),
            "displayWords": len(words),
            "tsnePerplexity": max_perplexity
        }
    }
    
    # Save to JSON file using the custom encoder
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    print(f"Embedding data saved to {output_path}")
    print(f"Total file size: {round(os.path.getsize(output_path) / (1024 * 1024), 2)} MB")

def update_config_with_model(output_path, model_name, description):
    """
    Update the config.yaml file with information about the new model
    """
    config_path = 'config.yaml'
    config = load_config(config_path)
    
    # Get the filename from the output path
    filename = os.path.basename(output_path)
    
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
    
    print(f"Updated config.yaml with model information: {model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare word embedding data for web visualization")
    parser.add_argument("model_path", help="Path to the Gensim model file")
    parser.add_argument("output_path", help="Path for the output JSON file")
    parser.add_argument("--num_words", type=int, default=500, help="Number of most frequent words to include")
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity parameter")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--model_name", help="Name of the model to display in the UI")
    parser.add_argument("--description", help="Description of the model to display in the UI")
    
    args = parser.parse_args()
    prepare_embedding_data(args.model_path, args.output_path, args.num_words, args.perplexity, args.config)
    
    # Update config with model information if provided
    if args.model_name:
        description = args.description or f"Model with {args.num_words} words"
        update_config_with_model(args.output_path, args.model_name, description)