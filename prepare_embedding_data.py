import json
import numpy as np
import pandas as pd
import os
from gensim.models import Word2Vec, KeyedVectors
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

def prepare_embedding_data(model_path, output_path, num_words=500, perplexity=30):
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
    """
    print(f"Loading model from {model_path}...")
    
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
                except Exception as e:
                    print(f"Error loading model: {e}")
                    return
    
    # Get the most common words (limiting to num_words)
    words = wv.index_to_key[:min(num_words, len(wv.index_to_key))]
    print(f"Using {len(words)} most frequent words")
    
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
        frequencies = [int(wv.get_vecattr(word, 'count')) for word in words]  # Convert np.int64 to Python int
    else:
        # Estimate from vocab rank
        for i, word in enumerate(words):
            # Use Zipf's law: rank ~ 1/frequency
            frequencies.append(num_words - i)
    
    # Find similar words for each word
    similar_words = {}
    for word in words:
        try:
            similars = wv.most_similar(word, topn=5)
            similar_words[word] = [{"word": w, "similarity": float(s)} for w, s in similars]  # Convert np.float32 to Python float
        except:
            # Skip if there's an issue finding similar words
            pass
    
    # Prepare data for JSON export
    result = {
        "words": list(words),  # Convert from KeyedVectors' keys view to list
        "coordinates": vectors_2d.tolist(),
        "frequencies": [int(f) for f in frequencies],  # Ensure all are Python int
        "similarWords": similar_words
    }
    
    # Save to JSON file using the custom encoder
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    print(f"Embedding data saved to {output_path}")
    print(f"Total file size: {round(os.path.getsize(output_path) / (1024 * 1024), 2)} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare word embedding data for web visualization")
    parser.add_argument("model_path", help="Path to the Gensim model file")
    parser.add_argument("output_path", help="Path for the output JSON file")
    parser.add_argument("--num_words", type=int, default=500, help="Number of most frequent words to include")
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity parameter")
    
    args = parser.parse_args()
    prepare_embedding_data(args.model_path, args.output_path, args.num_words, args.perplexity)