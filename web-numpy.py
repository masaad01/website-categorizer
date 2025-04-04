import os
import hashlib
import logging
import requests
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from bs4 import BeautifulSoup
import json

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# File path for persistent cache
CACHE_FILE = 'embedding_cache.json'

# Function to load tags from a JSON file
def load_tags_from_json(file_path="tags.json"):
    with open(file_path, "r") as file:
        return json.load(file)

# Load tags at the start of the app
tag_descriptions = load_tags_from_json()

# Load cache from a JSON file if it exists
def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as file:
                cache = json.load(file)
                # Convert cached embeddings back to numpy arrays
                for key, value in cache.items():
                    cache[key] = np.array(value)
                logger.info("Cache loaded successfully.")
                return cache
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from cache file: {e}")
            logger.info("Cache is corrupted. Starting with an empty cache.")
        except Exception as e:
            logger.error(f"Unexpected error while loading cache: {e}")
    
    logger.info("Cache file doesn't exist or is empty. Initializing a new cache.")
    return {}

# Save cache to a JSON file
def save_cache(cache):
    try:
        # Convert numpy arrays to lists for JSON serialization
        serializable_cache = {
            key: value.tolist() for key, value in cache.items()
        }
        with open(CACHE_FILE, 'w') as file:
            json.dump(serializable_cache, file, indent=4)
        logger.info("Cache saved successfully.")
    except Exception as e:
        logger.error(f"Error saving cache: {e}")

# In-memory cache for tag embeddings, using hash of text as the key
embedding_cache = load_cache()

app = Flask(__name__, static_url_path='/static')

# Function to compute the hash of a text
def compute_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# Function to get embeddings from Ollama's 'nomic-embed-text' model
def get_embeddings(texts):
    embeddings = []
    for text in texts:
        text_hash = compute_hash(text)  # Compute the hash of the tag description

        if text_hash in embedding_cache:
            logger.info(f"Cache hit for hash: {text_hash}")
            # If the hash is in the cache, use the cached embedding
            embeddings.append(embedding_cache[text_hash])
        else:
            logger.info(f"Cache miss for hash: {text_hash}. Requesting embedding from API.")
            # Fetch the embedding from the API and store it in the cache
            try:
                response = requests.post(
                    "http://localhost:11434/api/embed",  # Correct Ollama API endpoint
                    json={"model": "nomic-embed-text", "input": text}
                )
                response.raise_for_status()  # Will raise an exception for 4xx or 5xx responses
                if response.status_code == 200:
                    response_json = response.json()
                    embedding = np.array(response_json['embeddings'][0])
                    embedding_cache[text_hash] = embedding  # Cache the embedding using the hash
                    save_cache(embedding_cache)  # Persist cache to file
                    embeddings.append(embedding)
                else:
                    logger.error(f"Error in getting embedding for: {text}. Status Code: {response.status_code}")
                    embeddings.append(np.zeros(256))  # Fallback to a zero vector if error occurs
            except requests.RequestException as e:
                logger.error(f"Error in embedding request for hash {text_hash}: {e}")
                embeddings.append(np.zeros(256))  # Fallback to a zero vector if request fails
    return np.array(embeddings)

# Function to extract text from an HTML file or webpage
def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()

# Function to calculate cosine similarity using numpy
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# Function to perform similarity search with confidence score and threshold
def find_most_similar_tags(tag_descriptions, content, threshold=0.7, top_n=3):
    # Get embeddings for tag descriptions (calculated once at the start)
    tag_embeddings = get_embeddings([desc['tags_description'] for desc in tag_descriptions])

    # Get embedding for content
    content_embedding = get_embeddings([content])[0]

    # Calculate cosine similarity between the content embedding and tag embeddings
    similarities = [cosine_similarity(content_embedding, tag_embedding) for tag_embedding in tag_embeddings]

    # Filter and get indices of top N most similar tags above the threshold
    top_indices = [i for i, similarity in enumerate(similarities) if similarity >= threshold]
    
    if not top_indices:
        top_indices = np.argsort(similarities)[-top_n:][::-1]  # Sort indices by similarity (descending)

    most_similar_tags = []
    for idx in top_indices:
        most_similar_tags.append({
            "tags": tag_descriptions[idx]["tags"],
            "tags_description": tag_descriptions[idx]["tags_description"],
            "similarity_score": similarities[idx]
        })

    return most_similar_tags

# Web route to handle file upload and URL submission using JSON POST requests
@app.route('/process', methods=['POST'])
def process_content():
    data = request.get_json()

    # Validate input data
    if not data:
        logger.warning("No data provided in the request.")
        return jsonify({"error": "No data provided."}), 400
    
    content = ""

    # Process file content if provided
    if 'file' in data:
        file_content = data['file']
        file_type = data.get('file_type', 'text')
        
        if file_type == 'html':
            content = extract_text_from_html(file_content)
        elif file_type == 'text':
            content = file_content
        else:
            logger.warning(f"Unsupported file type: {file_type}")
            return jsonify({"error": "Unsupported file type."}), 400
    
    # Process URL if provided
    elif 'url' in data:
        url = data['url']
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise for HTTP errors
            content = extract_text_from_html(response.text)
        except requests.RequestException as e:
            logger.error(f"Error fetching URL content: {e}")
            return jsonify({"error": "Unable to fetch the URL content."}), 400
    
    else:
        logger.warning("No file or URL provided.")
        return jsonify({"error": "No file or URL provided."}), 400

    # Find most similar tags
    similar_tags = find_most_similar_tags(tag_descriptions, content, threshold=0.7)
    return jsonify(similar_tags)

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    # Precompute and cache the embeddings for tag descriptions once at startup
    logger.info("Precomputing tag embeddings at startup...")
    get_embeddings([desc['tags_description'] for desc in tag_descriptions])  # Precompute tag embeddings on startup
    logger.info("Tag embeddings precomputed successfully.")
    app.run(debug=True)
