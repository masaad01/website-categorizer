import os
import hashlib
import logging
import requests
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from bs4 import BeautifulSoup
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
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
        logger.debug("Cache saved successfully.")
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
            # If the hash is in the cache, use the cached embedding
            embeddings.append(embedding_cache[text_hash])
        else:
            logger.debug(f"Cache miss for hash: {text_hash}. Requesting embedding from API.")
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


def get_web_page_metadata(soup):
    try:        
        if soup:
            # Check if the soup object is valid
            if not isinstance(soup, BeautifulSoup):
                raise ValueError("Invalid BeautifulSoup object.")
        # Extract metadata
        metadata = {
            'title': soup.find('title').text if soup.find('title') else None,
            'description': soup.find('meta', attrs={'name': 'description'})['content'] if soup.find('meta', attrs={'name': 'description'}) else None,
            'keywords': soup.find('meta', attrs={'name': 'keywords'})['content'] if soup.find('meta', attrs={'name': 'keywords'}) else None,
            'og_title': soup.find('meta', property='og:title')['content'] if soup.find('meta', property='og:title') else None,
            'og_description': soup.find('meta', property='og:description')['content'] if soup.find('meta', property='og:description') else None,
        }
        
        return metadata
    except Exception as e:
        return f"An error occurred: {e}"


# Function to extract text from an HTML file or webpage
def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    metadata = get_web_page_metadata(soup)
    #combine metadata into a string
    metadata_str = " ".join([f"{key}: {value}" for key, value in metadata.items() if value])
    if len(metadata_str) > 30: # metadata_str should be at least 30 characters no be meaningful
        logger.info("Metadata string is long enough, returning metadata.")
        logger.debug(f"Metadata: {metadata_str}")
        return metadata_str
    logger.warning("Metadata string is too short, returning full text content.")
    result = metadata_str + " " + soup.get_text(separator=" ", strip=True)
    logger.debug(f"Extracted text: {result[:100]}...")  # Log the first 100 characters of the result
    return result

# Function to calculate cosine similarity using numpy
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# Function to perform similarity search with confidence score and threshold
def find_most_similar_tags(tag_descriptions, content, threshold=0.3, top_n=3):
    # Get embeddings for tag descriptions (calculated once at the start)
    tag_embeddings = get_embeddings([desc['tags_description'] for desc in tag_descriptions])

    # Get embedding for content
    content_embedding = get_embeddings([content])[0]

    # Calculate cosine similarity between the content embedding and tag embeddings
    similarities = [cosine_similarity(content_embedding, tag_embedding) for tag_embedding in tag_embeddings]

    top_indices = np.argsort(similarities)[-top_n:][::-1]  # Sort indices by similarity (descending)

    # Filter out tags with similarity below the threshold
    top_indices = [idx for idx in top_indices if similarities[idx] >= threshold]
    if not top_indices:
        logger.warning("No tags found with similarity above the threshold.")
        return []

    most_similar_tags = []
    for idx in top_indices:
        most_similar_tags.append({
            "tags": tag_descriptions[idx]["tags"],
            "tags_description": tag_descriptions[idx]["tags_description"],
            "similarity_score": similarities[idx]
        })

    return most_similar_tags

def get_url_data(url):
    # Configure retry strategy (Cloudflare often requires multiple attempts)
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))

    # Rotating headers to mimic different browsers
    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-language': 'en-US,en;q=0.9',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
    }

    try:
        response = session.get(
            url,
            headers=headers,
            timeout=10,
            allow_redirects=True,
            # Consider adding proxies here if needed
            # proxies={'http': 'proxy_ip:port', 'https': 'proxy_ip:port'}
        )

        # Check for Cloudflare challenge
        if response.status_code == 503 and 'cloudflare' in response.text.lower():
            raise requests.RequestException("Cloudflare protection detected")

        response.raise_for_status()
        
        domain = url.split('/')[2]
        path_str = url.split(domain)[-1]
        
        data = " ".join([domain, path_str, extract_text_from_html(response.text)])
        logger.debug(f"Extracted data from {url}: {data[:100]}...")  # Log the first 100 characters of the result
        return data

    except requests.RequestException as e:
        logger.error(f"Bot Block bypass failed: {e}")
        # Fallback to alternative methods
        return None

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
        content = get_url_data(url)
        if content is None:
            logger.warning(f"Failed to fetch content from URL: {url}")
            return jsonify({"error": "Unable to fetch the URL content."}), 400
    else:
        logger.warning("No file or URL provided.")
        return jsonify({"error": "No file or URL provided."}), 400

    # Find most similar tags
    similar_tags = find_most_similar_tags(tag_descriptions, content, threshold=0.4)
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
