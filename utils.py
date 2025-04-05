import os
import json
import hashlib
import logging
import numpy as np
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Logging setup
logger = logging.getLogger()

# Constants
CACHE_FILE = 'embedding_cache.json'
OLLAMA_API_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text"
TOP_N = 10
SIMILARITY_THRESHOLD = 0.3

# Global cache
embedding_cache = {}

# Load embedding cache from JSON file
def load_embedding_cache():
    global embedding_cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as file:
                cache = json.load(file)
                embedding_cache = {key: np.array(value) for key, value in cache.items()}
                logger.info("Embedding cache loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            embedding_cache = {}
    else:
        embedding_cache = {}

# Save embedding cache to JSON file
def save_embedding_cache():
    try:
        with open(CACHE_FILE, 'w') as file:
            json.dump({key: value.tolist() for key, value in embedding_cache.items()}, file, indent=4)
        logger.info("Embedding cache saved successfully.")
    except Exception as e:
        logger.error(f"Error saving cache: {e}")

# Load tags from JSON file
def load_tags_from_json(file_path="tags.json"):
    with open(file_path, "r") as file:
        return json.load(file)

# Compute hash of a text
def compute_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# Get embeddings from Ollama API
def get_embeddings(texts, save_to_file=False, show_progress=False):
    global embedding_cache
    embeddings = []
    texts_iter = tqdm(texts, desc="Processing embeddings", unit="embedding") if show_progress else texts

    for text in texts_iter:
        text_hash = compute_hash(text)

        if text_hash in embedding_cache:
            embeddings.append(embedding_cache[text_hash])
        else:
            try:
                response = requests.post(OLLAMA_API_URL, json={"model": EMBEDDING_MODEL, "input": text})
                response.raise_for_status()
                embedding = np.array(response.json()['embeddings'][0])
                embedding_cache[text_hash] = embedding
                if save_to_file:
                    save_embedding_cache()
                embeddings.append(embedding)
            except requests.RequestException as e:
                logger.error(f"Error fetching embedding for {text_hash}: {e}")
                embeddings.append(np.zeros(256))  # Fallback

    return np.array(embeddings)

# Extract metadata from HTML
def get_web_page_metadata(soup):
    try:
        return {
            'title': soup.find('title').text if soup.find('title') else None,
            'description': soup.find('meta', attrs={'name': 'description'})['content'] if soup.find('meta', attrs={'name': 'description'}) else None,
            'keywords': soup.find('meta', attrs={'name': 'keywords'})['content'] if soup.find('meta', attrs={'name': 'keywords'}) else None,
        }
    except Exception as e:
        logger.error(f"Metadata extraction failed: {e}")
        return {}

# Extract text from HTML
def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    metadata = get_web_page_metadata(soup)
    metadata_str = " ".join(f"{key}: {value}" for key, value in metadata.items() if value)
    
    return metadata_str if len(metadata_str) > 30 else metadata_str + " " + soup.get_text(separator=" ", strip=True)

# Cosine similarity function
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Find most similar tags
def find_most_similar_tags(tag_descriptions, content, threshold=SIMILARITY_THRESHOLD, top_n=TOP_N):
    global embedding_cache
    tag_embeddings = get_embeddings([desc['tags_description'] for desc in tag_descriptions])
    content_embedding = get_embeddings([content])[0]
    
    similarities = [cosine_similarity(content_embedding, tag_embedding) for tag_embedding in tag_embeddings]
    top_indices = [idx for idx in np.argsort(similarities)[-top_n:][::-1] if similarities[idx] >= threshold]

    if not top_indices:
        return []

    return [{"tags": tag_descriptions[idx]["tags"], "similarity_score": similarities[idx]} for idx in top_indices]

# Fetch URL data with retries
def get_url_data(url):
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Accept-Language': 'en-US,en;q=0.9'
    }

    try:
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return extract_text_from_html(response.text)
    except requests.RequestException as e:
        logger.error(f"Error fetching URL data: {e}")
        return None

