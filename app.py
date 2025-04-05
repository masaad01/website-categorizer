from flask import Flask, request, jsonify, send_from_directory
import logging
import json
import numpy as np
from utils import (
    load_tags_from_json,
    load_embedding_cache,
    get_embeddings,
    extract_text_from_html,
    find_most_similar_tags,
    get_url_data
)

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Constants
TOP_N = 10
SIMILARITY_THRESHOLD = 0.3

# Load tags and cache at the start of the app
tag_descriptions = load_tags_from_json()
load_embedding_cache()

app = Flask(__name__, static_url_path='/static')

@app.route('/process', methods=['POST'])
def process_content():
    data = request.get_json()
    if not data:
        logger.warning("No data provided in the request.")
        return jsonify({"error": "No data provided."}), 400
    
    content = ""
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
    elif 'url' in data:
        url = data['url']
        content = get_url_data(url)
        if content is None:
            logger.warning(f"Failed to fetch content from URL: {url}")
            return jsonify({"error": "Unable to fetch the URL content."}), 400
    else:
        logger.warning("No file or URL provided.")
        return jsonify({"error": "No file or URL provided."}), 400

    similar_tags = find_most_similar_tags(tag_descriptions, content, SIMILARITY_THRESHOLD, TOP_N)
    return jsonify(similar_tags)

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    logger.info("Precomputing tag embeddings at startup...")
    get_embeddings([desc['tags_description'] for desc in tag_descriptions], save_to_file=True, show_progress=True)
    logger.info("Tag embeddings precomputed successfully.")
    app.run(debug=True)
