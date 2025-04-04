import os
from flask import Flask, request, jsonify, send_from_directory
import requests
import numpy as np
from bs4 import BeautifulSoup
import json

# Function to load tags from a JSON file
def load_tags_from_json(file_path="tags.json"):
    with open(file_path, "r") as file:
        return json.load(file)

# Load tags at the start of the app
tag_descriptions = load_tags_from_json()

app = Flask(__name__)

# Function to get embeddings from Ollama's 'nomic-embed-text' model
def get_embeddings(texts):
    embeddings = []
    for text in texts:
        response = requests.post(
            "http://localhost:11434/api/embed",  # Correct Ollama API endpoint
            json={"model": "nomic-embed-text", "input": text}
        )
        if response.status_code == 200:
            response_json = response.json()
            embeddings.append(np.array(response_json['embeddings'][0]))  # Extract embedding from response
        else:
            print(f"Error in getting embedding for: {text}")
            embeddings.append(np.zeros(256))  # Fallback to a zero vector if error occurs
    return np.array(embeddings)

# Function to extract text from an HTML file or webpage
def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text()
    return text

# Function to calculate cosine similarity using numpy
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# Function to perform similarity search with confidence score and threshold
def find_most_similar_tags(tag_descriptions, content, threshold=0.7, top_n=3):
    # Get embeddings for tags
    tag_embeddings = get_embeddings([desc['tags_description'] for desc in tag_descriptions])

    # Get embedding for content
    content_embedding = get_embeddings([content])[0]

    # Calculate cosine similarity between the content embedding and tag embeddings
    similarities = []
    for tag_embedding in tag_embeddings:
        similarity = cosine_similarity(content_embedding, tag_embedding)
        similarities.append(similarity)

    # Get indices of top N most similar tags, filtered by threshold
    top_indices = [i for i, similarity in enumerate(similarities) if similarity >= threshold]
    
    # If no tags meet the threshold, return empty list or the top N
    if not top_indices:
        top_indices = np.argsort(similarities)[-top_n:][::-1]  # Sort indices by similarity (descending)
    
    most_similar_tags = []
    for idx in top_indices:
        most_similar_tags.append({
            "tags": tag_descriptions[idx]["tags"],  # Add tags (as multiple)
            "tags_description": tag_descriptions[idx]["tags_description"],
            "similarity_score": similarities[idx]  # Add similarity score to the result
        })

    return most_similar_tags


# Web route to handle file upload and URL submission using JSON POST requests
@app.route('/process', methods=['POST'])
def process_content():
    # Get JSON data
    data = request.get_json()
    content = ""
    
    # If the data contains a file, process it
    if 'file' in data:
        file_content = data['file']
        if data.get('file_type') == 'html':
            content = extract_text_from_html(file_content)
        else:
            content = file_content
    elif 'url' in data:
        url = data['url']
        response = requests.get(url)
        if response.status_code == 200:
            content = extract_text_from_html(response.text)
        else:
            return jsonify({"error": "Unable to fetch the URL content."})
    
    # Find the most similar tags
    similar_tags = find_most_similar_tags(tag_descriptions, content, threshold=0.7)

    return jsonify(similar_tags)

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
