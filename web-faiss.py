import os
import json
import faiss
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import requests
from bs4 import BeautifulSoup

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

# Function to load tags from a JSON file
def load_tags_from_json(file_path="tags.json"):
    with open(file_path, "r") as file:
        return json.load(file)

# Function to create a FAISS index for fast similarity search
def create_faiss_index(tag_embeddings):
    # Convert the embeddings to float32 (FAISS requires float32 data type)
    tag_embeddings = tag_embeddings.astype(np.float32)
    # Create a FAISS index
    index = faiss.IndexFlatL2(tag_embeddings.shape[1])  # L2 distance (Euclidean distance)
    # Add the embeddings to the index
    index.add(tag_embeddings)
    return index

# Load tags and create the FAISS index at the start
tag_descriptions = load_tags_from_json()
tag_embeddings = get_embeddings([desc['tags_description'] for desc in tag_descriptions])
faiss_index = create_faiss_index(tag_embeddings)

# Function to find most similar tags using FAISS
def find_most_similar_tags(tag_descriptions, content, threshold=0.7, top_n=3):
    # Get embedding for content
    content_embedding = get_embeddings([content])[0].astype(np.float32)
    # Perform the similarity search using FAISS
    _, indices = faiss_index.search(np.array([content_embedding]), top_n)  # Search for top_n nearest neighbors
    
    # Get the most similar tags based on the indices returned by FAISS
    similar_tags = []
    for idx in indices[0]:
        if idx == -1:  # If no valid neighbor is found, FAISS returns -1
            continue
        similar_tags.append({
            "tags": tag_descriptions[idx]["tags"],
            "tags_description": tag_descriptions[idx]["tags_description"],
            "similarity_score": 1 - np.linalg.norm(content_embedding - tag_embeddings[idx]) / np.linalg.norm(content_embedding)
        })
    
    # Filter out tags that have similarity below the threshold
    similar_tags = [tag for tag in similar_tags if tag["similarity_score"] >= threshold]
    
    return similar_tags

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

    # Find the most similar tags using FAISS
    similar_tags = find_most_similar_tags(tag_descriptions, content, threshold=0.7)
    return jsonify(similar_tags)

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)

