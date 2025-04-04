import os
from flask import Flask, request, jsonify, send_from_directory
import requests
import numpy as np
from bs4 import BeautifulSoup
import json
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections

# Function to load tags from a JSON file
def load_tags_from_json(file_path="tags.json"):
    with open(file_path, "r") as file:
        return json.load(file)

# Load tags at the start of the app
tag_descriptions = load_tags_from_json()

app = Flask(__name__)

# Milvus connection
connections.connect("default", host="localhost", port="19530")

# Define the schema for the Milvus collection
embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=256)  # Adjust dim as needed
id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
collection_schema = CollectionSchema(fields=[id_field, embedding_field], description="Tag Embeddings Collection")

# Create or load the collection
collection_name = "tag_embeddings"
if collection_name not in Collection.list():
    collection = Collection(name=collection_name, schema=collection_schema)
else:
    collection = Collection(name=collection_name)

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

    # Insert embeddings into Milvus collection
    ids = []
    insert_data = [tag_embeddings.tolist()]
    collection.insert(insert_data)

    # Get embedding for content
    content_embedding = get_embeddings([content])[0]

    # Search for similar vectors in Milvus collection
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}  # Adjust parameters as needed
    result = collection.search(
        [content_embedding.tolist()],
        "embedding",
        param=search_params,
        limit=top_n,
        output_fields=["id"]
    )

    most_similar_tags = []
    for res in result[0]:
        tag_idx = res.id  # Retrieve the index of the matching tag
        similarity_score = res.distance  # Cosine similarity (or inner product in this case)
        most_similar_tags.append({
            "tags": tag_descriptions[tag_idx]["tags"],
            "tags_description": tag_descriptions[tag_idx]["tags_description"],
            "similarity_score": similarity_score
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
