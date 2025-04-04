import requests
import numpy as np
from bs4 import BeautifulSoup

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
def find_most_similar_tags(tag_descriptions, html_content, threshold=0.7, top_n=3):
    # Get embeddings for tags
    tag_embeddings = get_embeddings([desc['tags_description'] for desc in tag_descriptions])

    # Extract text from the HTML content
    extracted_text = extract_text_from_html(html_content)

    # Get embedding for extracted HTML text
    html_text_embedding = get_embeddings([extracted_text])[0]

    # Calculate cosine similarity between the HTML text embedding and tag embeddings
    similarities = []
    for tag_embedding in tag_embeddings:
        similarity = cosine_similarity(html_text_embedding, tag_embedding)
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

# Example of tags and descriptions
tag_descriptions = [
    {"tags": ["entertainment", "novels"], "tags_description": "Entertainment and novels include books, movies, TV shows, and other forms of storytelling."},
    {"tags": ["news", "social media"], "tags_description": "News and social media cover updates, discussions, and trends in the digital space."},
    {"tags": ["learning", "education", "science"], "tags_description": "Learning and education focus on the acquisition of knowledge, skills, and scientific understanding."},
    {"tags": ["anime", "entertainment"], "tags_description": "Anime is a genre of animation originating from Japan, encompassing various artistic styles and themes."},
    {"tags": ["wiki", "information"], "tags_description": "Wiki refers to websites that allow collaborative editing of content, like Wikipedia."},
    {"tags": ["math", "education"], "tags_description": "Mathematics is the study of numbers, shapes, patterns, and their relationships."},
    {"tags": ["science", "education"], "tags_description": "Science focuses on the study of the physical and natural world through observation and experimentation."}
]

# Example of HTML content to compare with
html_content = """
<html>
<head><title>Introduction to Anime</title></head>
<body>
    <h1>What is Anime?</h1>
    <p>Anime refers to Japanese animation that has become a global phenomenon. It covers various genres such as action, drama, fantasy, and more.</p>
    <p>Many popular anime series have expanded into manga, video games, and other forms of media.</p>
</body>
</html>
"""

# Find the most similar tags
similar_tags = find_most_similar_tags(tag_descriptions, html_content, threshold=0.7)

# Output the most similar tags with similarity scores
print("Most similar tags:")
for tag in similar_tags:
    print(f"Tags: {', '.join(tag['tags'])}, Similarity Score: {tag['similarity_score']:.3f}")

