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
def find_most_similar_topics(topic_descriptions, html_content, threshold=0.7, top_n=3):
    # Get embeddings for topics and descriptions
    topic_embeddings = get_embeddings([desc['description'] for desc in topic_descriptions])

    # Extract text from the HTML content
    extracted_text = extract_text_from_html(html_content)

    # Get embedding for extracted HTML text
    html_text_embedding = get_embeddings([extracted_text])[0]

    # Calculate cosine similarity between the HTML text embedding and topic embeddings
    similarities = []
    for topic_embedding in topic_embeddings:
        similarity = cosine_similarity(html_text_embedding, topic_embedding)
        similarities.append(similarity)

    # Get indices of top N most similar topics, filtered by threshold
    top_indices = [i for i, similarity in enumerate(similarities) if similarity >= threshold]
    
    # If no topics meet the threshold, return empty list or the top N
    if not top_indices:
        top_indices = np.argsort(similarities)[-top_n:][::-1]  # Sort indices by similarity (descending)
    
    most_similar_topics = []
    for idx in top_indices:
        most_similar_topics.append({
            "topic": topic_descriptions[idx]["topic"],
            "description": topic_descriptions[idx]["description"],
            "similarity_score": similarities[idx]  # Add similarity score to the result
        })

    return most_similar_topics

# Example of topics and descriptions
topic_descriptions = [
    {"topic": "AI in Healthcare", "description": "AI is transforming the healthcare sector with advancements in diagnosis, treatment, and personalized medicine."},
    {"topic": "Quantum Computing", "description": "Quantum computing uses quantum-mechanical phenomena, such as superposition and entanglement, to perform computations."},
    {"topic": "Blockchain", "description": "Blockchain is a decentralized digital ledger technology, which enables secure and transparent transactions."},
    {"topic": "Climate Change", "description": "Climate change refers to long-term shifts in temperature, weather patterns, and environmental conditions caused by human activities."},
    {"topic": "Autonomous Vehicles", "description": "Autonomous vehicles, also known as self-driving cars, are equipped with technology to navigate without human intervention."}
]

# Example of HTML content to compare with
html_content = """
<html>
<head><title>AI and Healthcare</title></head>
<body>
    <h1>Advancements in AI for Healthcare</h1>
    <p>Artificial Intelligence (AI) is revolutionizing healthcare, especially in diagnosis, treatment, and improving patient outcomes. From machine learning models diagnosing diseases to AI-driven robots assisting in surgery, the possibilities are expanding.</p>
    <p>AI also has a huge potential to streamline administrative tasks, predict disease outbreaks, and personalize patient care.</p>
</body>
</html>
"""

# Find the most similar topics
similar_topics = find_most_similar_topics(topic_descriptions, html_content, threshold=0.7)

# Output the most similar topics with similarity scores
print("Most similar topics:")
for topic in similar_topics:
    print(f"Topic: {topic['topic']}, Similarity Score: {topic['similarity_score']:.3f}")
