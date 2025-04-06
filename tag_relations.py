import json
import os
import time
import logging
import numpy as np
from utils import load_embedding_cache, save_embedding_cache, get_embeddings, cosine_similarity

# File paths
TAG_IDS_FILE = "tag_ids.json"
RELATIONS_JSON_FILE = "tag_relations3.json"
RELATIONS_DOT_FILE = "tag_relations4.dot"
RELATIONS_REPORT_FILE = "tag_relations_report3.txt"

# Parameters
SIMILARITY_THRESHOLD = 0.8  # Only connect tags with at least this similarity

# Logging setup
LOG_FILE = "tag_relations.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.getLogger().addHandler(console_handler)

def load_json(file_path):
    """Load JSON data from a file."""
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File {file_path} not found.")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, file_path):
    """Save JSON data to a file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def build_relations(tags, embeddings):
    """Find strong relationships between tags based on similarity."""
    n = len(tags)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_embeddings = embeddings / (norms + 1e-10)

    # Compute cosine similarity matrix
    sim_matrix = np.dot(norm_embeddings, norm_embeddings.T)
    logging.info("Similarity matrix computed.")

    relations = []
    for i in range(n):
        for j in range(i + 1, n):  # Only process each pair once
            similarity = sim_matrix[i, j]
            if similarity >= SIMILARITY_THRESHOLD:
                relations.append({"tag1": tags[i]["tag"], "tag2": tags[j]["tag"], "similarity": float(similarity)})

    return relations

def generate_graphviz(relations):
    """Generate a Graphviz DOT file to visualize tag relationships."""
    lines = ["graph TagRelations {", '    node [shape=box, style=filled, fillcolor=lightgray];']
    
    for rel in relations:
        lines.append(f'    "{rel["tag1"]}" -- "{rel["tag2"]}" [label="{rel["similarity"]:.2f}"];')

    lines.append("}")
    
    with open(RELATIONS_DOT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logging.info(f"Graphviz DOT file saved to {RELATIONS_DOT_FILE}")

def generate_report(relations):
    """Generate a text report summarizing tag relationships."""
    lines = ["Tag Relations Report", "====================", ""]
    for rel in relations:
        lines.append(f'{rel["tag1"]} <--> {rel["tag2"]} (similarity: {rel["similarity"]:.4f})')

    with open(RELATIONS_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logging.info(f"Relations report saved to {RELATIONS_REPORT_FILE}")

def main():
    start_time = time.time()
    logging.info("Starting tag relations computation...")
    
    try:
        tag_candidates = load_json(TAG_IDS_FILE)
    except Exception as e:
        logging.error(f"Error loading tag candidates: {e}")
        return

    tags = tag_candidates
    tag_names = [entry["tag"] for entry in tags]
    logging.info(f"Loaded {len(tag_names)} tags.")

    load_embedding_cache()
    embeddings = get_embeddings(tag_names, show_progress=True)
    logging.info("Embeddings computed for all tags.")

    relations = build_relations(tags, embeddings)
    logging.info(f"Found {len(relations)} strong tag relationships.")

    save_json(relations, RELATIONS_JSON_FILE)
    logging.info(f"Relations saved to {RELATIONS_JSON_FILE}")

    generate_graphviz(relations)
    generate_report(relations)

    logging.info(f"Tag relations computation completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
    save_embedding_cache()

