import json
import os
import time
import logging
import numpy as np
from utils import load_embedding_cache, save_embedding_cache, get_embeddings

# File paths
TAG_IDS_FILE = "tag_ids.json"
HIERARCHY_JSON_FILE = "tag_hierarchy4.json"
HIERARCHY_DOT_FILE = "tag_hierarchy4.dot"
HIERARCHY_REPORT_FILE = "tag_hierarchy_report4.txt"

# Parameters
PARENT_SIMILARITY_THRESHOLD = 0.7  # Minimum similarity for parent-child relationships
MAX_PARENTS = 2                     # Maximum number of parents per tag
ONE_WORD_MULTIPLIER = 1.17            # Multiplier for similarity if one or both tags are single words

# Logging setup
LOG_FILE = "tag_hierarchy.log"
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

def is_single_word(tag):
    """Returns True if the tag is a single word (no spaces)."""
    return " " not in tag

def compute_genericity(sim_matrix, threshold=PARENT_SIMILARITY_THRESHOLD):
    """Compute genericity score based on similarity values above the threshold."""
    n = sim_matrix.shape[0]
    genericity = []
    for i in range(n):
        vals = [sim_matrix[i, j] for j in range(n) if j != i and sim_matrix[i, j] >= threshold]
        genericity.append(sum(vals) if vals else 0.0)
    return np.array(genericity)

def build_hierarchy(tags, embeddings):
    """Builds a hierarchical structure of tags with multiple parents."""
    n = len(tags)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_embeddings = embeddings / (norms + 1e-10)

    # Compute cosine similarity matrix
    sim_matrix = np.dot(norm_embeddings, norm_embeddings.T)

    # Adjust similarity if one or both tags are single words
    for i in range(n):
        for j in range(n):
            if i != j:
                if is_single_word(tags[i]["tag"]) or is_single_word(tags[j]["tag"]):
                    sim_matrix[i, j] *= ONE_WORD_MULTIPLIER

    # Compute genericity based on modified similarity
    genericity = compute_genericity(sim_matrix)
    logging.info("Genericity scores computed.")

    # Assign parents
    parents = [[] for _ in range(n)]
    for i in range(n):
        candidate_indices = [
            j for j in range(n)
            if genericity[j] > genericity[i] and sim_matrix[i, j] >= PARENT_SIMILARITY_THRESHOLD
        ]
        if candidate_indices:
            sorted_candidates = sorted(candidate_indices, key=lambda j: sim_matrix[i, j], reverse=True)
            parents[i] = sorted_candidates[:MAX_PARENTS]

    # Build nodes with assigned parents
    nodes = []
    for i in range(n):
        parent_tags = [tags[j]["tag"] for j in parents[i]] if parents[i] else []
        nodes.append({
            "id": tags[i]["id"],
            "tag": tags[i]["tag"],
            "assigned_parents": parent_tags
        })

    return nodes, parents, genericity, sim_matrix

def generate_graphviz(nodes, parents, tags):
    """Generate a Graphviz DOT file to visualize the tag hierarchy."""
    lines = ["digraph TagHierarchy {", '    node [shape=box, style=filled, fillcolor=lightgray];']

    n = len(tags)
    for i in range(n):
        for p in parents[i]:
            lines.append(f'    "{tags[p]["tag"]}" -> "{tags[i]["tag"]}";')

    lines.append("}")
    with open(HIERARCHY_DOT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logging.info(f"Graphviz DOT file saved to {HIERARCHY_DOT_FILE}")

def generate_report(nodes, genericity, tags, sim_matrix, parents):
    """Generate a text report summarizing the tag hierarchy."""
    lines = ["Tag Hierarchy Report", "====================", ""]

    # Genericity scores
    lines.append("Genericity scores:")
    for i, tag in enumerate(tags):
        lines.append(f"{tag['tag']}: {genericity[i]:.4f}")

    # Parent-child relationships
    lines.append("\nParent assignments (child -> parent(s)):")
    for i, tag in enumerate(tags):
        if parents[i]:
            parent_info = ", ".join([f"{tags[j]['tag']} (sim: {sim_matrix[i, j]:.4f})" for j in parents[i]])
            lines.append(f"{tag['tag']} -> {parent_info}")
        else:
            lines.append(f"{tag['tag']} -> None (top-level)")

    with open(HIERARCHY_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logging.info(f"Hierarchy report saved to {HIERARCHY_REPORT_FILE}")

def main():
    """Main function to compute the tag hierarchy and generate reports."""
    start_time = time.time()
    logging.info("Starting tag hierarchy computation with multiple parents...")

    try:
        tag_candidates = load_json(TAG_IDS_FILE)
    except Exception as e:
        logging.error(f"Error loading tag candidates: {e}")
        return

    if not tag_candidates:
        logging.error("No tag candidates found.")
        return

    tags = tag_candidates
    tag_names = [entry["tag"] for entry in tags]
    logging.info(f"Loaded {len(tag_names)} tags.")

    load_embedding_cache()
    embeddings = get_embeddings(tag_names, show_progress=True)
    logging.info("Embeddings computed for all tags.")

    nodes, parents, genericity, sim_matrix = build_hierarchy(tags, embeddings)
    logging.info("Hierarchy (with multiple parents) built successfully.")

    save_json(nodes, HIERARCHY_JSON_FILE)
    logging.info(f"Hierarchy JSON saved to {HIERARCHY_JSON_FILE}")

    generate_graphviz(nodes, parents, tags)
    generate_report(nodes, genericity, tags, sim_matrix, parents)

    logging.info(f"Tag hierarchy computation completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
    save_embedding_cache()
