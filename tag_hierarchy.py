import json
import os
import time
import logging
import numpy as np
from utils import load_embedding_cache, save_embedding_cache, get_embeddings, cosine_similarity

# File paths
TAG_IDS_FILE = "tag_ids.json"
HIERARCHY_JSON_FILE = "tag_hierarchy.json"
HIERARCHY_DOT_FILE = "tag_hierarchy.dot"
HIERARCHY_REPORT_FILE = "tag_hierarchy_report.txt"

# Parameters
PARENT_SIMILARITY_THRESHOLD = 0.6  

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

def compute_genericity(sim_matrix):
    """Compute a genericity score as the average similarity with all other tags."""
    n = sim_matrix.shape[0]
    return np.array([(np.sum(sim_matrix[i]) - sim_matrix[i, i]) / (n - 1) for i in range(n)])

def build_hierarchy(tags, embeddings):
    """Build a tag hierarchy using genericity and similarity scores."""
    n = len(tags)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_embeddings = embeddings / (norms + 1e-10)
    
    # Compute cosine similarity matrix
    sim_matrix = np.dot(norm_embeddings, norm_embeddings.T)
    genericity = compute_genericity(sim_matrix)
    logging.info("Genericity scores computed.")

    parents = [None] * n
    for i in range(n):
        candidate_indices = [j for j in range(n) if genericity[j] > genericity[i] and sim_matrix[i, j] >= PARENT_SIMILARITY_THRESHOLD]
        if candidate_indices:
            parents[i] = max(candidate_indices, key=lambda j: sim_matrix[i, j])
        else:
            parents[i] = None

    nodes = [{"id": tags[i]["id"], "tag": tags[i]["tag"], "children": []} for i in range(n)]
    root_nodes = []
    for i in range(n):
        p = parents[i]
        if p is None:
            root_nodes.append(nodes[i])
        else:
            nodes[p]["children"].append(nodes[i])

    return root_nodes, parents, genericity, sim_matrix

def generate_graphviz(hierarchy):
    """Generate a Graphviz DOT file to visualize the hierarchy."""
    lines = ["digraph TagHierarchy {", '    node [shape=box, style=filled, fillcolor=lightgray];']
    
    def traverse(node):
        """Recursively add edges to the graph."""
        for child in node["children"]:
            lines.append(f'    "{node["tag"]}" -> "{child["tag"]}";')
            traverse(child)

    for root in hierarchy:
        traverse(root)
    
    lines.append("}")
    
    with open(HIERARCHY_DOT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logging.info(f"Graphviz DOT file saved to {HIERARCHY_DOT_FILE}")

def generate_report(hierarchy, genericity, tags, sim_matrix, parents):
    """Generate a text report summarizing the hierarchy."""
    lines = ["Tag Hierarchy Report", "====================", ""]
    lines.append("Genericity scores:")
    for i, tag in enumerate(tags):
        lines.append(f"{tag['tag']}: {genericity[i]:.4f}")
    
    lines.append("\nParent assignments (child -> parent):")
    for i, tag in enumerate(tags):
        if parents[i] is not None:
            parent_tag = tags[parents[i]]["tag"]
            similarity = sim_matrix[i, parents[i]]
            lines.append(f"{tag['tag']} -> {parent_tag} (similarity: {similarity:.4f})")
        else:
            lines.append(f"{tag['tag']} -> None (top-level)")

    with open(HIERARCHY_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logging.info(f"Hierarchy report saved to {HIERARCHY_REPORT_FILE}")

def main():
    start_time = time.time()
    logging.info("Starting tag hierarchy computation...")
    
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

    hierarchy, parents, genericity, sim_matrix = build_hierarchy(tags, embeddings)
    logging.info("Hierarchy built successfully.")

    save_json(hierarchy, HIERARCHY_JSON_FILE)
    logging.info(f"Hierarchy saved to {HIERARCHY_JSON_FILE}")

    generate_graphviz(hierarchy)
    generate_report(hierarchy, genericity, tags, sim_matrix, parents)

    logging.info(f"Tag hierarchy computation completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
    save_embedding_cache()
