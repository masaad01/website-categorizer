import json
import os
import time
import logging
import numpy as np
from utils import load_embedding_cache, save_embedding_cache, get_embeddings, cosine_similarity

# Configuration parameters
PARENT_SIMILARITY_THRESHOLD = 0.6  # Minimum similarity for a parent-child relationship
HIERARCHY_OUTPUT_FILE = "tag_hierarchy.json"
REPORT_FILE = "tag_hierarchy_report.txt"
TAG_IDS_FILE = "tag_ids.json"  # File with candidate tags, e.g.: [{"tag": "entertainment", "id": 1}, {"tag": "novels", "id": 2}, ...]

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
    """Save data as JSON to a file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def compute_genericity(sim_matrix):
    """
    For each tag (row in sim_matrix), compute its genericity score as the mean similarity to all other tags.
    """
    # Exclude self-similarity (which is 1.0) by subtracting 1 and dividing by (n-1)
    n = sim_matrix.shape[0]
    genericity = []
    for i in range(n):
        # Sum similarities excluding self, then average
        total = np.sum(sim_matrix[i]) - sim_matrix[i, i]
        genericity.append(total / (n - 1))
    return np.array(genericity)

def build_hierarchy(tags, embeddings):
    """
    Build a hierarchy based on the following heuristic:
    - Compute a genericity score (average similarity with all others) for each tag.
    - For each tag, find a parent tag among those with a higher genericity score 
      (i.e. more generic) that has the highest similarity (above a threshold).
    - Tags that have no candidate parent become top-level nodes.
    """
    n = len(tags)
    # Normalize embeddings (each row vector)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_embeddings = embeddings / (norms + 1e-10)
    
    # Compute full cosine similarity matrix (n x n)
    sim_matrix = np.dot(norm_embeddings, norm_embeddings.T)
    logging.info("Pairwise similarity matrix computed.")
    
    genericity = compute_genericity(sim_matrix)
    logging.info("Genericity scores computed for each tag.")

    # For each tag, find a candidate parent.
    # A candidate parent must have a higher genericity score and similarity above threshold.
    parents = [None] * n
    for i in range(n):
        candidate_indices = [j for j in range(n) if genericity[j] > genericity[i] and sim_matrix[i, j] >= PARENT_SIMILARITY_THRESHOLD]
        if candidate_indices:
            # Choose the candidate with the highest similarity
            best_parent = max(candidate_indices, key=lambda j: sim_matrix[i, j])
            parents[i] = best_parent
        else:
            parents[i] = None  # No parent, top-level tag

    # Build a tree structure: each tag node will have "children"
    nodes = [{"id": tags[i]["id"], "tag": tags[i]["tag"], "children": []} for i in range(n)]
    root_nodes = []

    for i in range(n):
        p = parents[i]
        if p is None:
            root_nodes.append(nodes[i])
        else:
            nodes[p]["children"].append(nodes[i])

    return root_nodes, parents, genericity, sim_matrix

def generate_report(hierarchy, genericity, tags, sim_matrix, parents):
    """
    Generate a text report summarizing the hierarchy.
    """
    lines = []
    lines.append("Tag Hierarchy Report")
    lines.append("====================")
    lines.append("")
    lines.append("Genericity scores for each tag:")
    for i, tag in enumerate(tags):
        lines.append(f"ID {tag['id']} - {tag['tag']}: Genericity {genericity[i]:.4f}")
    lines.append("")
    lines.append("Parent assignments (child -> parent):")
    for i, tag in enumerate(tags):
        if parents[i] is not None:
            parent_tag = tags[parents[i]]["tag"]
            similarity = sim_matrix[i, parents[i]]
            lines.append(f"{tag['tag']} -> {parent_tag} (similarity: {similarity:.4f})")
        else:
            lines.append(f"{tag['tag']} -> None (top-level)")
    lines.append("")
    lines.append("Hierarchy Tree (Indented):")

    def print_tree(nodes, indent=0):
        for node in nodes:
            lines.append("  " * indent + f"- {node['tag']} (ID {node['id']})")
            if node["children"]:
                print_tree(node["children"], indent+1)

    print_tree(hierarchy)
    return "\n".join(lines)

def main():
    start_time = time.time()
    logging.info("Starting tag hierarchy computation...")
    
    try:
        tag_candidates = load_json(TAG_IDS_FILE)
    except Exception as e:
        logging.error(f"Error loading tag candidates: {e}")
        return

    if not tag_candidates:
        logging.error("No tag candidates found in the input file.")
        return

    # Get list of tags (each entry is a dictionary with "tag" and "id")
    tags = tag_candidates
    tag_names = [entry["tag"] for entry in tags]
    logging.info(f"Loaded {len(tag_names)} tags.")

    # Load embedding cache (if any) and compute embeddings for all tag names
    load_embedding_cache()
    embeddings = get_embeddings(tag_names, show_progress=True)
    logging.info("Embeddings computed for all tags.")

    # Build hierarchy
    hierarchy, parents, genericity, sim_matrix = build_hierarchy(tags, embeddings)
    logging.info("Hierarchy built successfully.")

    # Save hierarchy as JSON
    save_json(hierarchy, HIERARCHY_OUTPUT_FILE)
    logging.info(f"Hierarchy saved to {HIERARCHY_OUTPUT_FILE}")

    # Generate report and save it
    report_text = generate_report(hierarchy, genericity, tags, sim_matrix, parents)
    with open(REPORT_FILE, "w", encoding="utf-8") as rep_f:
        rep_f.write(report_text)
    logging.info(f"Hierarchy report saved to {REPORT_FILE}")
    logging.info(f"Tag hierarchy computation completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
    save_embedding_cache()

