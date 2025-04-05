import json
import os
import time
import logging
import numpy as np
from utils import (
    load_embedding_cache,
    save_embedding_cache,
    get_embeddings,
    cosine_similarity
)

# Configuration
SIMILARITY_THRESHOLD = 0.7  # Tags must have at least 70% similarity to be assigned
WORD_SIMILARITY_MULTIPLIER = 1.17  # Multiplier for single-word tags
TOP_N = 3  # Number of top tags to assign

# Logging setup
LOG_FILE = "reassign_tags.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # Reduce console spam
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.getLogger().addHandler(console_handler)

# File paths
TAG_IDS_FILE = "tag_ids.json"        
TAGS_FILE = "tags.json"              
OUTPUT_FILE = "tags_reassigned.json"
REPORT_FILE = "reassignment_report.txt"

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

def reassign_tags():
    start_time = time.time()
    logging.info("Starting tag reassignment process...")

    try:
        tag_candidates = load_json(TAG_IDS_FILE)  
        descriptions = load_json(TAGS_FILE)       
    except Exception as e:
        logging.error(f"Error loading JSON files: {e}")
        return

    if not tag_candidates or not descriptions:
        logging.error("One or both input files are empty.")
        return

    logging.info(f"Loaded {len(tag_candidates)} tag candidates and {len(descriptions)} description records.")

    # Get candidate tag names and their embeddings
    candidate_tags = [entry["tag"] for entry in tag_candidates]
    candidate_embeddings = get_embeddings(candidate_tags, show_progress=True)

    reassigned_records = []
    reassignment_details = []

    for idx, record in enumerate(descriptions):
        desc_text = record.get("tags_description", "").strip()
        old_tag = record.get("tags", "")

        if not desc_text:
            continue  # Skip empty descriptions

        try:
            desc_embedding = get_embeddings([desc_text])[0]
        except Exception as e:
            logging.warning(f"Skipping index {idx} due to embedding error: {e}")
            continue

        similarities = np.array([cosine_similarity(desc_embedding, emb) for emb in candidate_embeddings])

        # Apply multiplier for single-word tags
        for i, tag in enumerate(candidate_tags):
            if len(tag.split()) == 1:  # If tag is a single word
                similarities[i] *= WORD_SIMILARITY_MULTIPLIER

        # Get top N tags above the similarity threshold
        sorted_indices = np.argsort(similarities)[::-1]  # Sort in descending order
        top_indices = [i for i in sorted_indices if similarities[i] >= SIMILARITY_THRESHOLD][:TOP_N]

        if len(top_indices) > 0:
            assigned_tags = [{"tag": candidate_tags[i], "similarity": float(similarities[i])} for i in top_indices]
        else:
            # Fallback: Assign the highest similarity tag if none meet the threshold
            best_index = int(np.argmax(similarities))
            assigned_tags = [{"tag": candidate_tags[best_index], "similarity": float(similarities[best_index])}]

        new_record = record.copy()
        new_record["assigned_tags"] = assigned_tags
        reassigned_records.append(new_record)

        reassignment_details.append({
            "index": idx,
            "old_tag": old_tag,
            "assigned_tags": assigned_tags
        })

    save_json(reassigned_records, OUTPUT_FILE)
    logging.info(f"Reassigned records saved to {OUTPUT_FILE}")

    # Generate report
    report_lines = [
        f"Total descriptions processed: {len(descriptions)}",
        f"Records reassigned: {len(reassigned_records)}",
        "",
        "Detailed reassignment per record:"
    ]
    report_lines += [
        f"Index {detail['index']}: Old tag: '{detail['old_tag']}' -> New tags: {', '.join([f'{item['tag']} ({item['similarity']:.4f})' for item in detail['assigned_tags']])}"
        for detail in reassignment_details
    ]

    with open(REPORT_FILE, "w", encoding="utf-8") as rep_f:
        rep_f.write("\n".join(report_lines))

    logging.info(f"Reassignment report saved to {REPORT_FILE}")
    logging.info(f"Tag reassignment completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    load_embedding_cache()
    reassign_tags()
    save_embedding_cache()
