import json
import os

# Function to load the data from a JSON file
def load_data_from_json(filename='tags.json'):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: The file {filename} is not a valid JSON file.")
        return []

# Function to load the tags with IDs from tag_ids.json
def load_tags_with_ids(filename='tag_ids.json'):
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Error: The file {filename} is not a valid JSON file.")
            return []
    else:
        return []

# Function to extract and normalize tags with IDs
def extract_and_normalize_tags_with_ids(data, existing_tags):
    all_tags = {tag["tag"]: tag["id"] for tag in existing_tags}  # Load existing tags and IDs into a dictionary
    tag_id_counter = max(all_tags.values(), default=0) + 1  # Set the starting ID for new tags
    
    new_tags = []  # To keep track of new tags that need IDs
    for entry in data:
        tags = entry.get("tags", [])
        for tag in tags:
            normalized_tag = tag.lower()  # Normalize to lowercase
            if normalized_tag not in all_tags:
                all_tags[normalized_tag] = tag_id_counter
                new_tags.append({"tag": normalized_tag, "id": tag_id_counter})
                tag_id_counter += 1  # Increment the ID for the next unique tag

    # Create a sorted list of all tags with their IDs, sorted by the tag name
    all_tags_with_ids = [{"tag": tag, "id": tag_id} for tag, tag_id in all_tags.items()]
    all_tags_with_ids.sort(key=lambda x: x["tag"])  # Sort by tag name for consistency

    return all_tags_with_ids, new_tags

# Function to save the normalized tags back to tags.json
def save_normalized_tags_to_tags_json(data, filename='tags.json'):
    try:
        for entry in data:
            entry["tags"] = [tag.lower() for tag in entry["tags"]]
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    except IOError:
        print(f"Error: Unable to write to {filename}.")

# Function to save tags with IDs to tag_ids.json
def save_tags_to_json(tags_with_ids, filename='tag_ids.json'):
    try:
        with open(filename, 'w') as f:
            json.dump(tags_with_ids, f, indent=4)
    except IOError:
        print(f"Error: Unable to write to {filename}.")

# Main function
def main():
    # Load the original data from tags.json
    data = load_data_from_json('tags.json')

    # Load the existing tags with IDs from tag_ids.json
    existing_tags_with_ids = load_tags_with_ids('tag_ids.json')

    # Extract and normalize tags with IDs
    updated_tags_with_ids, new_tags = extract_and_normalize_tags_with_ids(data, existing_tags_with_ids)

    # Save the normalized tags back to tags.json
    save_normalized_tags_to_tags_json(data)

    # Save the updated tags with IDs to tag_ids.json
    save_tags_to_json(updated_tags_with_ids)

    # Output to verify
    print(f"Normalized tags have been saved to 'tags.json'.")
    print(f"Updated tags with IDs have been saved to 'tag_ids.json'.")
    if new_tags:
        print(f"New tags added with IDs: {new_tags}")
    else:
        print("No new tags were added.")

# Run the main function
if __name__ == "__main__":
    main()
