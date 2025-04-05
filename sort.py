import json

def sort_tags_by_id(input_file, output_file):
    """
    Reads a JSON file, sorts the tags by ID, and saves the sorted data to a new file.
    """
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)  # Load the JSON data
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {input_file}")
        return

    # Sort the data by ID
    sorted_data = sorted(data, key=lambda item: item["id"])  # Sort by item[1] (ID)

    # Save the sorted data to a new JSON file
    try:
        with open(output_file, 'w') as f:
            json.dump(sorted_data, f, indent=4)  # Write sorted data with indentation
        print(f"Tags sorted and saved to {output_file}")
    except Exception as e:
        print(f"Error writing to file: {e}")


# Example usage:
input_file = 'tag_ids.json'  # Replace with your input file name
output_file = 'sorted_tags.json' # Replace with your desired output file name
input_tags = []
output_tags = []

sort_tags_by_id(input_file, output_file)