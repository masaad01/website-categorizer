# **Website Categorizer**

This is a **Flask-based web service** that categorizes websites by extracting their metadata, computing text embeddings, and matching them against predefined categories using **cosine similarity**.

## **Features**  
✅ Extracts text and metadata from URLs or HTML files  
✅ Uses **machine learning embeddings** to analyze website content  
✅ Uses a **caching system** to avoid redundant processing  
✅ Matches websites to categories with **cosine similarity**  
✅ Handles web scraping challenges like **Cloudflare protection** (still experimental)

![image](https://github.com/user-attachments/assets/119f3508-6fbb-4441-a1b3-8dc8418d3305)

## **How It Works**  
1. **Extract Content**: Parses the website, retrieving metadata (`title`, `description`, `keywords`) and text.  
2. **Compute Embeddings**: Uses the **Ollama ‘nomic-embed-text’** model to generate text embeddings.  
3. **Find Similar Categories**: Compares embeddings against a predefined tag database to find the most relevant categories.  
4. **Return Results**: Responds with the **top matching categories** and similarity scores.  

## **Installation**  
### **Requirements**  
- Python 3.8+  
- **Ollama** installed on your localhost with the 'nomic-embed-text' model.
- `pip` to install dependencies  

### **Setting up Ollama on localhost**  
1. **Install Ollama**  
   Ollama provides an API for running machine learning models locally. Follow these steps to install it:
   - Visit the [Ollama installation page](https://ollama.com) and download the installer for your platform.
   - After downloading, follow the prompts to install Ollama.

2. **Install the 'nomic-embed-text' Model**  
   Once Ollama is installed, you'll need to download the 'nomic-embed-text' model by running the following command:
   ```bash
   ollama pull nomic-embed-text
   ```

3. **Verify the Installation**  
   To check if Ollama and the model were installed successfully, run:
   ```bash
   ollama list
   ```
   This will show the list of models available locally. Ensure 'nomic-embed-text' is listed.

### **Set up the Environment**  
1. Clone the repository and navigate into the project directory:
   ```bash
   git clone https://github.com/your-repo/website-categorizer.git
   cd website-categorizer
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment**:
   - On **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - On **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### **Run the App**  
1. After installing the dependencies, start the Flask application:
   ```bash
   python3 web-numpy.app
   ```

The application should now be running, and you can start using the API.

## **API Usage**  
### **1. Process a URL**  
To categorize a website using a URL, send a POST request to `/process`:

```http
POST /process
Content-Type: application/json
{
    "url": "https://example.com"
}
```

### **2. Process an HTML File**  
To categorize a website by directly providing an HTML file, send a POST request with the HTML content:

```http
POST /process
Content-Type: application/json
{
    "file": "<html>...</html>",
    "file_type": "html"
}
```

### **Response Format**
```json
[
    {
        "tags": ["Technology", "AI"],
        "tags_description": "AI and machine learning news",
        "similarity_score": 0.85
    }
]
```

## **Customization**  
- Edit `tags.json` to modify categories and descriptions.  
- Adjust `threshold` in `find_most_similar_tags()` to control category matching sensitivity.  

## **Contributing**  
Pull requests and improvements are welcome! 🚀
