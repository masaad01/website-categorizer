
# **Website Categorizer**

This is a **Flask-based web service** that categorizes websites by extracting their metadata, computing text embeddings, and matching them against predefined categories using **cosine similarity**.  

## **Features**  
✅ Extracts text and metadata from URLs or HTML files  
✅ Uses **machine learning embeddings** to analyze website content  
✅ Uses a **caching system** to avoid redundant processing  
✅ Matches websites to categories with **cosine similarity**  
✅ Handles web scraping challenges like **Cloudflare protection**  *still experimental

![image](https://github.com/user-attachments/assets/119f3508-6fbb-4441-a1b3-8dc8418d3305)


## **How It Works**  
1. **Extract Content**: Parses the website, retrieving metadata (`title`, `description`, `keywords`) and text.  
2. **Compute Embeddings**: Uses the **Ollama ‘nomic-embed-text’** model to generate text embeddings.  
3. **Find Similar Categories**: Compares embeddings against a predefined tag database to find the most relevant categories.  
4. **Return Results**: Responds with the **top matching categories** and similarity scores.  

## **Installation**  
### **Requirements**  
- Python 3.8+  
- `pip install -r requirements.txt`  

### **Run the App**  
```bash
python3 web-numpy.app
```

## **API Usage**  
### **1. Process a URL**  
```http
POST /process
Content-Type: application/json
{
    "url": "https://example.com"
}
```

### **2. Process an HTML File**  
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
