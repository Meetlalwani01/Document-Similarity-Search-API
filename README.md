# 🔎 Document Similarity Search API (ML + Backend)


---

## 📁 File Structure

```
.
├── document_similarity_api.ipynb   # Jupyter Notebook with full implementation
├── README.md                       # Project documentation (this file)
└── requirements.txt                # All necessary dependencies
```

---

## 🧠 Project Objective

Build a simple yet powerful backend API that:
- Loads a corpus of documents
- Converts them into vector embeddings
- Stores embeddings in a vector database (FAISS)
- Serves search functionality via a public API using FastAPI and ngrok

---
## 📚 Dataset: **Simple English Wikipedia**
The dataset is sourced from [Hugging Face Datasets](https://huggingface.co/datasets/wikipedia), using the **Simple English Wikipedia** subset.

### 📌 Why this dataset?
### 📊 Dataset Details:
- **Name:** `wikipedia`
- **Subset:** `20220301.simple`
- **Source:** Hugging Face
- **Access Method:**
  ```python
  from datasets import load_dataset
  wiki_data = load_dataset("wikipedia", "20220301.simple", split="train[:1000]")
  ```
- **Fields:** Each record contains a `"text"` field with a full Wikipedia paragraph.
- **Limit:** We load the first `600–1000` articles to keep processing lightweight.

---

## ✅ Solution Walkthrough
### 🔹 Preprocessing
- Lowercasing
- Tokenization (`word_tokenize`)
- Stopword removal (`stopwords.words('english')`)
- Lemmatization (`WordNetLemmatizer`)
- Removes all non-alphanumeric tokens

### 🔹 Embeddings
Uses Hugging Face's transformer model:
- **Model:** `all-MiniLM-L6-v2`
- Fast, accurate sentence-level embeddings
- 384-dimensional vectors
- Converts all documents + queries into dense vector form

### 🔹 Vector Index (FAISS)
- Type: `IndexIVFPQ` for fast, scalable similarity search
- Parameters:
  - `nlist = 50` clusters
  - `m = 8` subquantizers
  - `nprobe = 10` for broader search
- Allows fast `top-k` similarity retrieval in milliseconds

### 🔹 API Layer
- Built using **FastAPI**
- Two main endpoints:
  - `GET /api/search`: Retrieve top-k documents
  - `POST /api/add`: Add a new document dynamically
- Live preview using Swagger UI (`/docs`)

---

## 🚀 Running the API

### 1. 📦 Install Requirements

```bash
pip install fastapi uvicorn faiss-cpu sentence-transformers scikit-learn nltk nest_asyncio pyngrok datasets
```

Also download NLTK corpora:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

### 2. 🌐 Set up ngrok

- Sign up at [https://ngrok.com](https://ngrok.com)
- Copy your **authtoken** from the dashboard
- Paste into your notebook:

```python
from pyngrok import ngrok
ngrok.set_auth_token("your-ngrok-token")
public_url = ngrok.connect(8000)
print("Public URL:", public_url)
```

---

### 3. ⚙️ Start Server

```python
import nest_asyncio
nest_asyncio.apply()
uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

### 4. 🔍 Use Swagger UI

Once the server is running, open:

```
https://<your-ngrok-subdomain>.ngrok.io/docs
```

You’ll see an interactive interface to test:
- `/api/search`
- `/api/add`

---

## 📡 API Reference

### 🔍 `/api/search`

**Method:** `GET`  
**Query Parameters:**
| Name   | Type   | Description |
|--------|--------|-------------|
| `q`    | string | Search query |
| `top_k`| int    | Top results (default 5) |

**Example:**
```
curl "https://<ngrok-url>/api/search?q=quantum computing&top_k=3"
```

**Response:**
```json
{
  "query": "quantum computing",
  "top_k": 3,
  "results": [
    "Quantum computing is based on quantum mechanics...",
    "In quantum physics, computation...",
    "The future of computing involves qubits and entanglement..."
  ]
}
```

---

### ➕ `/api/add`

**Method:** `POST`  
**Payload:**
```json
{
  "text": "New document text on a novel topic."
}
```

**Example:**
```bash
curl -X POST "https://<ngrok-url>/api/add" \
-H "Content-Type: application/json" \
-d '{"text": "This is a new document on deep learning."}'
```

**Response:**
```json
{
  "message": "Document added successfully!",
  "current_doc_count": 601
}
```
