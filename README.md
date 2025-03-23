```markdown
# 📚 Document Similarity Search API

A RESTful API that allows users to search for the top N most similar document segments from a dataset using **sentence embeddings** and **FAISS-based vector similarity search**. Built using **FastAPI** and **Hugging Face sentence transformers**, this solution demonstrates the integration of modern ML techniques with scalable backend architecture.

---

## 🚀 Project Overview

The goal of this project is to build a document similarity search system that can take a user query and return the most relevant document segments from a preprocessed corpus. It uses semantic embeddings to represent documents and an efficient vector search algorithm (FAISS) for fast retrieval.

---

## 🧠 Features

- Loads the **20 Newsgroups dataset** (5 sample documents by default)
- Preprocesses and **chunks** each document into overlapping 500-character segments
- Generates **semantic vector embeddings** using `all-MiniLM-L6-v2` from `sentence-transformers`
- Uses **FAISS IndexIVFPQ** for efficient Approximate Nearest Neighbor (ANN) search
- RESTful API with endpoints to:
  - `/api/search?q=<query>&top_k=<int>`: Return top-k similar document chunks
  - `/api/add`: Add new documents in real time (real-time indexing)
- Supports basic **text cleaning**: tokenization, stopword removal, lemmatization

---

## 🧰 Tech Stack

| Component         | Tool/Library                               |
|------------------|--------------------------------------------|
| Language         | Python 3                                    |
| Backend Framework| FastAPI                                     |
| Embeddings       | Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Search    | FAISS (IndexIVFPQ for ANN)                  |
| Dataset          | scikit-learn's 20 Newsgroups                |
| Text Processing  | NLTK                                        |

---

## 📁 Folder Structure

.
├── main.py               # Main FastAPI app
├── README.md             # Project documentation
├── requirements.txt      # Dependencies
└── nltk_data/            # NLTK corpora (for Railway/local deploy)

---

## 📊 Dataset & Preprocessing

### 📦 20 Newsgroups Dataset

This repository utilizes the **20 Newsgroups** dataset, a widely adopted benchmark for text classification and natural language processing tasks. The dataset comprises approximately 20,000 newsgroup posts, divided evenly among 20 different topics.

#### Overview

The 20 Newsgroups dataset provides a rich resource of textual data extracted from newsgroup posts. Each post contains unstructured text data that can include headers, the body of the message, and sometimes footers. This raw text data is typically pre-processed to remove noise (like headers or quotes) before being used in machine learning models.

#### Dataset Structure

**1. Data (Text)**  
- A list of raw text documents, where each document corresponds to a newsgroup post.

**2. Target Labels**  
- An array of integer labels. Each integer (0–19) represents a specific newsgroup category.

**3. Target Names**  
- A list of the 20 newsgroup categories corresponding to the labels:
  - `alt.atheism`
  - `comp.graphics`
  - `comp.os.ms-windows.misc`
  - `comp.sys.ibm.pc.hardware`
  - `comp.sys.mac.hardware`
  - `comp.windows.x`
  - `misc.forsale`
  - `rec.autos`
  - `rec.motorcycles`
  - `rec.sport.baseball`
  - `rec.sport.hockey`
  - `sci.crypt`
  - `sci.electronics`
  - `sci.med`
  - `sci.space`
  - `soc.religion.christian`
  - `talk.politics.guns`
  - `talk.politics.mideast`
  - `talk.politics.misc`
  - `talk.religion.misc`

**4. Additional Attributes**  
- `filenames`: File paths or document identifiers (optional)  
- `DESCR`: A description of dataset structure, origin, and typical use cases

#### How to Load the Dataset

```python
from sklearn.datasets import fetch_20newsgroups

# Load the training subset, removing headers, footers, and quotes for cleaner text data
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

# Display the number of training documents and available categories
print("Number of training documents:", len(newsgroups_train.data))
print("Categories:", newsgroups_train.target_names)
```

---

## 🔍 Embedding & Vector Search

- **Model Used:** `all-MiniLM-L6-v2`, a lightweight transformer that balances performance and speed.
- **Embedding Format:** Each chunk is converted to a 384-dimensional dense vector.
- **FAISS Index:** `IndexIVFPQ` with the following configuration:
  - `nlist` = 50 (number of clusters)
  - `m` = 8 (number of subquantizers)
  - `nprobe` = 10 (clusters to search over)

This structure ensures fast and memory-efficient approximate search for large embedding sets.

---

## 🧪 API Usage

### 🔎 GET `/api/search`

Returns the top-k similar document chunks for a given query.

**Parameters:**
- `q` (str): User search query (required)
- `top_k` (int): Number of similar chunks to return (default: 5)

**Example:**
```http
GET /api/search?q=space shuttle&top_k=3
```

**Response:**
```json
{
  "query": "space shuttle",
  "top_k": 3,
  "results": [
    "Space shuttle launch occurred on...",
    "NASA engineers prepared the...",
    "Astronauts discussed the mission..."
  ]
}
```

---

### 🆕 POST `/api/add`

Adds a new document to the database with real-time indexing.

**Payload:**
```json
{
  "text": "Your full document text here..."
}
```

**Response:**
```json
{
  "message": "Document added successfully.",
  "new_chunks": 3,
  "total_chunks": 25
}
```

---

## 💻 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/doc-similarity-api.git
cd doc-similarity-api
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

Make sure `faiss-cpu` and `sentence-transformers` are installed properly.

### 3. Run the Server

```bash
uvicorn main:app --reload
```

Server will be live at: `http://127.0.0.1:8000`

---

## 🔬 Testing the API

You can test the endpoints directly via:
- Swagger UI: `http://127.0.0.1:8000/docs`
- Curl/Postman for manual API calls

---

## 🎁 Bonus Features Implemented

- ✅ **Real-time indexing** – New documents can be dynamically added using `/api/add`
- ⚙️ **Support for multiple similarity metrics** – (Not yet implemented, but extensible)

---

## 🛠️ Possible Extensions

- Add support for cosine similarity, dot product scoring
- Use persistent FAISS index (save/load from disk)
- Add Dockerfile for containerized deployment
- Frontend UI for interacting with the search API
- Support for larger document sets and pagination

---

## 📚 References

- [Hugging Face Sentence Transformers](https://www.sbert.net/)
- [FAISS by Facebook AI](https://github.com/facebookresearch/faiss)
- [20 Newsgroups Dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

## ✍️ Author

**Meet Lalwani**  
Final Project Submission – Document Similarity API

---
```

---

Let me know if you'd like:

- A `requirements.txt` auto-generated from your code
- This README saved as a `.md` file for download
- A clean zipped folder structure for uploading

Want me to prep all that as a submission pack?




# 📚 Document Similarity Search API

A RESTful API that allows users to search for the top N most similar document segments from a dataset using **sentence embeddings** and **FAISS-based vector similarity search**.

---

## 🚀 Project Overview

The goal of this project is to build a document similarity search system that can take a user query and return the most relevant document segments from a preprocessed corpus. It uses semantic embeddings to represent documents and an efficient vector search algorithm (FAISS) for fast retrieval.

---

## 🧠 Features

- Loads the **20 Newsgroups dataset** (5 sample documents by default)
- Preprocesses and **chunks** each document into overlapping 500-character segments
- Generates **semantic vector embeddings** using `all-MiniLM-L6-v2` from `sentence-transformers`
- Uses **FAISS IndexIVFPQ** for efficient Approximate Nearest Neighbor (ANN) search
- RESTful API with endpoints to:
  - `/api/search?q=<query>&top_k=<int>`: Return top-k similar document chunks
  - `/api/add`: Add new documents in real time (real-time indexing)
- Supports basic **text cleaning**: tokenization, stopword removal, lemmatization

---

## 🧰 Tech Stack

| Component         | Tool/Library                               |
|------------------|--------------------------------------------|
| Language         | Python 3                                    |
| Backend Framework| FastAPI                                     |
| Embeddings       | Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Search    | FAISS (IndexIVFPQ for ANN)                  |
| Dataset          | scikit-learn's 20 Newsgroups                |
| Text Processing  | NLTK                                        |

---

## 📁 Folder Structure

{
  "query": "space shuttle",
  "top_k": 3,
  "results": [
    "Space shuttle launch occurred on...",
    "NASA engineers prepared the...",
    "Astronauts discussed the mission..."
  ]
}

# 20 Newsgroups Dataset 

This repository utilizes the **20 Newsgroups** dataset, a widely adopted benchmark for text classification and natural language processing tasks. The dataset comprises approximately 20,000 newsgroup posts, divided evenly among 20 different topics.
## Overview
The **20 Newsgroups** dataset provides a rich resource of textual data extracted from newsgroup posts. Each post contains unstructured text data that can include headers, the body of the message, and sometimes footers. This raw text data is typically pre-processed to remove noise (like headers or quotes) before being used in machine learning models.
## Dataset Structure
The key components of the dataset are as follows:
### 1. Data (Text)
- **Description:**  
  The `data` attribute is a list of raw text documents, where each document corresponds to a newsgroup post.
### 2. Target Labels
- **Description:**  
  The `target` attribute is an array of integer labels. Each integer (ranging from 0 to 19) represents a specific newsgroup category.
### 3. Target Names
- **Description:**  
  The `target_names` attribute is a list of the 20 newsgroup categories corresponding to the integer labels in `target`.
- **Categories Include:**
  - alt.atheism
  - comp.graphics
  - comp.os.ms-windows.misc
  - comp.sys.ibm.pc.hardware
  - comp.sys.mac.hardware
  - comp.windows.x
  - misc.forsale
  - rec.autos
  - rec.motorcycles
  - rec.sport.baseball
  - rec.sport.hockey
  - sci.crypt
  - sci.electronics
  - sci.med
  - sci.space
  - soc.religion.christian
  - talk.politics.guns
  - talk.politics.mideast
  - talk.politics.misc
  - talk.religion.misc
### 4. Additional Attributes
- **Filenames (Optional):**  
  Some versions of the dataset may include a `filenames` attribute, which provides file paths or unique identifiers for each document.
- **DESCR:**  
  The `DESCR` attribute contains an in-depth description of the dataset, detailing its origin, structure, and typical use cases.

## How to Load the Dataset

You can load the dataset using scikit-learn's `fetch_20newsgroups` function. Here’s a brief example:

```python
from sklearn.datasets import fetch_20newsgroups

# Load the training subset, removing headers, footers, and quotes for cleaner text data
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

# Display the number of training documents and available categories
print("Number of training documents:", len(newsgroups_train.data))
print("Categories:", newsgroups_train.target_names)
```

