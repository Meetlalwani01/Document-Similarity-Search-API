"""
Document Similarity Search API

This FastAPI application loads a public dataset (20 Newsgroups), preprocesses and embeds
the documents using Sentence Transformers, builds a FAISS index for vector similarity search,
and exposes RESTful endpoints to:
  - Search for the top-K similar documents given a query.
  - Add a new document and rebuild the index.

Usage (locally):
  1. Install dependencies (see requirements.txt).
  2. Download necessary NLTK data:
         python -m nltk.downloader stopwords
         python -m nltk.downloader punkt
         python -m nltk.downloader wordnet
  3. Run the app:
         uvicorn main:app --reload

For Railway deployment, Railway will set the PORT environment variable, and the app will be
served on the appropriate port.
"""

import os
import re
import numpy as np
import faiss
import nltk

from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List

from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups

# Download necessary NLTK data if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------
# Global Configuration
# -------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
model = SentenceTransformer(EMBEDDING_MODEL)

# Global storage for documents and embeddings
documents = []          # List of document texts
doc_embeddings = None   # Numpy array for document embeddings
faiss_index = None      # FAISS index for similarity search

# FAISS hyperparameters for Product Quantization (IVFPQ)
NLIST = 50   # Number of clusters/centroids
M = 8        # Number of sub-vectors
NPROBE = 10  # Number of clusters to search at query time

# -------------------------
# Preprocessing Functions
# -------------------------

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def advanced_preprocess(text: str) -> str:
    """
    Preprocess the text:
      1. Lowercase the text.
      2. Tokenize the text.
      3. Remove tokens that are not alphanumeric.
      4. Remove stopwords.
      5. Lemmatize the tokens.
    Returns the cleaned text.
    """
    text = text.lower().strip()
    tokens = word_tokenize(text)
    processed_tokens = []
    
    for token in tokens:
        if not re.match(r"^[a-z0-9]+$", token):
            continue
        if token in stop_words:
            continue
        lemma = lemmatizer.lemmatize(token)
        processed_tokens.append(lemma)
    
    return " ".join(processed_tokens)

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Given a list of texts, preprocess each text and compute its embedding using
    SentenceTransformer. Returns a numpy array of float32 embeddings.
    """
    cleaned_texts = [advanced_preprocess(t) for t in texts]
    embeddings = model.encode(cleaned_texts, show_progress_bar=True)
    return np.array(embeddings).astype("float32")

# -------------------------
# FAISS Index Building
# -------------------------

def build_ivfpq_index(embeddings: np.ndarray, nlist: int = NLIST, m: int = M, nprobe: int = NPROBE) -> faiss.IndexIVFPQ:
    """
    Build and return a FAISS IVFPQ index using the provided embeddings.
    """
    dim = embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)  # 8 bits per sub-vector
    
    print("Training FAISS index...")
    index.train(embeddings)
    print("Adding embeddings to the index...")
    index.add(embeddings)
    index.nprobe = nprobe
    print("FAISS index built successfully!")
    return index

# -------------------------
# Data Loading and Initialization
# -------------------------

def initialize_data_and_index(max_docs: int = 2000):
    """
    Load up to max_docs from the 20 Newsgroups dataset, compute embeddings,
    and build the FAISS index.
    """
    global documents, doc_embeddings, faiss_index
    
    print("Fetching the 20 Newsgroups dataset (train subset)...")
    newsgroups_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    all_docs = newsgroups_data.data
    print(f"Total available documents: {len(all_docs)}")
    
    documents = all_docs[:max_docs]
    print(f"Using {len(documents)} documents for indexing.")
    
    print("Computing embeddings...")
    doc_embeddings = embed_texts(documents)
    
    print("Building FAISS index...")
    faiss_index = build_ivfpq_index(doc_embeddings, nlist=NLIST, m=M, nprobe=NPROBE)
    print("Data initialization complete!")

# Initialize the data and index at startup
initialize_data_and_index(max_docs=2000)

# -------------------------
# FastAPI Application
# -------------------------

app = FastAPI(
    title="Document Similarity Search API",
    description="API for finding similar documents using embeddings and FAISS.",
    version="1.0.0"
)

# Pydantic models for API request/response validation
class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: List[str]

class AddDocRequest(BaseModel):
    text: str

@app.get("/api/search", response_model=SearchResponse)
def search_documents(
    q: str = Query(..., description="User query"),
    top_k: int = Query(5, description="Number of similar documents to retrieve")
):
    """
    Search for similar documents given a query string.
    """
    if not q:
        return {"query": q, "top_k": top_k, "results": []}
    
    print(f"Processing search query: '{q}'")
    query_embedding = embed_texts([q])
    distances, indices = faiss_index.search(query_embedding, top_k)
    result_docs = [documents[idx] for idx in indices[0]]
    
    print(f"Found {len(result_docs)} similar documents.")
    return {"query": q, "top_k": top_k, "results": result_docs}

@app.post("/api/add")
def add_document(request: AddDocRequest):
    """
    Add a new document, update the embeddings, and rebuild the FAISS index.
    """
    global documents, doc_embeddings, faiss_index
    
    new_doc = request.text
    print("Adding new document (first 60 chars):", new_doc[:60])
    documents.append(new_doc)
    
    new_embedding = embed_texts([new_doc])
    if doc_embeddings is None:
        doc_embeddings = new_embedding
    else:
        doc_embeddings = np.concatenate((doc_embeddings, new_embedding), axis=0)
    
    print("Rebuilding FAISS index with new document...")
    faiss_index = build_ivfpq_index(doc_embeddings, nlist=NLIST, m=M, nprobe=NPROBE)
    
    return {"message": "Document added successfully!", "current_doc_count": len(documents)}

# -------------------------
# Running the Server
# -------------------------

if __name__ == "__main__":
    import uvicorn
    # Read the PORT environment variable (Railway sets this automatically), default to 8000 if not set.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
