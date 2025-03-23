import os
import re
import numpy as np
import faiss
import nltk
import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data to local directory (for Railway)
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_path)
for res in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(res)
    except LookupError:
        nltk.download(res, download_dir=nltk_data_path)

# Load NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
token_pattern = re.compile(r"^[a-z0-9]+$")

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
model = SentenceTransformer(EMBEDDING_MODEL)

# Simple text splitter function
def split_document(text: str, chunk_size=500, chunk_overlap=50) -> List[str]:
    chunks = []
    length = len(text)
    for i in range(0, length, chunk_size - chunk_overlap):
        end = min(i + chunk_size, length)
        chunks.append(text[i:end])
    return chunks

# Global vars
doc_chunks = []
chunk_embeddings = None
faiss_index = None

# FAISS config
nlist = 50
m = 8
nprobe = 10

def advanced_preprocess(text: str) -> str:
    text = text.lower().strip()
    tokens = word_tokenize(text)
    cleaned = []
    for token in tokens:
        if token in stop_words: continue
        if not token_pattern.match(token): continue
        lemma = lemmatizer.lemmatize(token)
        cleaned.append(lemma)
    return " ".join(cleaned)

def embed_texts(texts: List[str]) -> np.ndarray:
    cleaned = [advanced_preprocess(t) for t in texts]
    return model.encode(cleaned, show_progress_bar=False).astype("float32")

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexIVFPQ:
    dim = embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = nprobe
    return index

def initialize_dataset(max_docs=5):
    global doc_chunks, chunk_embeddings, faiss_index
    data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    raw_docs = data.data[:max_docs]
    for doc in raw_docs:
        doc_chunks.extend(split_document(doc))
    chunk_embeddings = embed_texts(doc_chunks)
    faiss_index = build_faiss_index(chunk_embeddings)

# Initialize on startup
initialize_dataset()

# -------------------- FastAPI Setup --------------------
class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: List[str]

class AddDocRequest(BaseModel):
    text: str

app = FastAPI(title="Document Similarity API")

@app.get("/api/search", response_model=SearchResponse)
def search_documents(q: str = Query(...), top_k: int = Query(5)):
    query_emb = embed_texts([q])
    distances, indices = faiss_index.search(query_emb, top_k)
    results = [doc_chunks[i] for i in indices[0] if i < len(doc_chunks)]
    return {"query": q, "top_k": top_k, "results": results}

@app.post("/api/add")
def add_document(req: AddDocRequest):
    global doc_chunks, chunk_embeddings, faiss_index
    new_chunks = split_document(req.text)
    doc_chunks.extend(new_chunks)
    new_embeddings = embed_texts(new_chunks)
    chunk_embeddings = np.concatenate((chunk_embeddings, new_embeddings), axis=0)
    faiss_index.add(new_embeddings)
    return {
        "message": "Document added successfully.",
        "new_chunks": len(new_chunks),
        "total_chunks": len(doc_chunks)
    }

# For local/dev testing
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
