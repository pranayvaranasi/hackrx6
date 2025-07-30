import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

# Initialize Pinecone client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", "pcsk_pWXCQ_NhvkuczBWartnMUPQSLPK6iUzgqGxxxKP2RwTFRC9Ln4KbLvjhaVQbdcreqq7hA"))
index_name = "hackrx-documents"

# Initialize Sentence Transformer for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model for efficiency

def ensure_pinecone_index():
    """Create Pinecone index if it doesn't exist."""
    try:
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=384,  # Dimension of all-MiniLM-L6-v2 embeddings
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Free plan-supported region
            )
        return pc.Index(index_name)
    except Exception as e:
        available_indexes = pc.list_indexes().names()
        raise Exception(f"Failed to create or access Pinecone index: {str(e)}. Available indexes: {available_indexes}")

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    return model.encode(texts, show_progress_bar=False).tolist()

def upsert_chunks(chunks: List[Dict[str, Any]], index):
    """Upsert document chunks with embeddings to Pinecone."""
    texts = [chunk['chunk_text'] for chunk in chunks]
    embeddings = generate_embeddings(texts)
    vectors = []
    for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
        # Sanitize metadata to remove null values and ensure valid types
        sanitized_metadata = {}
        for key, value in chunk['metadata'].items():
            if value is None:
                sanitized_metadata[key] = ""  # Replace None with empty string
            elif isinstance(value, (str, int, float, bool, list)):
                # Ensure list contains only strings
                if isinstance(value, list):
                    if all(isinstance(item, str) for item in value):
                        sanitized_metadata[key] = value
                    else:
                        sanitized_metadata[key] = str(value)  # Convert invalid lists to string
                else:
                    sanitized_metadata[key] = value
            else:
                sanitized_metadata[key] = str(value)  # Convert other types to string
        vectors.append({
            "id": f"chunk_{i}",
            "values": embedding,
            "metadata": sanitized_metadata
        })
    index.upsert(vectors=vectors)

def query_pinecone(query: str, index, top_k: int = 5) -> List[Dict[str, Any]]:
    """Query Pinecone for top-k relevant chunks."""
    query_embedding = generate_embeddings([query])[0]
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return [
        {
            "chunk_text": match['metadata'].get('text', ''),
            "metadata": match['metadata'],
            "score": match['score']
        }
        for match in results['matches']
    ]