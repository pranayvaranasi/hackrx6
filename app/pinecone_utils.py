import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from nomic import embed, login
from typing import List, Dict, Any

# Load environment variables from .env
load_dotenv()

# Authenticate with Nomic
nomic_token = os.getenv("NOMIC_API_KEY")
if not nomic_token:
    raise ValueError("NOMIC_API_KEY is not set in environment.")
login(nomic_token)

# Authenticate with Pinecone
pinecone_token = os.getenv("PINECONE_API_KEY")
if not pinecone_token:
    raise ValueError("PINECONE_API_KEY is not set in environment.")
pc = Pinecone(api_key=pinecone_token)

# Pinecone index settings
index_name = "hackrx-documents"

def ensure_pinecone_index():
    """Create Pinecone index if it doesn't exist."""
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(index_name)

def generate_embeddings(texts: List[str], task_type: str = 'search_document') -> List[List[float]]:
    """Generate embeddings using Nomic Embed v1."""
    response = embed.text(
        texts=texts,
        model='nomic-embed-text-v1',
        task_type=task_type,
        dimensionality=768
    )
    return response['embeddings']

def upsert_chunks(chunks: List[Dict[str, Any]], index):
    """Upsert document chunks with embeddings to Pinecone."""
    texts = [chunk['chunk_text'] for chunk in chunks]
    embeddings = generate_embeddings(texts, task_type='search_document')
    vectors = []
    for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
        sanitized_metadata = {}
        for key, value in chunk['metadata'].items():
            if value is None:
                sanitized_metadata[key] = ""
            elif isinstance(value, (str, int, float, bool, list)):
                if isinstance(value, list):
                    sanitized_metadata[key] = value if all(isinstance(v, str) for v in value) else str(value)
                else:
                    sanitized_metadata[key] = value
            else:
                sanitized_metadata[key] = str(value)
        vectors.append({
            "id": f"chunk_{i}",
            "values": embedding,
            "metadata": sanitized_metadata
        })
    index.upsert(vectors=vectors)

def query_pinecone(query: str, index, top_k: int = 5) -> List[Dict[str, Any]]:
    """Query Pinecone for top-k relevant chunks."""
    query_embedding = generate_embeddings([query], task_type='search_query')[0]
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
