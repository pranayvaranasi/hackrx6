from app.document_loader import parse_document_in_memory
from app.pinecone_utils import query_pinecone, ensure_pinecone_index
from typing import List

def process_query(document_url: str, questions: List[str]) -> List[str]:
    # Parse document and store in Pinecone
    parse_document_in_memory(document_url)
    index = ensure_pinecone_index()
    
    answers = []
    for question in questions:
        # Retrieve top-5 relevant chunks
        relevant_chunks = query_pinecone(question, index, top_k=5)
        
        # Format answer with chunk text and metadata for traceability
        answer = f"Question: {question}\nRelevant Chunks:\n"
        for i, chunk in enumerate(relevant_chunks, 1):
            answer += (
                f"Chunk {i}: {chunk['chunk_text'][:100]}... "
                f"(Section: {chunk['metadata'].get('section', 'N/A')}, "
                f"Page: {chunk['metadata'].get('page', 'N/A')}, "
                f"Score: {chunk['score']:.4f})\n"
            )
        answers.append(answer)
    
    return answers