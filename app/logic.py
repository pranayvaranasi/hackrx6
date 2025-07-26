from app.document_loader import parse_document_in_memory

def process_query(document_url: str, questions: list[str]) -> list[str]:
    parsed_chunks = parse_document_in_memory(document_url)
    # For testing, just return the first 2 parsed chunks and echo questions
    return [f"Sample chunk: {ch[:100]}" for ch in parsed_chunks[:2]] + [f"Question: {q}" for q in questions]

