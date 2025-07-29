from app.document_loader import parse_document_in_memory

def process_query(document_url: str, questions: list[str]) -> list[str]:
    parsed_chunks = parse_document_in_memory(document_url)
    # Example: Return first 2 chunks with metadata, plus questions
    answers = []
    for chunk in parsed_chunks[:2]:
        answers.append(
            f"Chunk: {chunk['chunk_text'][:100]}... "
            f"(Section: {chunk['metadata']['section'] or 'N/A'}, "
            f"Page: {chunk['metadata']['page']})"
        )
    answers.extend([f"Question: {q}" for q in questions])
    return answers