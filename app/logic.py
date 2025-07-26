def process_query(document_url: str, questions: list[str]) -> list[str]:
    return [f"Answer to: {q}" for q in questions]
