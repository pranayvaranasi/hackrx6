from app.document_loader import parse_document_in_memory
from app.pinecone_utils import query_pinecone, ensure_pinecone_index
from typing import List

from app.document_loader import parse_document_in_memory
from app.pinecone_utils import query_pinecone, ensure_pinecone_index
from typing import List
from app.query_parser import parse_query_with_mistral


def process_query(document_url: str, questions: List[str]) -> List[str]:
    # Step 1: Parse and embed document
    parse_document_in_memory(document_url)
    index = ensure_pinecone_index()

    answers = []
    for question in questions:
        try:
            parsed = parse_query_with_mistral(question)
        except Exception as e:
            answers.append(f"Failed to parse query: {e}")
            continue

        parsed = parse_query_with_mistral(question)
        intent = parsed.get("intent")
        entity = parsed.get("entity")
        conditions = parsed.get("conditions", [])

        # Step 2: Construct refined query for semantic search
        query_text = f"{intent} for {entity}. Conditions: {', '.join(conditions)}"

        # Step 3: Retrieve top-5 relevant chunks
        relevant_chunks = query_pinecone(query_text, index, top_k=5)

        # Step 4: Build explainable response
        answer = f"Intent: {intent}\nEntity: {entity}\n Top Matches:\n"
        for i, chunk in enumerate(relevant_chunks, 1):
            answer += (
                f"{i}. {chunk['chunk_text'][:200]}... "
                f"(Section: {chunk['metadata'].get('section', 'N/A')}, "
                f"Page: {chunk['metadata'].get('page', 'N/A')}, "
                f"Score: {chunk['score']:.4f})\n"
            )
        answers.append(answer)

    return answers
