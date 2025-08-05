from app.document_loader import parse_document_in_memory
from app.pinecone_utils import query_pinecone, ensure_pinecone_index
from app.query_parser import parse_query_with_mistral
from together import Together
from typing import List
import os
import json

# Initialize Together API client
together_api = Together(api_key=os.getenv("TOGETHER_API_KEY"))

def rerank_chunks_with_llm(question: str, chunks: List[dict]) -> List[dict]:
    """
    Re-rank top-K Pinecone chunks using LLM, assigning a score (0–1) based on relevance.
    """
    prompt = f"""
You are a smart assistant for processing insurance, legal, HR, and compliance documents.

Question: "{question}"

Below are candidate passages from a document. Rank them by relevance to the question.

Return only a list of:
[
  {{ "passage": <index>, "score": <float between 0 and 1> }},
  ...
]
{''.join([f'\nPassage {i+1}:\n{chunk["chunk_text"]}\n' for i, chunk in enumerate(chunks)])}
""".strip()

    try:
        response = together_api.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.1",
            messages=[
                {"role": "system", "content": "You rank document passages by relevance to questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        content = response.choices[0].message.content.strip()
        scores = json.loads(content)

        # Attach scores to chunks
        for item in scores:
            idx = item["passage"] - 1
            if 0 <= idx < len(chunks):
                chunks[idx]["llm_score"] = item["score"]

    except Exception as e:
        # Fallback: linear descending scores
        for i, chunk in enumerate(chunks):
            chunk["llm_score"] = 1.0 - (i * 0.1)

    # Final hybrid score
    for chunk in chunks:
        chunk["final_score"] = (
            0.3 * chunk.get("score", 0) +
            0.7 * chunk.get("llm_score", 0)
        )

    return sorted(chunks, key=lambda x: x["final_score"], reverse=True)

def apply_keyword_boost(chunks: List[dict], query_keywords: List[str], boost: float = 0.05) -> List[dict]:
    """
    Boosts chunk scores if they contain any keyword from the query.
    Each matching keyword adds `boost` to the chunk's score.
    """
    for chunk in chunks:
        text = chunk.get("chunk_text", "").lower()
        match_count = sum(1 for kw in query_keywords if kw.lower() in text)
        if match_count > 0:
            chunk["score"] += boost * match_count
    return chunks


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

        intent = parsed.get("intent")
        entity = parsed.get("entity")
        conditions = parsed.get("conditions", [])
        keywords = parsed.get("keywords", [])

        # Step 2: Semantic search query
        query_parts = [intent.replace("_", " ")]
        if entity:
            query_parts.append(entity)
        if conditions:
            query_parts.extend(conditions)
        if keywords:
            query_parts.extend(keywords)
        query_text = " ".join(query_parts)

        # Step 3: Retrieve top-10 relevant chunks
        initial_chunks = query_pinecone(query_text, index, top_k=10)
        initial_chunks = apply_keyword_boost(initial_chunks, keywords)
        top_chunks = rerank_chunks_with_llm(question, initial_chunks)

        # Step 4: Generate answer
        answer = f"Intent: {intent}\nEntity: {entity}\nTop Matches:\n"
        for i, chunk in enumerate(top_chunks[:5], 1):  # Limit output to top 5
            answer += (
                f"{i}. {chunk['chunk_text'][:200]}... "
                f"(Section: {chunk['metadata'].get('section', 'N/A')}, "
                f"Page: {chunk['metadata'].get('page', 'N/A')}, "
                f"Score: {chunk['final_score']:.4f})\n"
            )
        answers.append(answer)

    return answers
