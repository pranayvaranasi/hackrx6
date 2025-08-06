from app.document_loader import parse_document_in_memory
from app.pinecone_utils import query_pinecone, ensure_pinecone_index
from app.query_parser import parse_query_with_mistral
from together import Together
from typing import List
import os

# Initialize Together API client
together_api = Together(api_key=os.getenv("TOGETHER_API_KEY"))

rag_prompt_template = """
You are an intelligent insurance, legal, HR, and compliance policy assistant.
You are provided with one or more extracted policy/contract clauses, and a user question.
ONLY use the content in the retrieved clauses as evidence for your response.
NEVER make up information. If the answer is not found, reply: "Information not found in document."

Instructions:
- Quote or paraphrase relevant policy text.
- Clearly state conditions, waiting periods, inclusions, or exclusions exactly as specified.
- Output your answer as a single paragraph.

Context:
{retrieved_clauses}

User Question:
{user_question}
"""

def generate_final_answer_with_llm(question: str, top_chunks: List[dict]) -> str:
    context = "\n\n".join(chunk["chunk_text"] for chunk in top_chunks)
    prompt = rag_prompt_template.format(retrieved_clauses=context, user_question=question)

    try:
        response = together_api.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.1",
            messages=[
                {"role": "system", "content": "You answer based only on provided policy clauses."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Failed to generate answer: {str(e)}"

def apply_keyword_boost(chunks: List[dict], query_keywords: List[str], boost: float = 0.05) -> List[dict]:
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

        # Step 2: Build semantic query
        query_parts = [intent.replace("_", " ")]
        if entity:
            query_parts.append(entity)
        if conditions:
            query_parts.extend(conditions)
        if keywords:
            query_parts.extend(keywords)
        query_text = " ".join(query_parts)

        # Step 3: Retrieve top-K chunks + keyword boost
        initial_chunks = query_pinecone(query_text, index, top_k=10)
        boosted_chunks = apply_keyword_boost(initial_chunks, keywords)
        sorted_chunks = sorted(boosted_chunks, key=lambda c: c["score"], reverse=True)
        top_chunks = sorted_chunks[:5]

        # Step 4: Generate final answer using top chunks
        final_answer = generate_final_answer_with_llm(question, top_chunks)
        answers.append(final_answer)

    return answers
