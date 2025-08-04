import re
from io import BytesIO
import requests
from unstructured.partition.auto import partition
from nltk.tokenize import sent_tokenize, WordPunctTokenizer
import nltk
from typing import List, Dict, Any
from app.pinecone_utils import ensure_pinecone_index, upsert_chunks

# Download NLTK data for sentence tokenization
nltk.download('punkt')

def clean_text(text: str, doc_ext: str) -> str:
    """Clean text by removing headers/footers, normalizing whitespace, and fixing broken words."""
    # Remove common headers/footers (e.g., company names, page numbers, document IDs)
    text = re.sub(r'(?:Page \d+ of \d+|UIN: [A-Z0-9]+|\b[A-Z][a-zA-Z\s]*Co\.\s*Ltd\.|Confidential\s*?\n)', '', text, flags=re.IGNORECASE)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Fix broken words (e.g., hyphenated splits in PDFs)
    text = re.sub(r'(\w+)-\s*(\w+)', r'\1\2', text)
    # Email-specific cleaning: remove signatures, replies, and boilerplate
    if doc_ext == '.eml':
        text = re.sub(r'(?:--+\s*Sent from.*|On .* wrote:.*|Best regards,.*|Sincerely,.*)', '', text, flags=re.IGNORECASE)
    # DOCX-specific cleaning: remove formatting artifacts
    if doc_ext == '.docx':
        text = re.sub(r'(\[.*?\]|\{.*?\})', '', text)
    # Remove repetitive or boilerplate phrases common in legal/insurance documents
    text = re.sub(r'(?:This document is.*|All rights reserved|Version \d+\.\d+)', '', text, flags=re.IGNORECASE)
    return text.strip()

def detect_structure(elements: List[Any]) -> List[Dict[str, Any]]:
    """
    Detect logical structure (sections, clauses, lists, tables) in the document.
    Returns a list of dictionaries with text, heading, and metadata.
    """
    structured_elements = []
    current_section = None
    current_clause = None
    current_page = 1
    last_heading = None
    context_stack = []  # Track contextual qualifiers (e.g., "NOT covered")

    for element in elements:
        text = element.text.strip() if hasattr(element, 'text') else ''
        if not text:
            continue

        # Update page number if available
        if hasattr(element, 'metadata') and element.metadata.page_number:
            current_page = element.metadata.page_number

        # Detect headings (e.g., "Section 1", "1.1 Title", all-caps, or categorized as Title)
        heading_match = re.match(r'^(?:\d+\.\s*[A-Z\s]+|\d+\.\d+\.\s*[A-Z\s]+|[A-Z\s]{10,}|[0-9]+\.\s*[A-Za-z].+)', text, re.IGNORECASE)
        if heading_match or (hasattr(element, 'category') and element.category == 'Title'):
            current_section = text
            last_heading = text
            context_stack = []  # Reset context for new section
            structured_elements.append({
                'text': text,
                'heading': last_heading,
                'metadata': {
                    'section': current_section,
                    'clause': None,
                    'page': current_page,
                    'context': []
                }
            })
            continue

        # Detect clauses (e.g., "a)", "1.", "Code - Excl01")
        clause_match = re.match(r'^(?:[a-z]\)|\d+\.|Code\s*-\s*[A-Za-z0-9]+|[ivx]+\.)', text, re.IGNORECASE)
        if clause_match:
            current_clause = clause_match.group(0)
            structured_elements.append({
                'text': text,
                'heading': last_heading,
                'metadata': {
                    'section': current_section,
                    'clause': current_clause,
                    'page': current_page,
                    'context': context_stack
                }
            })
            continue

        # Detect contextual qualifiers (e.g., "NOT covered", "Exclusions", "Conditions apply")
        context_match = re.search(r'(?:not covered|exclusions?|conditions\s*apply|subject to|except\s*for|unless\s*otherwise\s*stated)', text, re.IGNORECASE)
        if context_match:
            context_stack.append(context_match.group(0))

        # Handle lists, tables, or annexures
        if hasattr(element, 'category') and element.category == 'ListItem':
            structured_elements.append({
                'text': text,
                'heading': last_heading,
                'metadata': {
                    'section': current_section,
                    'clause': current_clause,
                    'page': current_page,
                    'context': context_stack
                }
            })
        elif hasattr(element, 'category') and element.category == 'Table':
            structured_elements.append({
                'text': text,
                'heading': last_heading,
                'metadata': {
                    'section': current_section,
                    'clause': current_clause,
                    'page': current_page,
                    'context': context_stack
                }
            })
        elif 'annexure' in text.lower() or 'schedule' in text.lower():
            structured_elements.append({
                'text': text,
                'heading': last_heading or text,
                'metadata': {
                    'section': current_section or text,
                    'clause': None,
                    'page': current_page,
                    'context': context_stack
                }
            })
        else:
            # Regular text
            structured_elements.append({
                'text': text,
                'heading': last_heading,
                'metadata': {
                    'section': current_section,
                    'clause': current_clause,
                    'page': current_page,
                    'context': context_stack
                }
            })

    return structured_elements

def semantic_chunking(text: str, heading: str, metadata: Dict[str, Any], max_tokens: int = 500, overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Split text into semantic chunks, respecting clause boundaries and including heading/context.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = [heading] if heading else []
    tokenizer = WordPunctTokenizer()
    current_length = len(tokenizer.tokenize(heading)) if heading else 0
    current_context = metadata.get('context', [])
    current_clause = metadata.get('clause', None)

    # Fallback context detection if metadata['context'] is empty
    if not current_context:
        for sentence in sentences:
            context_match = re.search(r'(?:not covered|exclusions?|conditions\s*apply|subject to|except\s*for|unless\s*otherwise\s*stated)', sentence, re.IGNORECASE)
            if context_match and context_match.group(0) not in current_context:
                current_context.append(context_match.group(0))

    clause_buffer = []
    clause_length = 0

    for sentence in sentences:
        sentence_tokens = len(tokenizer.tokenize(sentence))
        is_clause_start = re.match(r'^(?:[a-z]\)|\d+\.|Code\s*-\s*[A-Za-z0-9]+|[ivx]+\.)', sentence, re.IGNORECASE)

        if is_clause_start and clause_buffer:
            # Finalize previous clause
            if current_length + clause_length <= max_tokens:
                current_chunk.extend(clause_buffer)
                current_length += clause_length
            else:
                # Prepend context to chunk_text if present
                chunk_text = ' '.join(current_chunk)
                if current_context:
                    chunk_text = f"[{' '.join(current_context)}] {chunk_text}"
                chunks.append({
                    'chunk_text': chunk_text,
                    'heading': heading,
                    'metadata': {**metadata, 'clause': current_clause or '', 'context': current_context}
                })
                # Include overlap sentences
                overlap_sentences = clause_buffer[-min(len(clause_buffer), 1 if clause_length < 100 else 2):]
                overlap_length = sum(len(tokenizer.tokenize(s)) for s in overlap_sentences)
                current_chunk = [heading] + overlap_sentences if heading else overlap_sentences
                current_length = len(tokenizer.tokenize(heading)) + overlap_length if heading else overlap_length
            clause_buffer = []
            clause_length = 0
            current_clause = is_clause_start.group(0)

        clause_buffer.append(sentence)
        clause_length += sentence_tokens

    # Process remaining clause
    if clause_buffer:
        if current_length + clause_length <= max_tokens:
            current_chunk.extend(clause_buffer)
        else:
            chunk_text = ' '.join(current_chunk)
            if current_context:
                chunk_text = f"[{' '.join(current_context)}] {chunk_text}"
            chunks.append({
                'chunk_text': chunk_text,
                'heading': heading,
                'metadata': {**metadata, 'clause': current_clause or '', 'context': current_context}
            })
            overlap_sentences = clause_buffer[-min(len(clause_buffer), 1 if clause_length < 100 else 2):]
            overlap_length = sum(len(tokenizer.tokenize(s)) for s in overlap_sentences)
            current_chunk = [heading] + overlap_sentences if heading else overlap_sentences
            current_length = len(tokenizer.tokenize(heading)) + overlap_length if heading else overlap_length
            chunk_text = ' '.join(current_chunk)
            if current_context:
                chunk_text = f"[{' '.join(current_context)}] {chunk_text}"
            chunks.append({
                'chunk_text': chunk_text,
                'heading': heading,
                'metadata': {**metadata, 'clause': current_clause or '', 'context': current_context}
            })

    # Add final chunk if any
    if current_chunk and (len(current_chunk) > 1 or (current_chunk and current_chunk[0] != heading)):
        chunk_text = ' '.join(current_chunk)
        if current_context:
            chunk_text = f"[{' '.join(current_context)}] {chunk_text}"
        chunks.append({
            'chunk_text': chunk_text,
            'heading': heading,
            'metadata': {**metadata, 'clause': current_clause or '', 'context': current_context}
        })

    return chunks

def parse_document_in_memory(document_url: str) -> List[Dict[str, Any]]:
    """
    Downloads the file to memory, parses with Unstructured, and applies semantic-aware chunking.
    Returns a list of chunks with metadata in JSON format.
    """
    # Determine document extension
    ext = document_url.split('?', 1)[0]
    ext = ext[ext.rfind('.'):].lower()

    # Download document
    try:
        file_like = BytesIO(requests.get(document_url).content)
    except requests.RequestException as e:
        raise ValueError(f"Failed to download document: {str(e)}")

    # Parse document using unstructured
    elements = partition(file=file_like, file_filename=f'document{ext}', include_page_breaks=True)

    # Clean text
    raw_text = ' '.join([el.text.strip() for el in elements if hasattr(el, 'text') and el.text.strip()])
    cleaned_text = clean_text(raw_text, ext)

    # Detect structure
    structured_elements = detect_structure(elements)

    # Process chunks by section
    final_chunks = []
    current_section_text = []
    current_heading = None
    current_metadata = None
    max_tokens = 500
    overlap = 100

    for element in structured_elements:
        text = element['text']
        heading = element['heading']
        metadata = element['metadata']

        if heading != current_heading:
            # Process previous section
            if current_section_text:
                section_text = ' '.join(current_section_text)
                final_chunks.extend(semantic_chunking(section_text, current_heading, current_metadata, max_tokens, overlap))
                current_section_text = []
            current_heading = heading
            current_metadata = metadata

        current_section_text.append(text)

    # Process final section
    if current_section_text:
        section_text = ' '.join(current_section_text)
        final_chunks.extend(semantic_chunking(section_text, current_heading, current_metadata, max_tokens, overlap))

    # Parse clause relationships (e.g., "See Section 4.2", "Refer to Clause A-1")
    for chunk in final_chunks:
        references = re.findall(r'(?:See|Refer to)\s*(?:Section|Clause)\s*[A-Z0-9\-.\s]+', chunk['chunk_text'], re.IGNORECASE)
        if references:
            chunk['metadata']['references'] = references

    index = ensure_pinecone_index()
    upsert_chunks(final_chunks, index)

    return final_chunks