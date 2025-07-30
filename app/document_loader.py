import re
from io import BytesIO
import requests
from unstructured.partition.auto import partition
from nltk.tokenize import sent_tokenize
import nltk
from typing import List, Dict, Any
from app.pinecone_utils import ensure_pinecone_index, upsert_chunks

# Download NLTK data for sentence tokenization
nltk.download('punkt')

def clean_text(text: str, doc_ext: str) -> str:
    """Clean text based on document extension, removing headers/footers, normalizing whitespace, and fixing broken words."""
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
        text = re.sub(r'(\[.*?\]|\{.*?\})', '', text)  # Remove placeholder tags
    return text.strip()

def detect_structure(elements: List[Any]) -> List[Dict[str, Any]]:
    """
    Detect logical structure (sections, clauses, lists, tables) in the document.
    Returns a list of dictionaries with text and metadata.
    """
    structured_elements = []
    current_section = None
    current_clause = None
    current_page = 1
    last_heading = None

    for element in elements:
        text = element.text.strip() if hasattr(element, 'text') else ''
        if not text:
            continue

        # Update page number if available in metadata
        if hasattr(element, 'metadata') and element.metadata.page_number:
            current_page = element.metadata.page_number

        # Detect headings (e.g., "Section 1", "1.1 Title", or any all-caps/leading numbers)
        heading_match = re.match(r'^(?:\d+\.\s*[A-Z\s]+|\d+\.\d+\.\s*[A-Z\s]+|[A-Z\s]{10,})', text, re.IGNORECASE)
        if heading_match or (hasattr(element, 'category') and element.category == 'Title'):
            current_section = text
            last_heading = text
            structured_elements.append({
                'text': text,
                'metadata': {
                    'section': current_section,
                    'clause': None,
                    'page': current_page,
                    'heading': last_heading
                }
            })
            continue

        # Detect clauses (e.g., "a)", "1.", "Code - Excl01")
        clause_match = re.match(r'^(?:[a-z]\)|\d+\.|Code\s*-\s*[A-Za-z0-9]+|[ivx]+\.)', text, re.IGNORECASE)
        if clause_match:
            current_clause = clause_match.group(0)
            structured_elements.append({
                'text': text,
                'metadata': {
                    'section': current_section,
                    'clause': current_clause,
                    'page': current_page,
                    'heading': last_heading
                }
            })
            continue

        # Handle lists and tables
        if hasattr(element, 'category') and element.category == 'ListItem':
            structured_elements.append({
                'text': text,
                'metadata': {
                    'section': current_section,
                    'clause': current_clause,
                    'page': current_page,
                    'heading': last_heading
                }
            })
        elif hasattr(element, 'category') and element.category == 'Table':
            structured_elements.append({
                'text': text,
                'metadata': {
                    'section': current_section,
                    'clause': current_clause,
                    'page': current_page,
                    'heading': last_heading
                }
            })
        else:
            # Regular text
            structured_elements.append({
                'text': text,
                'metadata': {
                    'section': current_section,
                    'clause': current_clause,
                    'page': current_page,
                    'heading': last_heading
                }
            })

    return structured_elements

def sliding_window_chunking(text: str, window_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Apply sliding window chunking to long text with overlap.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_tokens = len(sentence.split())
        if current_length + sentence_tokens > window_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            overlap_sentences = []
            overlap_length = 0
            for sent in current_chunk[::-1]:
                sent_tokens = len(sent.split())
                if overlap_length + sent_tokens <= overlap:
                    overlap_sentences.append(sent)
                    overlap_length += sent_tokens
                else:
                    break
            current_chunk = overlap_sentences[::-1]
            current_length = overlap_length
        current_chunk.append(sentence)
        current_length += sentence_tokens

    if current_chunk:
        chunks.append(' '.join(current_chunk))

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
    
    # Process chunks
    final_chunks = []
    current_chunk_text = []
    current_metadata = None
    current_token_count = 0
    window_size = 500
    overlap = 100

    for element in structured_elements:
        text = element['text']
        metadata = element['metadata']
        token_count = len(text.split())

        # Handle tables
        if hasattr(element, 'category') and element.category == 'Table':
            final_chunks.append({
                'chunk_text': text,
                'metadata': metadata
            })
            continue

        # Handle lists (group logically)
        if hasattr(element, 'category') and element.category == 'ListItem':
            current_chunk_text.append(text)
            current_token_count += token_count
            if current_token_count >= window_size:
                final_chunks.append({
                    'chunk_text': ' '.join(current_chunk_text),
                    'metadata': metadata
                })
                current_chunk_text = current_chunk_text[-2:]  # Overlap with last 2 items
                current_token_count = sum(len(t.split()) for t in current_chunk_text)
            continue

        # Handle annexures or schedules
        if 'annexure' in text.lower() or 'schedule' in text.lower():
            if current_chunk_text:
                final_chunks.append({
                    'chunk_text': ' '.join(current_chunk_text),
                    'metadata': current_metadata
                })
                current_chunk_text = []
                current_token_count = 0
            final_chunks.append({
                'chunk_text': text,
                'metadata': metadata
            })
            continue

        # Regular text chunking
        current_chunk_text.append(text)
        current_token_count += token_count
        current_metadata = metadata

        if current_token_count >= window_size:
            chunk_text = ' '.join(current_chunk_text)
            sub_chunks = sliding_window_chunking(chunk_text, window_size, overlap)
            for sub_chunk in sub_chunks:
                final_chunks.append({
                    'chunk_text': sub_chunk,
                    'metadata': metadata
                })
            current_chunk_text = current_chunk_text[-2:]  # Overlap with last 2 sentences
            current_token_count = sum(len(t.split()) for t in current_chunk_text)

    # Add remaining text
    if current_chunk_text:
        final_chunks.append({
            'chunk_text': ' '.join(current_chunk_text),
            'metadata': current_metadata
        })

    # Parse clause relationships (e.g., "See Section 4.2", "Refer to Clause A-1")
    for chunk in final_chunks:
        references = re.findall(r'(?:See|Refer to)\s*(?:Section|Clause)\s*[A-Z0-9\-.\s]+', chunk['chunk_text'], re.IGNORECASE)
        if references:
            chunk['metadata']['references'] = references

    index = ensure_pinecone_index()
    upsert_chunks(final_chunks, index)
    
    return final_chunks