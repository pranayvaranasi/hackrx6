import requests
from io import BytesIO
from unstructured.partition.auto import partition

def parse_document_in_memory(document_url: str):
    """Downloads the file to memory and parses with Unstructured."""
    ext = document_url.split('?', 1)[0]
    ext = ext[ext.rfind('.'):].lower()
    file_like = BytesIO(requests.get(document_url).content)
    elements = partition(file=file_like, file_filename=f'document{ext}')
    return [el.text.strip() for el in elements if hasattr(el, "text") and el.text.strip()]
