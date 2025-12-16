from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

def split_documents(docs, chunk_size=800, chunk_overlap=150):
    """
    Splits loaded documents into smaller chunks for embedding.
    Essential for building FAISS vector stores and improving retrieval quality.

    Parameters:
        docs (List[Document])
        chunk_size (int)
        chunk_overlap (int)

    Returns:
        List[Document]: chunked documents
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)
