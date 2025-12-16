from langchain.vectorstores.faiss import FAISS
from typing import List


def build_vectorstore(chunks: List, embedding_model):
    """
    Builds a FAISS vector store from document chunks.

    Parameters:
        chunks (List[Document])
        embedding_model (Embeddings)

    Returns:
        FAISS: in-memory vector DB
    """
    vector_db = FAISS.from_documents(chunks, embedding_model)
    return vector_db


def get_retriever(vector_db, k: int = 3):
    """
    Returns a retriever object for similarity search.

    Parameters:
        vector_db (FAISS)
        k (int): number of chunks to retrieve

    Returns:
        retriever
    """
    return vector_db.as_retriever(search_kwargs={"k": k})
