from langchain.document_loaders import UnstructuredURLLoader
from typing import List


def load_urls(url_list: List[str]):
    """
    Loads and extracts text from a list of user-provided URLs.
    Uses LangChain's UnstructuredURLLoader for robust web extraction.

    Parameters:
        url_list (List[str]): List of URLs to load.

    Returns:
        docs (List[Document]): Loaded documents containing extracted text.
    """
    loader = UnstructuredURLLoader(urls=url_list)
    docs = loader.load()
    return docs
