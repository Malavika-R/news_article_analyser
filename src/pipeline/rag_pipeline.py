"""
rag_pipeline.py
----------------
Orchestrates the full RAG pipeline:
    Load → Split → Embed → Vector Store → Retrieve → LLM → Output
"""

from typing import List
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

from .loader import load_urls
from .splitter import split_documents
from .embeddings import CustomEmbeddings
from .vectorstore import build_vectorstore, get_retriever
from .prompts import QA_PROMPT, SUMMARY_PROMPT


from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI


class NewsAnalyserRAG:
    """
    End-to-end RAG pipeline class for news summarization and Q&A.
    """

    def __init__(self, llm_model="gpt-4o-mini"):
        self.embedding_model = CustomEmbeddings()
        self.llm = ChatOpenAI(model=llm_model)

    def prepare_documents(self, urls: List[str]):
        """Runs loader + splitter."""
        docs = load_urls(urls)
        chunks = split_documents(docs)
        return chunks

    def build_kb(self, chunks):
        """
        Builds FAISS DB and returns retriever.
        """
        db = build_vectorstore(chunks, self.embedding_model)
        retriever = get_retriever(db)
        return retriever

    def get_summary_chain(self, retriever):
        """Returns retrieval-augmented summarization chain."""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": SUMMARY_PROMPT}
        )


    def get_qa_chain(self, retriever):
        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=QA_PROMPT
        )

        retrieval_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=document_chain
        )

        return retrieval_chain


