"""
app.py
-------
Main Streamlit application for the AI-Powered News Article Analyzer.
Provides a clean, corporate UI with:

1. URL input
2. Company name input
3. Summarization
4. Q&A

The UI calls the RAG pipeline defined in pipeline/rag_pipeline.py
"""
import os

# Prevent OpenAI client from receiving unsupported proxy args
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

import streamlit as st
from pipeline.rag_pipeline import NewsAnalyserRAG

# ----------------------------------------
# Load model credentials
# ----------------------------------------
from dotenv import load_dotenv
load_dotenv()

# ----------------------------------------
# Page Config
# ----------------------------------------
st.set_page_config(
    page_title="AI-Powered News Article Analyzer",
    layout="centered",
)


# ----------------------------------------
# App Header
# ----------------------------------------
st.title("📰 AI-Powered News Article Analyzer")
# st.caption("RAG-based system with LangChain • Custom Embeddings • OpenAI gpt-4o-mini")


# ----------------------------------------
# Inputs
# ----------------------------------------
url_input = st.text_area(
    "Enter one or multiple news URLs (one per line):",
    placeholder="https://www.reuters.com/...\nhttps://economictimes.indiatimes.com/..."
)

company_name = st.text_input(
    "Company Name (optional, improves specificity):",
    placeholder="Example: Apple, Tesla, HDFC Bank"
)

user_question = st.text_input(
    "Ask a question about the article(s):",
    placeholder="How will this impact the company's revenue?"
)


# ----------------------------------------
# Initialize RAG Pipeline
# ----------------------------------------
rag = NewsAnalyserRAG(llm_model="gpt-4.1")

# ----------------------------------------
# Button: Build Knowledge Base
# ----------------------------------------
from utils.helper import hash_urls

url_list = [u.strip() for u in url_input.split("\n") if u.strip()]
current_hash = hash_urls(url_list)

if st.button("Build Knowledge Base", type="primary"):
    if st.session_state.get("kb_hash") == current_hash:
        st.info("Knowledge base already exists for these URLs.")
    else:
        with st.spinner("Processing URLs..."):
            chunks = rag.prepare_documents(url_list)
            retriever = rag.build_kb(chunks)

            st.session_state["retriever"] = retriever
            st.session_state["kb_hash"] = current_hash

            st.success("Knowledge Base successfully built!")


# ----------------------------------------
# Button: Summarize
# ----------------------------------------
if st.button("Generate Company-Specific Summary"):
    if "retriever" not in st.session_state:
        st.error("Please build the knowledge base first.")
    
    elif not company_name.strip():
        st.warning("Please enter a company name or proceed with a general summary.")

    else:
        summary_chain = rag.get_summary_chain(st.session_state["retriever"])

        with st.spinner("Generating summary..."):
            response = summary_chain.invoke({
                "query": f"Summarize the article with specific focus on {company_name or 'the company'}"
            })

            result = response["result"]

            st.subheader("📌 Summary")
            st.write(result)


# ----------------------------------------
# Button: Ask Question
# ----------------------------------------
if st.button("Answer My Question"):
    if "retriever" not in st.session_state:
        st.error("Please build the knowledge base first.")

    elif not user_question.strip():
        st.warning("Please enter a question.")

    else:
        qa_chain = rag.get_qa_chain(st.session_state["retriever"])

        with st.spinner("Thinking..."):
            response = qa_chain.invoke({
                "input": user_question.strip()
            })

            answer = response["answer"]

            st.subheader("💬 Answer")
            st.write(answer)


