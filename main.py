import os
import streamlit as st
from dotenv import load_dotenv

from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import NewsURLLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

from summarizer import summarize_long_content

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(layout="wide")
st.title("📚 News Research Tool")

# Initialize session state
if "summaries" not in st.session_state:
    st.session_state.summaries = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Sidebar: Enter URLs
st.sidebar.title("🔗 Enter News Article URLs")
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")

# LLM setup
llm = OpenAI(temperature=0.9, max_tokens=500)
index_dir = "faiss_index"

# Process URLs
if process_url_clicked:
    # ---------- 1. Structured Loader for Summaries ----------
    summary_loader = NewsURLLoader(urls=urls)
    summary_docs = summary_loader.load()

    summaries = []
    for i, doc in enumerate(summary_docs):
        content = doc.page_content
        summary = summarize_long_content(content)
        summaries.append((urls[i], summary))
    st.session_state.summaries = summaries

    # ---------- 2. Unstructured Loader for Vectorstore ----------
    qa_loader = UnstructuredURLLoader(urls=urls)
    raw_docs = qa_loader.load()
    print("printing raw document..........", raw_docs)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", ","]
    )
    docs = splitter.split_documents(raw_docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    st.session_state.vectorstore = vectorstore
    vectorstore.save_local(index_dir)

# Layout: Two Columns
col1, col2 = st.columns([1, 1])

# LEFT: Summaries
with col1:
    st.header("📝 Article Summaries")
    if st.session_state.summaries:
        for i, (url, summary) in enumerate(st.session_state.summaries):
            st.markdown(f"**URL {i+1}:** {url}")
            st.write(summary)
            st.markdown("---")
    else:
        st.info("Enter URLs and click 'Process URLs' to see summaries.")

# RIGHT: Q&A
with col2:
    st.header("❓ Ask Questions")
    query = st.text_input("Enter your question here:")
    if query:
        if not st.session_state.vectorstore and os.path.exists(index_dir):
            st.session_state.vectorstore = FAISS.load_local(
                index_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True
            )
        if st.session_state.vectorstore:
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm, retriever=st.session_state.vectorstore.as_retriever()
            )
            result = chain({"question": query}, return_only_outputs=True)
            print("printing result.........", result)
            st.subheader("Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                for src in sources.split("\n"):
                    st.write(src)

# UI separator
st.markdown("---")
