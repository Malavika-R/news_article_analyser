# 📰 AI-Powered News Article Analyzer  
### Retrieval-Augmented Generation (RAG) • LangChain • Custom Embeddings • OpenAI GPT-4.1

This project is a **RAG-based news analysis system** that allows users to:

- Analyze news articles from **user-provided URLs**
- Generate **company-specific summaries**
- Ask **company-specific Q&A** about the articles
- View results powered by **custom embeddings** + **prompt orchestration**

---

## 🚀 Features

### 🔹 Retrieval-Augmented Generation (RAG)
- Extracts text from URLs  
- Splits into semantic chunks  
- Embeds using **custom OpenAI embeddings**  
- Stores in FAISS vector database  
- Enables precise retrieval for Q&A

### 🔹 Company-Specific Summaries  
Uses a custom financial-analysis prompt for targeted insights.

### 🔹 Clean Corporate UI  
Built with Streamlit.

---

## 🧱 Architecture

```mermaid
flowchart TD
    A[User URLs] --> B[URL Loader]
    B --> C[Text Splitter]
    C --> D[Custom Embeddings]
    D --> E[FAISS Vector Store]
    F[User Query] --> G[Retriever]
    E --> G
    G --> H[LLM + Prompt Orchestration]
    H --> I[Answer / Summary Output]
