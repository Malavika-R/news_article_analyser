# 📚 News Research Tool

An interactive **Streamlit-based web app** that allows users to:
- ✅ Summarize multiple news articles
- 🔍 Ask natural language questions based on the articles
- 📌 Get citations and references from the original articles

Powered by **LangChain**, **FAISS**, **OpenAI**, and **Streamlit**, this app helps you turn lengthy news reports into digestible insights and searchable knowledge.

---

## 🖼️ Overview

This app provides two main functionalities:

### 1. 📝 **Summarization Panel (Left Column)**
- Input up to **3 news article URLs**
- Automatically extracts and summarizes content

### 2. ❓ **Q&A Panel (Right Column)**
- Ask any question about the input articles
- Get answers with **source references**

🧠 Built using:
- `langchain` document loaders (`NewsURLLoader`, `UnstructuredURLLoader`)
- `OpenAIEmbeddings` for vector creation
- `FAISS` for fast retrieval
- `RetrievalQAWithSourcesChain` for citation-backed Q&A

---

## 🚀 How to Run the App

### 🧰 Prerequisites

Make sure you have the following installed:

- Python ≥ 3.8
- An OpenAI API key (added to `.env`)
- Git (optional, for cloning)

---

### 🔧 Step-by-Step Setup

1. **Clone the repository** (or download manually):

    ```bash
    git clone https://github.com/your-username/news-research-tool.git
    cd news-research-tool
    ```

2. **Create a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Create a `.env` file and add your OpenAI API key:**

    ```
    OPENAI_API_KEY=your-api-key-here
    ```

5. **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

---
