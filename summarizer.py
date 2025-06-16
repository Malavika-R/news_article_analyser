from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables
llm = OpenAI(temperature=0.9, max_tokens=500)

def summarize_long_content(content, chunk_size=1500):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(content)

    chunk_summaries = []
    for chunk in chunks:
        prompt = f"Summarize the following part of a news article:\n\n{chunk}"
        summary = llm(prompt)
        chunk_summaries.append(summary)

    combined_summary_prompt = (
        "Combine the following summaries into a single coherent news summary:\n\n" +
        "\n\n".join(chunk_summaries)
    )
    final_summary = llm(combined_summary_prompt)
    return final_summary
