from langchain.prompts import PromptTemplate

QA_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="""
You are an AI analyst.

Use ONLY the context below to answer the user's question.

IMPORTANT:
- Do NOT repeat or restate any question found in the context.
- Do NOT include phrases like "Question:" or "Answer:".
- Return ONLY the final answer as a short paragraph or bullet points.

Context:
{context}

Final Answer:
"""
)


SUMMARY_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="""
You are an AI financial analyst.

Summarize the following news content focusing on:
- Key events
- Business impact
- Financial implications

Context:
{context}

Return a concise professional summary.
"""
)
