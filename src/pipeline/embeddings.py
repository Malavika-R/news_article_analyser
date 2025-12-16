from langchain_openai import OpenAIEmbeddings
import os


class CustomEmbeddings(OpenAIEmbeddings):
    def __init__(self, model_name: str = "text-embedding-3-small"):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not found. Please set it in .env or environment variables."
            )

        super().__init__(
            model=model_name,
            openai_api_key=api_key
        )

