from langchain_openai import OpenAIEmbeddings
import os


class CustomEmbeddings(OpenAIEmbeddings):
    def __init__(self, model_name: str = "text-embedding-3-small"):
        api_key = os.getenv("API_CODE")

        if not api_key:
            raise EnvironmentError(
                "API_CODE not found. Please set it in .env or environment variables."
            )

        super().__init__(
            model=model_name,
            api_code=api_key
        )

