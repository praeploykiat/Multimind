import os
from tenacity import retry, stop_after_attempt, wait_random_exponential
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

try:
    import openai
except ImportError:
    is_openai_available = False
else:
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    if openai.api_key is None:
        is_openai_available = False
    else:
        is_openai_available = True

try:
    import google.generativeai as genai
except ImportError:
    is_gemini_available = False
else:
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if google_api_key is None:
        is_gemini_available = False
    else:
        is_gemini_available = True
        genai.configure(api_key=google_api_key)


BACKEND_MODEL = {
    "gemini": "models/gemini-embedding-001",  # 3072
    "openai": "text-embedding-ada-002"  # 1536
}


@retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
def get_embeddings(content, backend="gemini"):
    content = content.replace("\n\n", "\n").replace("\n", " ")
    if backend == "gemini":
        result = genai.embed_content(
            model=BACKEND_MODEL[backend],
            content=content,
            task_type="semantic_similarity"
        )
        embedding = result["embedding"]
    elif backend == "openai":
        result = openai.Embedding.create(
            input=content,
            model=BACKEND_MODEL[backend]
        )
        embedding = result.data[0].embedding
    else:
        embedding = []
    return embedding


if __name__ == "__main__":
    content = "How are you?"
    result = get_embeddings(content, backend="gemini")
    print(result)
    print(len(result))
