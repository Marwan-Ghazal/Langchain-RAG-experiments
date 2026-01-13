import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv

CHROMA_PATH = "chroma"
QUERY_TEXT = "What is Alice doing at the start?"
K = 5
MIN_SCORE = 0.0
GEMINI_MODEL = "gemini-2.5-flash-lite"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"))
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. Add it to your .env file (and keep .env in .gitignore)."
        )
    genai.configure(api_key=api_key)

    query_text = QUERY_TEXT

    # Prepare the DB.
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=K)
    if len(results) == 0:
        print("Unable to find matching results.")
        return

    for i, (_doc, score) in enumerate(results, start=1):
        print(f"Result {i} relevance score: {score:.4f}")

    if results[0][1] < MIN_SCORE:
        print(f"Top result score {results[0][1]:.4f} is below min-score {MIN_SCORE:.4f}.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)

    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    response_text = getattr(response, "text", None) or str(response)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
