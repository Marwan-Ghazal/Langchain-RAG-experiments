from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np

def main():
    # Get embedding for a word.
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {vector}")
    print(f"Vector length: {len(vector)}")

    # Compare vector of two words
    words = ("apple", "iPhone")
    v1 = np.array(embedding_function.embed_query(words[0]), dtype=np.float32)
    v2 = np.array(embedding_function.embed_query(words[1]), dtype=np.float32)
    cosine_similarity = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    print(f"Cosine similarity ({words[0]}, {words[1]}): {cosine_similarity}")


if __name__ == "__main__":
    main()
