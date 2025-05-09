# src/embedding_utils.py
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "jinaai/jina-embeddings-v2-base-en")
_model = None

def get_embedding_model():
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME}...")
        _model = SentenceTransformer(MODEL_NAME)
        print("Embedding model loaded.")
    return _model

def get_embeddings(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    if not texts or not isinstance(texts, list):
        return []
    # Ensure model is on the correct device if using GPU for main LLM
    # _model.to(device) # if needed, but sentence-transformers usually handle this well.
    return model.encode(texts, convert_to_tensor=False).tolist()

def get_embedding_function_for_chroma(): # Renamed for clarity
    """
    Returns a function that can be used by ChromaDB's API (as an object)
    for generating embeddings. sentence-transformers embedding functions are now directly supported.
    """
    from chromadb.utils import embedding_functions
    # This uses the SentenceTransformerEmbeddingFunction from chromadb.utils
    # which is generally recommended and handles batching, etc.
    st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )
    return st_ef

if __name__ == '__main__':
    sample_texts = [
        "This is a test sentence.",
        "Another example for embedding."
    ]
    embeddings = get_embeddings(sample_texts)
    print(f"Generated {len(embeddings)} embeddings.")
    if embeddings:
        print(f"Dimension of first embedding: {len(embeddings[0])}")

    # Test the ChromaDB compatible embedding function
    # chroma_ef = get_embedding_function_for_chroma()
    # chroma_embeddings = chroma_ef(sample_texts) # This directly returns embeddings
    # print(f"Generated {len(chroma_embeddings)} embeddings using ChromaDB's EF wrapper.")
    # if chroma_embeddings:
    #     print(f"Dimension of first ChromaDB EF embedding: {len(chroma_embeddings[0])}")
