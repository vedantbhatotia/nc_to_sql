import os
import chromadb
from dotenv import load_dotenv
from loguru import logger

# --- Configuration ---
load_dotenv()
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "argo_knowledge")

def main():
    """Queries the vector index with a sample question."""
    logger.info(f"Connecting to ChromaDB at '{CHROMA_DB_PATH}'...")

    if not os.path.exists(CHROMA_DB_PATH):
        logger.error(f"ChromaDB path not found at '{CHROMA_DB_PATH}'.")
        logger.error("Please run 'scripts/build_vector_index.py' first.")
        return

    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        logger.exception(f"Failed to connect to ChromaDB or get collection: {e}")
        return

    query_text = "What are the key variables for salinity measurements?"
    logger.info(f"Performing query: '{query_text}'")

    results = collection.query(query_texts=[query_text], n_results=3)

    logger.info("Query results:")
    for i, doc in enumerate(results["documents"][0]):
        distance = results["distances"][0][i]
        metadata = results["metadatas"][0][i]
        logger.info(f"\n--- Result {i+1} (Distance: {distance:.4f}) ---")
        logger.info(f"Source: {metadata.get('source')}, ID: {metadata.get('filename') or metadata.get('profile_id')}")
        logger.info(f"Text: {doc[:300].replace(os.linesep, ' ')}...")


if __name__ == "__main__":
    main()
