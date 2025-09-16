import os
import glob
import sys
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
from tqdm import tqdm

# --- Configuration ---
load_dotenv()

PG_HOST = os.environ.get("PG_HOST", "postgres")
PG_PORT = os.environ.get("PG_PORT", "5432")
PG_USER = os.environ.get("PG_USER", "argo")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "argo_pass")
PG_DB = os.environ.get("PG_DB", "argo_db")
PG_CONN = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "argo_knowledge"
KNOWLEDGE_BASE_DIR = "knowledge_base"

# --- Database Connection ---
try:
    engine = create_engine(PG_CONN, pool_size=10, max_overflow=20)
except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    engine = None

# --- Helper Functions ---

def load_kb_documents(kb_dir: str) -> list[dict]:
    """Loads all markdown files from the knowledge_base directory."""
    logger.info(f"Loading documents from '{kb_dir}'...")
    docs = []
    for file_path in glob.glob(os.path.join(kb_dir, "*.md")):
        filename = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        docs.append({
            "id": f"kb_{filename}",
            "text": content,
            "metadata": {"source": "knowledge_base", "filename": filename}
        })
    logger.info(f"Found {len(docs)} documents in knowledge base.")
    return docs

def load_db_summaries() -> list[dict]:
    """Queries the profiles table for all non-null summary_text entries."""
    if not engine:
        logger.error("Database engine not available. Skipping DB summary loading.")
        return []
    
    logger.info("Loading profile summaries from database...")
    docs = []
    query = text("SELECT profile_id, summary_text FROM profiles WHERE summary_text IS NOT NULL")
    
    try:
        with engine.connect() as conn:
            results = conn.execute(query).fetchall()
            for profile_id, summary_text in results:
                docs.append({
                    "id": f"profile_{profile_id}",
                    "text": summary_text,
                    "metadata": {"source": "database_summary", "profile_id": str(profile_id)}
                })
        logger.info(f"Found {len(docs)} summaries in database.")
    except Exception as e:
        logger.exception(f"Failed to query database for summaries: {e}")
    
    return docs


def main():
    """Main function to build and populate the vector index."""
    fresh_build = "--fresh" in sys.argv
    
    logger.info(f"Starting vector index build process... (Fresh build: {fresh_build})")

    # 1. Load all text sources
    kb_docs = load_kb_documents(KNOWLEDGE_BASE_DIR)
    db_docs = load_db_summaries()
    all_docs = kb_docs + db_docs

    if not all_docs:
        logger.warning("No documents found to index. Exiting.")
        return

    # 2. Initialize Embedding Model
    logger.info(f"Loading embedding model: '{EMBEDDING_MODEL}'")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # 3. Initialize ChromaDB
    logger.info(f"Initializing ChromaDB client at '{CHROMA_DB_PATH}'")
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # 4. On a fresh build, clear out old data first to handle deletions.
    if fresh_build and collection.count() > 0:
        logger.info("Fresh build requested. Deleting existing documents from collection...")
        collection.delete(where={"source": "knowledge_base"})
        collection.delete(where={"source": "database_summary"})
        logger.info(f"Collection count after deletion: {collection.count()}")

    # 5. Prepare data for ChromaDB
    documents = [doc["text"] for doc in all_docs]
    metadatas = [doc["metadata"] for doc in all_docs]
    ids = [doc["id"] for doc in all_docs]

    # 6. Embed all texts
    logger.info(f"Generating embeddings for {len(documents)} documents...")
    # Ensure embeddings are a list of lists of floats
    embeddings = np.array(model.encode(documents, show_progress_bar=True)).tolist()

    # 7. Add data to the collection
    logger.info(f"Adding {len(documents)} vectors to ChromaDB collection '{COLLECTION_NAME}'...")
    # Use upsert to avoid errors on re-running
    collection.upsert(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    # 8. Confirmation
    logger.success("Vector index build complete!")
    logger.info(f"Total vectors indexed: {collection.count()}")
    logger.info(f"  - From Knowledge Base: {len(kb_docs)}")
    logger.info(f"  - From Database Summaries: {len(db_docs)}")
    logger.info(f"Sample of indexed items: {collection.peek(limit=3)}")


if __name__ == "__main__":
    main()
