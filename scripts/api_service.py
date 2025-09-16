import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

# --- Configuration ---
PG_HOST = os.environ.get("PG_HOST", "postgres")
PG_PORT = os.environ.get("PG_PORT", "5432")
PG_USER = os.environ.get("PG_USER", "argo")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "argo_pass")
PG_DB = os.environ.get("PG_DB", "argo_db")
PG_CONN = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"

CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "argo_knowledge")

# --- Database & Model Setup ---
engine = create_engine(PG_CONN, pool_size=10, max_overflow=20)

# Placeholder for a real LLM client
class MockLLM:
    def query(self, context: str, question: str) -> str:
        return f"Based on the context, the answer to '{question}' is likely related to the provided documents."


# --- FastAPI App ---
app = FastAPI(
    title="Argo Float API",
    description="API for querying Argo float metadata and profile data.",
    version="0.1.0",
)

@app.on_event("startup")
def startup_event():
    """Initializes resources on application startup."""
    logger.info("Initializing API resources...")
    # Initialize ChromaDB client
    try:
        app.state.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        app.state.chroma_collection = app.state.chroma_client.get_collection(name=COLLECTION_NAME)
        logger.success("ChromaDB client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}. Please run build_vector_index.py first.")
        app.state.chroma_client = None
        app.state.chroma_collection = None
    # Load Sentence Transformer model
    app.state.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    app.state.llm_client = MockLLM() # Initialize our placeholder LLM
    logger.success("Embedding model loaded.")

# --- Pydantic Models (for response validation) ---
class Float(BaseModel):
    float_id: str = Field(..., description="The unique ID of the float.")
    platform_type: Optional[str] = Field(None, description="Type of the platform.")
    wmo_id: Optional[str] = Field(None, description="WMO ID of the float.")

    class Config:
        orm_mode = True

# --- API Endpoints ---
@app.get("/", include_in_schema=False)
def read_root():
    return {"message": "Welcome to the Argo Float API. Go to /docs for documentation."}

@app.get("/floats", response_model=List[Float], tags=["Floats"])
def get_floats(limit: int = Query(10, ge=1, le=100), offset: int = Query(0, ge=0)):
    """
    Retrieve a list of Argo floats from the database.
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT float_id, platform_type, wmo_id FROM floats ORDER BY float_id LIMIT :limit OFFSET :offset"), {"limit": limit, "offset": offset})
            floats = result.fetchall()
            return floats
    except Exception as e:
        logger.exception("Database query failed")
        raise HTTPException(status_code=500, detail="Error connecting to the database.")


