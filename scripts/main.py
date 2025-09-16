import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# --- Database Connection ---
PG_CONN = os.environ.get("PG_CONN", "postgresql+psycopg2://argo:argo_pass@postgres:5432/argo_db")
engine = create_engine(PG_CONN)

# --- FastAPI App ---
app = FastAPI(
    title="Argo Float API",
    description="API for querying Argo float metadata and profile data.",
    version="0.1.0",
)

# --- Pydantic Models (for response validation) ---
class Float(BaseModel):
    float_id: str = Field(..., description="The unique ID of the float.")
    platform_type: Optional[str] = Field(None, description="Type of the platform.")
    wmo_id: Optional[str] = Field(None, description="WMO ID of the float.")

class QueryRequest(BaseModel):
    query: str = Field(..., description="User natural-language question.")
    top_k: int = Field(3, ge=1, le=10, description="Number of context docs to retrieve from vector DB.")

class QueryResponse(BaseModel):
    original_query: str
    generated_sql: Optional[str] = None
    retrieved_context: List[RetrievedContextItem]
    query_results: List[Dict[str, Any]]
    error: Optional[str] = None





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
            floats = [dict(row) for row in result.mappings().all()]
            return floats
    except Exception:
        logger.exception("Database query failed")
        raise HTTPException(status_code=500, detail="Error connecting to the database.")