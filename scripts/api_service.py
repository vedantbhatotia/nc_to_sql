# scripts/api_service.py
import os
import re
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field
from loguru import logger
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

# -----------------------------
# Configuration (env with defaults)
# -----------------------------
PG_HOST = os.environ.get("PG_HOST", "postgres")
PG_PORT = os.environ.get("PG_PORT", "5432")
PG_USER = os.environ.get("PG_USER", "argo")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "argo_pass")
PG_DB = os.environ.get("PG_DB", "argo_db")
PG_CONN = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"

CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "argo_knowledge")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# -----------------------------
# DB engine (global)
# -----------------------------
engine = create_engine(PG_CONN, pool_size=10, max_overflow=20)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Argo RAG API",
    version="0.1.0",
    description="RAG + MCP API for Argo data: semantic search + SQL generation (PoC)."
)

# -----------------------------
# Pydantic models
# -----------------------------
class Float(BaseModel):
    float_id: str = Field(..., description="The unique ID of the float.")
    platform_type: Optional[str] = Field(None, description="Type of the platform.")
    wmo_id: Optional[str] = Field(None, description="WMO ID of the float.")

class QueryRequest(BaseModel):
    query: str = Field(..., description="User natural-language question.")
    top_k: int = Field(3, ge=1, le=10, description="Number of context docs to retrieve from vector DB.")

class RetrievedContextItem(BaseModel):
    source: str
    document: str
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    original_query: str
    generated_sql: Optional[str]
    retrieved_context: List[RetrievedContextItem]
    query_results: List[Dict[str, Any]]
    error: Optional[str] = None

# -----------------------------
# Mock LLM (PoC)
# -----------------------------
class MockLLM:
    """
    Simple rule-based/mock LLM for hackathon PoC.
    Returns SQL for a few recognized intents; fallback safe SELECT otherwise.
    Replace with a real LLM client in production.
    """
    def query(self, context: str, question: str) -> str:
        q = question.lower()
        # heuristics to return simple SQL for demo
        if "list" in q and "floats" in q:
            return "sql\nSELECT float_id, platform_type, wmo_id FROM floats LIMIT 50;\n"
        if "first" in q and "profiles" in q:
            return "sql\nSELECT profile_id, float_id, cycle_number, profile_date FROM profiles ORDER BY profile_date DESC LIMIT 10;\n"
        if "salinity" in q or "psal" in q:
            return ("sql\nSELECT profile_id, float_id, profile_date, summary_text "
                    "FROM profiles WHERE summary_text ILIKE '%PSAL%' OR summary_text ILIKE '%salinity%' LIMIT 20;\n")
        if "count" in q and "profiles" in q:
            return "sql\nSELECT COUNT(*) as total_profiles FROM profiles;\n"
        # fallback to safe SELECT
        return "sql\nSELECT float_id, platform_type, wmo_id FROM floats LIMIT 10;\n"

# -----------------------------
# Utility: SQL extraction & safety
# -----------------------------
_SQL_SAFE_RE = re.compile(r"^\s*(?:select\b)", flags=re.IGNORECASE)

def extract_sql_from_llm(text: str) -> str:
    """
    Robustly extract SQL from LLM output.
    Supports fenced code blocks, leading 'sql' marker, or plain SQL.
    """
    if not text:
        return ""
    # fenced code block ```sql ... ```
    m = re.search(r"```sql\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # leading 'sql' token followed by SQL
    m = re.search(r"sql[:\s]*\n(.*)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # whole text looks like SQL
    if _SQL_SAFE_RE.match(text):
        return text.strip()
    # try to capture first SELECT ... ; or SELECT ... (no semicolon)
    m = re.search(r"(select\b[\s\S]*?;)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # fallback: return whole text (caller will validate)
    return text.strip()

def is_sql_safe(sql: str) -> bool:
    """
    Very basic safety checks:
      - disallow common DML/DDL keywords
      - require SQL start with SELECT
      - disallow multiple statements (semicolon)
    This is intentionally strict for PoC.
    """
    if not sql:
        return False
    lowered = sql.lower()
    forbidden = ["insert ", "update ", "delete ", "drop ", "alter ", "create ", "grant ", "revoke ", "truncate "]
    if any(k in lowered for k in forbidden):
        return False
    # disallow multiple statements
    if ";" in sql and sql.strip().count(";") > 1:
        return False
    # require SELECT at start
    return bool(_SQL_SAFE_RE.match(sql))

# -----------------------------
# Startup: initialize embedding, chroma, llm
# -----------------------------
@app.on_event("startup")
def startup_event():
    logger.info("Startup: initializing embedding model and ChromaDB...")
    # Embedding model
    try:
        app.state.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.success(f"Loaded embedding model: {EMBEDDING_MODEL}")
    except Exception as e:
        logger.exception("Failed to load embedding model")
        raise

    # ChromaDB client + collection
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        try:
            collection = client.get_or_create_collection(name=COLLECTION_NAME)
        except Exception:
            # fallback for variations of the chroma API
            collection = client.get_collection(name=COLLECTION_NAME)
        app.state.chroma_client = client
        app.state.chroma_collection = collection
        logger.success(f"Connected to ChromaDB at '{CHROMA_DB_PATH}', collection '{COLLECTION_NAME}'")
    except Exception as e:
        logger.exception("Failed to initialize ChromaDB client/collection. Run the indexer first.")
        app.state.chroma_client = None
        app.state.chroma_collection = None

    # LLM client (mock for PoC)
    app.state.llm_client = MockLLM()
    logger.success("Mock LLM initialized (PoC).")

# -----------------------------
# Health endpoint
# -----------------------------
@app.get("/health")
def health():
    ok = True
    details = {"db": True, "chroma": True}
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        details["db"] = False
        ok = False
    if not getattr(app.state, "chroma_collection", None):
        details["chroma"] = False
        ok = False
    return {"ok": ok, "details": details}

# -----------------------------
# Simple floats listing endpoint
# -----------------------------
@app.get("/floats", response_model=List[Float], tags=["floats"])
def get_floats(limit: int = Query(10, ge=1, le=100), offset: int = Query(0, ge=0)):
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT float_id, platform_type, wmo_id FROM floats ORDER BY float_id LIMIT :limit OFFSET :offset"),
                {"limit": limit, "offset": offset}
            ).mappings().all()
        # .mappings().all() returns a list of dict-like objects that Pydantic can handle
        return rows
    except Exception:
        logger.exception("Failed to fetch floats")
        raise HTTPException(status_code=500, detail="Database query failed")

# -----------------------------
# /query endpoint: RAG + MCP -> SQL
# -----------------------------
@app.post("/query", response_model=QueryResponse, tags=["rag"])
def query_endpoint(req: QueryRequest, http_request: Request):
    collection = getattr(http_request.app.state, "chroma_collection", None)
    embedding_model = getattr(http_request.app.state, "embedding_model", None)
    llm_client = getattr(http_request.app.state, "llm_client", None)

    if collection is None:
        raise HTTPException(status_code=503, detail="Vector DB (Chroma) not initialized. Run indexer first.")
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded.")

    user_query = req.query
    top_k = req.top_k
    logger.info(f"Received query (top_k={top_k}): {user_query}")

    retrieved_context: List[RetrievedContextItem] = []
    generated_sql: Optional[str] = None
    query_results: List[Dict[str, Any]] = []
    error_msg: Optional[str] = None

    try:
        # 1) Embed user query
        emb = embedding_model.encode([user_query], show_progress_bar=False)
        if hasattr(emb, "tolist"):
            emb_list = emb.tolist()
        else:
            emb_list = [list(e) for e in emb]

        # 2) Retrieve context via Chroma
        rag_res = collection.query(query_embeddings=[emb_list[0]], n_results=top_k)
        docs = rag_res.get("documents", [[]])[0]
        metas = rag_res.get("metadatas", [[]])[0]

        for doc, meta in zip(docs, metas):
            retrieved_context.append(RetrievedContextItem(
                source=meta.get("source", "unknown"),
                document=doc,
                metadata=meta
            ))

        # 3) Construct MCP prompt
        schema_parts = []
        domain_parts = []
        profile_parts = []
        for it in retrieved_context:
            src = (it.source or "").lower()
            fname = (it.metadata or {}).get("filename", "").lower()
            if "schema" in it.document.lower() or "schema" in fname:
                schema_parts.append(it.document)
            elif src.startswith("knowledge") or src.startswith("kb"):
                domain_parts.append(it.document)
            else:
                profile_parts.append(it.document)

        prompt_parts = [
            "You are an expert oceanographic data analyst and SQL query generator.",
            "Instructions: Convert the following natural language question into an executable PostgreSQL SQL query. ONLY respond with valid SQL (no explanation, no commentary).",
            "",
            "=== Relevant Schema Info ===",
            "\n\n".join(schema_parts) if schema_parts else "None available.",
            "",
            "=== Domain Knowledge ===",
            "\n\n".join(domain_parts) if domain_parts else "None available.",
            "",
            "=== Relevant Profile Data ===",
            "\n\n".join(profile_parts) if profile_parts else "None available.",
            "",
            f"User Question: {user_query}",
            "",
            "Output format: Provide only SQL. Example output:",
            "sql\nSELECT ...;\n"
        ]
        mcp_prompt = "\n".join(prompt_parts)

        # 4) Call LLM (mock PoC)
        llm_response = llm_client.query(context=mcp_prompt, question=user_query)

        # 5) Extract SQL
        generated_sql = extract_sql_from_llm(llm_response)
        logger.info(f"Extracted SQL: {generated_sql}")

        # 5.5) Safety checks
        if not is_sql_safe(generated_sql):
            error_msg = "Generated SQL failed safety checks (only simple SELECTs allowed in PoC)."
            logger.warning(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # 6) Execute SQL
        with engine.connect() as conn:
            result = conn.execute(text(generated_sql))
            query_results = result.mappings().all()

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during RAG/MCP processing")
        error_msg = str(e)

    # 7) Build response
    return QueryResponse(
        original_query=user_query,
        generated_sql=generated_sql,
        retrieved_context=[RetrievedContextItem(source=i.source, document=i.document, metadata=i.metadata) for i in retrieved_context],
        query_results=query_results,
        error=error_msg
    )

# -----------------------------
# Run with `python scripts/api_service.py` for local dev (optional)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("scripts.api_service:app", host="0.0.0.0", port=8000, reload=True)
