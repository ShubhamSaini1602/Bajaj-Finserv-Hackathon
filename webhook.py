import os
import shutil
import tempfile
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Import your new API-safe utility functions ---
from utils.api_utils import (
    download_file_from_url,
    load_document_from_path,
    chunk_documents_api,
    create_and_store_embeddings_api
)
# --- Import your other existing modules ---
from utils.llm_manager import initialize_gemini_models
from core.qa_chain import create_qa_chain

# --- 1. INITIALIZE APP AND MODELS ---
load_dotenv()
app = FastAPI(
    title="Legal Policy Reader API",
    description="API for processing legal documents and answering questions."
)

# Load models once on startup to improve performance
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY or not API_KEY.startswith("AIza"):
    raise ValueError("FATAL: GOOGLE_API_KEY is not set or invalid in .env file.")

LLM, EMBEDDINGS = initialize_gemini_models(API_KEY)


# --- 2. DEFINE REQUEST/RESPONSE MODELS ---
class QueryRequest(BaseModel):
    documents: list[str]
    questions: list[str]

class QueryResponse(BaseModel):
    answers: list[str]


# --- 3. CREATE THE API ENDPOINT ---
@app.post("/hackrx/run", response_model=QueryResponse)
async def process_policy_query(payload: QueryRequest, request: Request):
    # Check for authentication header
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    doc_url = payload.documents[0]
    questions = payload.questions

    # Use temporary directories for isolation and easy cleanup
    temp_db_path = tempfile.mkdtemp()
    local_file_path = None
    
    try:
        # 1. Download document from the URL provided in the request
        local_file_path = download_file_from_url(doc_url)
        if not local_file_path:
            raise HTTPException(status_code=400, detail="Could not download document from URL.")

        # 2. Load the downloaded document using its local path
        documents = load_document_from_path(local_file_path)
        if not documents:
            raise HTTPException(status_code=400, detail="Could not load the downloaded document.")
        
        # 3. Chunk the document
        chunks = chunk_documents_api(documents)
        if not chunks:
            raise HTTPException(status_code=500, detail="Failed to chunk document.")

        # 4. Create vector store in the temporary directory
        vectorstore = create_and_store_embeddings_api(chunks, EMBEDDINGS, temp_db_path)
        if not vectorstore:
            raise HTTPException(status_code=500, detail="Failed to create vector embeddings.")

        # 5. Create QA chain and process all questions
        qa_chain = create_qa_chain(LLM, vectorstore)
        answers = [qa_chain({"query": q}).get("result", "No answer found.") for q in questions]

        return QueryResponse(answers=answers)

    except Exception as e:
        # Catch-all for any unexpected errors
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

    finally:
        # 6. Clean up temporary files and directories to save space
        if local_file_path and os.path.exists(local_file_path):
            os.remove(local_file_path)
        if os.path.exists(temp_db_path):
            shutil.rmtree(temp_db_path)

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Legal Policy Reader API is running."}
