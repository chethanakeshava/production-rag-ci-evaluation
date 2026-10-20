
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from google.genai.errors import ServerError
from pydantic import BaseModel, Field

from app.rag_pipeline import RAGPipeline

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = RAGPipeline()
    yield


app = FastAPI(
    title="Production RAG Pipeline",
    description="Document question answering with hybrid retrieval.",
    version="0.1.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)
    candidate_k: int = Field(default=10, ge=1, le=50)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents")
async def upload_document(
    file: Annotated[UploadFile, File()],
):
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    content = await file.read(MAX_UPLOAD_BYTES + 1)

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="PDF must be 10 MB or smaller.",
        )

    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    temp_path = None

    try:
        with NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        chunk_count = app.state.pipeline.ingest_pdf(temp_path)

        return {
            "filename": Path(file.filename).name,
            "chunks_indexed": chunk_count,
        }

    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        await file.close()



@app.post("/ask")
def ask_question(request: AskRequest):
    try:
        return app.state.pipeline.ask(
            question=request.question,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except ServerError as exc:
        raise HTTPException(
            status_code=503,
            detail="The AI model is temporarily unavailable. Please try again shortly.",
        ) from exc
