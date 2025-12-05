import time
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.schemas.summarize_request import SummarizeRequest
from app.schemas.summarize_response import SummarizeResponse
from app.services.summarization_service import summarize_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Text Summarizer",
    description="A local, privacy-focused AI text summarizer backend.",
    version="1.0.0"
)

# Add CORS middleware to allow frontend to access the API
# This is critical for file:// frontends to talk to localhost backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler to ensure we always return a JSON error
    instead of crashing or returning a generic 500 HTML page.
    """
    logger.error(f"Global exception occurred: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"},
    )

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {"status": "ok"}

@app.post("/api/v1/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """
    Summarize the provided text based on the requested length.
    """
    start_time = time.time()
    logger.info(f"Received summarization request. Length: {len(request.text)} chars. Preset: {request.length}")

    try:
        # Basic validation
        if not request.text.strip():
             raise HTTPException(status_code=400, detail="Text cannot be empty.")

        response = await summarize_text(request)
        
        elapsed = time.time() - start_time
        logger.info(f"Request completed in {elapsed:.2f}s")
        return response

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
