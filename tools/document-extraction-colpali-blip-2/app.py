from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from utils import load_images
from colpali_service import retrieve_closest_document, generate_response
import logging

# Initialize FastAPI app
app = FastAPI()

# Initialize logging
logger = logging.getLogger(__name__)

@app.get("/status/health")
async def health_check():
    """
    Health check endpoint.
    """
    logger.info("Health check endpoint accessed.")
    return {"status": "healthy"}

@app.post("/v1/query")
async def query_documents(query: str = Form(...), files: list[UploadFile] = File(...)):
    """
    Endpoint to process document query via FastAPI.
    """
    try:
        logger.info("Received query and documents for processing.")
        documents = load_images(files=files)
        closest_document = retrieve_closest_document(query, documents)
        response = generate_response(closest_document, query)
        return JSONResponse(content={"response": response})
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)