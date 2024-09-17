import os
import torch
from dotenv import load_dotenv
from colpali_engine.models import ColPali, ColPaliProcessor
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import logging
from PIL.Image import Image

# Initialize logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get Hugging Face token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set in the environment or .env file.")

# Set the device to 'mps' for Apple Silicon
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")

def retrieve_closest_document(query: str, documents: list[Image], batch_size: int = 4) -> Image:
    """
    Use ColPali to find the closest matching document to a query, with batching to avoid memory issues.
    """
    logger.info("Initializing ColPali model for document retrieval.")

    # Load the ColPali model with mixed precision (bfloat16) for MPS backend
    try:
        model = ColPali.from_pretrained(
            "vidore/colpali-v1.2",
            torch_dtype=torch.bfloat16,  # Use mixed precision to save memory
            device_map=device,           # MPS for Apple Silicon
            token=HF_TOKEN      # Hugging Face Token for authentication
        )
        processor = ColPaliProcessor.from_pretrained(
            "google/paligemma-3b-mix-448",
            token=HF_TOKEN
        )
    except Exception as e:
        logger.error(f"Failed to load ColPali model or processor: {e}")
        raise

    logger.info("Processing images and queries in batches.")

    # In theory we could support multiple questions at once for this algorithm check test-col-pali
    queries = [query]
    # Process the query
    batch_queries = processor.process_queries(queries).to(device)
    with torch.no_grad():
        all_query_embeddings = model(**batch_queries)

    # Initialize lists to store embeddings and documents
    all_image_embeddings = []
    all_documents : list[Image] = []

    # Process documents in batches
    for i in range(0, len(documents), batch_size):
        batch_of_documents = documents[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1} of {len(documents) // batch_size + 1}")

        # Process images and move them to the device
        batch_images = processor.process_images(batch_of_documents).to(model.device)

        with torch.no_grad():
            # Compute embeddings for the batch of images
            image_embeddings = model(**batch_images)

        # Debug the shape of image embeddings
        logger.info(f"Shape of image embeddings for batch {i // batch_size + 1}: {image_embeddings.shape}")

        # Collect embeddings and documents
        all_image_embeddings.extend(image_embeddings)
        all_documents.extend(batch_of_documents)

    scores = processor.score_multi_vector(all_query_embeddings, all_image_embeddings)
    # Debug the shape of the similarity scores
    logger.info(f"Shape of similarity scores: {scores.shape} {scores}")

    # The result is a multiarray tensor depending on the queries and number of images
    # # Ouput for 3 Images and 2 Queries => per Question one entry with scores for each image => tensor([
    #   [10.5625,  7.6875, 12.6875],
    #   [10.3750,  9.3750,  9.1875]
    #])
    # Asserting because we take the shortcut of assuming it is only one query.
    assert len(queries) == 1
    first_query_score = scores[0]
    # Find the best matching document    
    best_match_index = torch.argmax(first_query_score).item()
    best_match_document = all_documents[best_match_index]

    logger.info(f"Closest document found at index {best_match_index} with score: {first_query_score[best_match_index]}")

    return best_match_document


def generate_response(image, text_query: str) -> str:
    """
    Use BLIP-2 to generate a response from the closest document and query.
    """
    logger.info("Generating response using BLIP-2.")

    # Load the BLIP-2 processor and model
    try:
        blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16, device_map="auto")
    except Exception as e:
        logger.error(f"Failed to load BLIP-2 processor or model: {e}")
        raise

    # Move image and query to device (MPS/CPU) and use mixed precision
    inputs = blip_processor(images=image, text=text_query, return_tensors="pt").to(blip_model.device, dtype=torch.bfloat16)

    with torch.no_grad():
        out = blip_model.generate(**inputs)

    # Decode the response
    response = blip_processor.decode(out[0], skip_special_tokens=True)

    return response