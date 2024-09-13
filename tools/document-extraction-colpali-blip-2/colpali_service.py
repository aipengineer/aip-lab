import os
import torch
from dotenv import load_dotenv
from colpali_engine.models import ColPali, ColPaliProcessor
from transformers import BlipProcessor, BlipForConditionalGeneration
import logging

# Initialize logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get Hugging Face token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set in the environment or .env file.")

# Set the device to 'mps' for Apple Silicon or fallback to 'cpu'
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")

def retrieve_closest_document(query: str, documents: list, batch_size: int = 4):
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
            use_auth_token=HF_TOKEN      # Hugging Face Token for authentication
        )
        processor = ColPaliProcessor.from_pretrained(
            "google/paligemma-3b-mix-448",
            use_auth_token=HF_TOKEN
        )
    except Exception as e:
        logger.error(f"Failed to load ColPali model or processor: {e}")
        raise

    logger.info("Processing images and queries in batches.")

    # Process the query
    batch_queries = processor.process_queries([query]).to(device)

    # Debug the shape of query embeddings
    logger.info(f"Shape of query embeddings: {batch_queries['input_ids'].shape}")

    # Initialize lists to store embeddings and documents
    all_image_embeddings = []
    all_documents = []

    # Process documents in batches
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1} of {len(documents) // batch_size + 1}")

        # Process images and move them to the device
        batch_images = processor.process_images(batch).to(device, dtype=torch.bfloat16)

        with torch.no_grad():
            # Compute embeddings for the batch of images
            image_embeddings = model(**batch_images)

        # Debug the shape of image embeddings
        logger.info(f"Shape of image embeddings for batch {i // batch_size + 1}: {image_embeddings.shape}")

        # Collect embeddings and documents
        all_image_embeddings.append(image_embeddings)
        all_documents.extend(batch)

    # Concatenate all image embeddings
    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)

    # Debug final concatenated image embeddings
    logger.info(f"Shape of concatenated image embeddings: {all_image_embeddings.shape}")

    # Convert the processor's feature output to a tensor and cast it to float32
    batch_queries_tensor = batch_queries['input_ids'].to(device, dtype=torch.float32)

    # Ensure that the query embeddings are projected to the same dimension as the image embeddings
    if batch_queries_tensor.shape[-1] != all_image_embeddings.shape[-1]:
        logger.info(f"Projecting query embeddings to match image embeddings.")
        query_projection = torch.nn.Linear(batch_queries_tensor.shape[-1], all_image_embeddings.shape[-1], device=device)

        # Apply the linear projection in float32 and cast back to bfloat16 if necessary
        batch_queries_tensor = query_projection(batch_queries_tensor).to(torch.bfloat16)

    # Compute similarity scores between the query and all image embeddings
    with torch.no_grad():
        # Check the dimensions before calculating similarity
        logger.info(f"Query tensor shape: {batch_queries_tensor.shape}, Image embeddings shape: {all_image_embeddings.shape}")

        scores = processor.score_multi_vector(batch_queries_tensor, all_image_embeddings)

    # Debug the shape of the similarity scores
    logger.info(f"Shape of similarity scores: {scores.shape}")

    # Find the best matching document
    best_match_index = torch.argmax(scores).item()
    best_match_document = all_documents[best_match_index]

    logger.info(f"Closest document found at index {best_match_index} with score: {scores[best_match_index]}")

    return best_match_document


def generate_response(image, text_query: str) -> str:
    """
    Use BLIP-2 to generate a response from the closest document and query.
    """
    logger.info("Generating response using BLIP-2.")

    # Load the BLIP-2 processor and model
    try:
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl", use_auth_token=HF_TOKEN)
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", use_auth_token=HF_TOKEN)
    except Exception as e:
        logger.error(f"Failed to load BLIP-2 processor or model: {e}")
        raise

    # Move image and query to device (MPS/CPU) and use mixed precision
    inputs = blip_processor(images=image, text=text_query, return_tensors="pt").to(device, dtype=torch.bfloat16)

    with torch.no_grad():
        out = blip_model.generate(**inputs)

    # Decode the response
    response = blip_processor.decode(out[0], skip_special_tokens=True)

    return response