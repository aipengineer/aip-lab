import os
from typing import cast
import torch
from PIL import Image
from dotenv import load_dotenv
import requests
from io import BytesIO

from colpali_engine.models import ColPali, ColPaliProcessor
import logging

# Load environment variables from .env file
load_dotenv()

# Get Hugging Face token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set in the environment or .env file.")

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")


def open_image_from_url(url):
    try:
        # Make the HTTP request to get the image
        response = requests.get(url, stream=True)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Check if the Content-Type is an image (png in this case)
            if 'image/png' in response.headers['Content-Type']:
                # Open the image using PIL from the response content
                img = Image.open(BytesIO(response.content)).convert('RGBA')
                ##img.show()  # Display the image or do any further processing
                # Add white background
                return Image.composite(img, Image.new('RGB', img.size, 'white'), img)
            elif 'image/jpeg' in response.headers['Content-Type']:
                # Open the image using PIL from the response content
                img = Image.open(BytesIO(response.content))
                ##img.show()  # Display the image or do any further processing
                # Add white background
                return img
            else:
                raise ValueError(f"Error: URL did not return a supported image. Content-Type: {response.headers['Content-Type']}")

        else:
            raise ValueError(f"Error: Failed to fetch image. HTTP Status Code: {response.status_code}")
    except Exception as e:
        logger.exception(f"Error: {url} {e}")
        raise

model = cast(
    ColPali,
    ColPali.from_pretrained(
        "vidore/colpali-v1.2",
        torch_dtype=torch.bfloat16,
        device_map=device
    ),
)

processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained("google/paligemma-3b-mix-448", token=HF_TOKEN))

images_url = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/sea_and_island.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat.jpg"
]

# Your inputs
images = list(map(open_image_from_url, images_url))

queries = [
    "Is this a car?",
    "Can you see a stop sign?",
    "Can you see a car?",
    "Can you see a Lion?"
]

# Process the inputs
batch_images = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    querry_embeddings = model(**batch_queries)

# Ouput for 3 Images and 2 Queries => per Question one entry with scores for each image => tensor([
#   [10.5625,  7.6875, 12.6875],
#   [10.3750,  9.3750,  9.1875]
#])
scores = processor.score_multi_vector(querry_embeddings, image_embeddings)

# Overall best matching image, not per query.
best_images = torch.argmax(scores, dim=1)

logger.info('Results:')

logger.info(f'Best Matching Images: {best_images}')  # Output: tensor([0, 1])

# Find the best matching document
best_match_index = torch.argmax(scores).item()

for i, query in enumerate(queries):
    # Get the best matching image for this query
    best_image = torch.argmax(scores[i])  # i-th query
    best_score = scores[i, best_image]
    logger.info(f"{query}: Best Image = Image {best_image.item()} {images_url[best_image.item()]}, Score = {best_score.item()}")

