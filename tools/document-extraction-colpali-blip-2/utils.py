import os
import logging
from typing import List
from fastapi import UploadFile
from PIL import Image
import io
from pdf2image import convert_from_bytes, convert_from_path

# Initialize logging
logger = logging.getLogger(__name__)

def load_images(files: List[UploadFile] = None, path: str = None):
    """
    Load images from either uploaded files, a file path (file or folder), or PDFs.
    """
    images = []
    
    if files:
        # Load images from multipart file uploads
        for file in files:
            file_content = file.file.read()
            if file.filename.endswith('.pdf'):
                # Convert PDF to images
                try:
                    pdf_images = convert_from_bytes(file_content)
                    images.extend(pdf_images)
                    logger.info(f"Converted PDF {file.filename} to {len(pdf_images)} images.")
                except Exception as e:
                    logger.error(f"Failed to convert PDF {file.filename} to images: {e}")
            else:
                try:
                    image = Image.open(io.BytesIO(file_content))
                    images.append(image)
                except Exception as e:
                    logger.error(f"Failed to load image from uploaded file: {file.filename}: {e}")
    elif path:
        # Load images from a directory or a single file
        if os.path.isdir(path):
            for file_name in os.listdir(path):
                file_path = os.path.join(path, file_name)
                if file_name.endswith('.pdf'):
                    # Convert PDF to images
                    try:
                        pdf_images = convert_from_path(file_path)
                        images.extend(pdf_images)
                        logger.info(f"Converted PDF {file_name} to {len(pdf_images)} images.")
                    except Exception as e:
                        logger.error(f"Failed to convert PDF {file_name} to images: {e}")
                else:
                    try:
                        image = Image.open(file_path)
                        images.append(image)
                    except Exception as e:
                        logger.error(f"Failed to load {file_name}: {e}")
        elif os.path.isfile(path):
            if path.endswith('.pdf'):
                # Convert PDF to images
                try:
                    pdf_images = convert_from_path(path)
                    images.extend(pdf_images)
                    logger.info(f"Converted PDF {path} to {len(pdf_images)} images.")
                except Exception as e:
                    logger.error(f"Failed to convert PDF {path} to images: {e}")
            else:
                try:
                    image = Image.open(path)
                    images.append(image)
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e}")
    else:
        raise ValueError("Either files or path must be provided.")

    if not images:
        raise ValueError("No valid images found.")

    return images