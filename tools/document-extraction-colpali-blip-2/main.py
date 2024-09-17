import argparse
import uvicorn
from app import app
from utils import load_images
from colpali_service import retrieve_closest_document, generate_response
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CLI command to run document retrieval and response generation
def run_query_cli(path: str, query: str):
    """
    Execute the CLI command to retrieve the closest document and generate a response.
    """
    try:
        documents = load_images(path=path)
        closest_document = retrieve_closest_document(query, documents)
        #closest_document.show()
        response = generate_response(closest_document, query)
        logger.info(f"Generated response: {response}")
    except Exception as e:
        logger.exception(f"Error: {e}")

# CLI function to start FastAPI server
def run_server():
    """
    Start the FastAPI server.
    """
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Main function to handle both CLI and server commands
def main():
    """
    Main CLI function for handling 'query' and 'server' commands.
    """
    parser = argparse.ArgumentParser(description="Document Query and FastAPI Server")
    subparsers = parser.add_subparsers(dest="command")

    # CLI query command
    query_parser = subparsers.add_parser("query", help="Query documents from a folder or file.")
    query_parser.add_argument("path", type=str, help="Path to a document or folder with documents (images or PDFs).")
    query_parser.add_argument("query", type=str, help="The query or question to ask about the document(s).")

    # Server command
    server_parser = subparsers.add_parser("server", help="Start the FastAPI server.")

    args = parser.parse_args()

    try:
        if args.command == "query":
            run_query_cli(args.path, args.query)
        elif args.command == "server":
            run_server()
        else:
            parser.print_help()
    except Exception as e:
        logger.exception(f'Main Error {e}')
        raise

if __name__ == "__main__":
    main()