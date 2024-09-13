# Multimodal Document Retrieval and Question-Answering System

This project is a multimodal document retrieval and question-answering system that uses **ColPali** and **BLIP-2** models for document retrieval and response generation. It supports two main modes of operation:

1. **CLI Mode**: Allows querying of a folder or file of documents (images) directly from the command line.
2. **FastAPI REST Server**: Provides a REST API for querying documents and generating responses through HTTP requests.

The system is designed to handle multimodal datasets, including images and text, and is suitable for **retrieval-augmented generation (RAG)** tasks.

## Features

- **Multimodal Document Retrieval**: Uses **ColPali** to retrieve the most relevant document based on text queries.
- **Response Generation**: Leverages **BLIP-2** to generate context-aware answers from the retrieved document and the input query.
- **Command Line Interface (CLI)**: Query documents from your local system.
- **REST API**: Expose document-query functionality over HTTP via a **FastAPI** server.

## Installation

To install the dependencies and set up the project, you will need **Poetry** for dependency management. Follow the instructions below to get started:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Install Dependencies

```bash
poetry install
```

This will install all the necessary dependencies, including **ColPali**, **BLIP-2**, **FastAPI**, and **Torch**.

## Usage

You can run the project in two different modes: **CLI Mode** for direct querying, or **Server Mode** to expose a REST API.

### CLI Mode

You can query a folder or file of documents and get a response based on a text query.

#### Example Usage:

```bash
poetry run python app.py query /path/to/folder "Explain the chart trends in the document"
```

In this command:
- `/path/to/folder` is the path to the folder or file of images (documents) to query.
- `"Explain the chart trends in the document"` is the text query.

The response will be printed to the terminal.

### FastAPI Server Mode

You can start the FastAPI server to expose the document-query functionality over HTTP.

#### Start the Server:

```bash
poetry run python app.py server
```

The server will be running on `http://0.0.0.0:8000`.

#### Available Endpoints:

- **Health Check**: `GET /status/health`
  - This endpoint checks the server's health.
  - Example Response:
    ```json
    { "status": "healthy" }
    ```

- **Document Query**: `POST /v1/query`
  - This endpoint accepts a multipart form with files (documents) and a query.
  - Example Request using `curl`:
    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/v1/query' \
      -F 'files=@/path/to/image1.png' \
      -F 'files=@/path/to/image2.png' \
      -F 'query=Explain the chart trends in the document'
    ```
  - Example Response:
    ```json
    { "response": "The document shows an upward trend in sales over the last quarter." }
    ```

## Project Structure

- `app.py`: The main application file that supports both CLI and FastAPI server modes.
- `pyproject.toml`: Contains the project dependencies and configuration for **Poetry**.

## Contributors

- **Jens Weber** - [jens@aip.engineer](mailto:jens@aip.engineer)

## Website

Visit our website: [https://aiproduct.engineer](https://aiproduct.engineer)
