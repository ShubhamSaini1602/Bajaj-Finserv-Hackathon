import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import requests

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

def download_file_from_url(url: str) -> str | None:
    """
    Downloads a file from a URL to a temporary local path.
    Includes a User-Agent header to mimic a browser request.
    """
    # --- THIS IS THE FIX ---
    # Some servers block requests that don't look like they're from a browser.
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    # -----------------------

    try:
        # Use a timeout and the new headers
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status() # Raise an HTTPError for bad responses

        # Create a temporary file with the correct extension
        suffix = Path(url).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(response.content)
            print(f"File downloaded successfully to: {tmp_file.name}")
            return tmp_file.name
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file from {url}: {e}")
        return None

def load_document_from_path(temp_file_path: str):
    """
    Loads a document from a local file path.
    This is a backend-safe version of your original load_document function.
    """
    file_extension = Path(temp_file_path).suffix.lower().strip('.')
    print(f"Loading document from path: {temp_file_path} with extension: {file_extension}")

    try:
        if file_extension == "pdf":
            loader = PyPDFLoader(temp_file_path)
        elif file_extension in ["docx", "doc"]:
            loader = UnstructuredWordDocumentLoader(temp_file_path)
        elif file_extension == "txt":
            loader = TextLoader(temp_file_path, encoding='utf-8')
        else:
            print(f"Unsupported file type: {file_extension}")
            return None

        documents = loader.load()
        print(f"Document loaded successfully! Found {len(documents)} pages/sections.")
        return documents
    except Exception as e:
        print(f"Error loading document: {str(e)}")
        return None

def chunk_documents_api(documents):
    """
    Splits documents into smaller chunks (backend-safe version).
    """
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Document split into {len(chunks)} chunks.")
    return chunks

def create_and_store_embeddings_api(chunks, embeddings_model, persist_path):
    """
    Creates embeddings and stores them in ChromaDB (backend-safe version).
    """
    if not chunks:
        print("No chunks to embed.")
        return None

    try:
        print("Creating embeddings and storing in vector database...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=persist_path
        )
        print("Embeddings stored successfully!")
        return vectorstore
    except Exception as e:
        print(f"Error creating/storing embeddings: {str(e)}")
        return None
