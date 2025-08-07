import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import requests

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# This print statement will help us confirm if the file is being loaded correctly by the server.
print("--- Loading api_utils.py ---")

def download_file_from_url(url: str) -> str | None:
    """
    Downloads a file from a URL to a temporary local path.
    Includes a User-Agent header to mimic a browser request.
    """
    print(f"Step 1: Starting download from URL: {url}")
    try:
        # Some servers block requests that don't look like they're from a browser.
        # Adding a User-Agent header makes the request look legitimate.
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        
        # Use a timeout to prevent the request from hanging indefinitely.
        # Using stream=True is more memory-efficient for large files.
        with requests.get(url, headers=headers, timeout=60, stream=True) as response:
            # Raise an exception if the download failed (e.g., 404 Not Found, 403 Forbidden).
            response.raise_for_status()

            # Create a temporary file to store the downloaded content.
            # The 'suffix' ensures the file has the correct extension (e.g., .pdf).
            suffix = Path(url).suffix
            with NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                # Write the file to disk in chunks to save memory
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                
                temp_file_path = tmp_file.name
                print(f"File downloaded successfully to temporary path: {temp_file_path}")
                return temp_file_path

    except requests.exceptions.RequestException as e:
        # This will catch any network-related errors during download.
        print(f"[ERROR] Failed to download file. Reason: {e}")
        return None

def load_document_from_path(temp_file_path: str):
    """
    Loads a document from a local file path.
    This is a backend-safe version of your original load_document function.
    """
    print(f"Step 2: Loading document from path: {temp_file_path}")
    file_extension = Path(temp_file_path).suffix.lower().strip('.')
    
    try:
        if file_extension == "pdf":
            loader = PyPDFLoader(temp_file_path)
        elif file_extension in ["docx", "doc"]:
            loader = UnstructuredWordDocumentLoader(temp_file_path)
        elif file_extension == "txt":
            loader = TextLoader(temp_file_path, encoding='utf-8')
        else:
            print(f"[ERROR] Unsupported file type: {file_extension}")
            return None

        documents = loader.load()
        print(f"Document loaded successfully! Found {len(documents)} pages/sections.")
        return documents
    except Exception as e:
        print(f"[ERROR] Failed to load document. Reason: {e}")
        return None

def chunk_documents_api(documents):
    """
    Splits documents into smaller chunks (backend-safe version).
    """
    print("Step 3: Chunking document...")
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
    print("Step 4: Creating and storing embeddings...")
    if not chunks:
        print("[WARNING] No chunks to embed.")
        return None

    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=persist_path
        )
        print("Embeddings stored successfully!")
        return vectorstore
    except Exception as e:
        print(f"[ERROR] Failed to create/store embeddings. Reason: {e}")
        return None

print("--- api_utils.py loaded successfully ---")
