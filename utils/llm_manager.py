import os
import asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

def initialize_gemini_models(api_key):
    """
    Initializes and returns the Gemini LLM and Embeddings model.
    This version includes a robust fix for the asyncio event loop issue in Streamlit.
    """
    print("🔧 Initializing Gemini models...")

    # --- Event Loop Fix for Streamlit ---
    # This is the key change to prevent the 'no current event loop' error.
    # It ensures an event loop is running before the Google libraries, which use grpc, are called.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    # ------------------------------------

    if not api_key or not api_key.strip() or not api_key.startswith("AIza"):
        print("❌ FATAL: GOOGLE_API_KEY is not set, empty, or invalid.")
        raise ValueError("Invalid or missing GOOGLE_API_KEY.")

    try:
        # Initialize the main LLM for chat
        llm_instance = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            api_key=api_key
        )

        # Initialize the embeddings model
        embeddings_instance = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )

        print("✅ Gemini models initialized successfully!")
        return llm_instance, embeddings_instance

    except Exception as e:
        # Catch any exception during initialization
        print(f"❌ Error initializing Gemini models: {str(e)}")
        print("Please check that your API key is valid and has the necessary permissions.")
        # Re-raise the exception to stop the application from starting with faulty models
        raise e
