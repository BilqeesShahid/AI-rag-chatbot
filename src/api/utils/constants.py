import os

# Configuration constants for the RAG system

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "https://59a61c34-39c5-4424-889d-21ebdbce5b6d.sa-east-1-0.aws.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.TcJLMaUiB19eoi5ZiWVkcqn-Gadg7FxOR7ejaql5ck0")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "book_embeddings")

# Google Generative AI Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCS9Nm9G9q6ldsnw5eyQoGPthmlvoeWUyw")

# Local Embedding Configuration - Use 768 to match existing collection
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))

# Document Processing Configuration
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))  # words per chunk
DEFAULT_OVERLAP = 50  # words overlap between chunks
MAX_CHUNK_SIZE = 1000  # maximum words per chunk
MIN_CHUNK_SIZE = 100  # minimum words per chunk

# API Configuration
DEFAULT_CONTEXT_WINDOW = 5  # number of previous messages to include in context
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "1000"))  # maximum context length in tokens
MAX_QUERY_LENGTH = 1000  # maximum query length in characters

# Performance Configuration
MAX_RESPONSE_TIME = 0.5  # maximum response time in seconds (500ms)
DEFAULT_TIMEOUT = 30  # default timeout in seconds

# Language Configuration
SUPPORTED_LANGUAGES = ["en", "ur"]  # English and Urdu
DEFAULT_LANGUAGE = "en"

# Collection Configuration
VECTOR_COLLECTION_NAME = QDRANT_COLLECTION_NAME
FASTAPI_ENV = os.getenv("FASTAPI_ENV", "development")