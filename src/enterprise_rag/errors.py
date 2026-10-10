class RAGError(Exception):
    """Base exception for RAG application errors."""
    pass


class ModelUnavailableError(RAGError):
    """External model invocation failed."""
    pass