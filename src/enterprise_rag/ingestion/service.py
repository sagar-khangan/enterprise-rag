
from pathlib import Path

from enterprise_rag.models import Document, Chunk
from enterprise_rag.ingestion.parser import parse_markdown
from enterprise_rag.ingestion.chunker import chunk_document


def ingest_document(
    path: Path,
    document_id: str,
    tenant_id: str,
    allowed_groups: list[str],
    version: int = 1,
) -> tuple[Document, list[Chunk]]:

    text, content_hash = parse_markdown(path)

    document = Document(
        document_id=document_id,
        tenant_id=tenant_id,
        title=path.stem,
        source_path=str(path),
        allowed_groups=allowed_groups,
        version=version,
        content_hash=content_hash,
    )

    chunks = chunk_document(
        document=document,
        text=text,
        chunk_size=500,
        overlap=100,
    )

    return document, chunks