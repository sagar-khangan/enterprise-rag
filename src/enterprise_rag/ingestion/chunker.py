from enterprise_rag.models import Document, Chunk


def chunk_document(
    document: Document,
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Chunk]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        chunk_text = text[start:end]

        if chunk_text.strip():
            chunks.append(
                Chunk(
                    chunk_id=(
                        f"{document.tenant_id}:"
                        f"{document.document_id}:"
                        f"v{document.version}:"
                        f"{index}"
                    ),
                    document_id=document.document_id,
                    tenant_id=document.tenant_id,
                    text=chunk_text,
                    chunk_index=index,
                    allowed_groups=document.allowed_groups.copy(),
                    start_offset=start,
                    end_offset=end,
                )
            )
            index += 1

        if end == len(text):
            break

        start = end - overlap

    return chunks