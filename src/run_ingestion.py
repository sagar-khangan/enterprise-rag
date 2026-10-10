from pathlib import Path

from enterprise_rag.ingestion.service import ingest_document


document, chunks = ingest_document(
    path=Path("data/raw/database_incident.md"),
    document_id="incident-001",
    tenant_id="acme",
    allowed_groups=["sre", "platform"],
)

print(f"Document: {document.title}")
print(f"SHA-256: {document.content_hash}")
print(f"Total chunks: {len(chunks)}")

for chunk in chunks:
    print(f"\n--- Chunk {chunk.chunk_index} ---")
    print(f"ID: {chunk.chunk_id}")
    print(f"Offsets: {chunk.start_offset}:{chunk.end_offset}")
    print(chunk.text)