from pathlib import Path

from enterprise_rag.ingestion.indexer import index_document
from enterprise_rag.storage.index import SQLiteIndex

from enterprise_rag.embeddings.provider import (
    BedrockEmbeddingProvider,FakeEmbeddingProvider
)

provider = BedrockEmbeddingProvider(
    region="us-east-1"
)
# provider = FakeEmbeddingProvider()

store = SQLiteIndex()


result = index_document(
    path=Path("data/raw/database_incident.md"),
    document_id="incident-001",
    tenant_id="acme",
    allowed_groups=["sre", "platform"],
    provider=provider,
    store=store,
)

print(result)

store.conn.close()