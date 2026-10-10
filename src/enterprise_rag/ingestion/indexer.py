import hashlib

from enterprise_rag.ingestion.service import ingest_document


def embedding_cache_key(text, model_id, dimensions):
    text_hash = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    raw_key = f"{model_id}:{dimensions}:{text_hash}"

    return hashlib.sha256(
        raw_key.encode("utf-8")
    ).hexdigest()


def index_document(
    path,
    document_id,
    tenant_id,
    allowed_groups,
    provider,
    store,
    version=1,
):
    document, chunks = ingest_document(
        path=path,
        document_id=document_id,
        tenant_id=tenant_id,
        allowed_groups=allowed_groups,
        version=version,
    )

    prepared = []
    hits = 0
    misses = 0

    for chunk in chunks:
        key = embedding_cache_key(
            chunk.text,
            provider.model_id,
            provider.dimensions,
        )

        vector = store.get_embedding(key)

        if vector is None:
            vector = provider.embed(chunk.text)

            if len(vector) != provider.dimensions:
                raise ValueError(
                    "Embedding dimensions mismatch"
                )

            store.put_embedding(
                key,
                provider.model_id,
                vector,
            )
            misses += 1
        else:
            hits += 1

        prepared.append((chunk, key))

    store.replace_document(document, prepared)

    return {
        "document_id": document_id,
        "chunks": len(chunks),
        "cache_hits": hits,
        "cache_misses": misses,
    }