import json

import numpy as np


def vector_search(
    store,
    query_vector,
    tenant_id,
    allowed_groups,
    model_id,
    dimensions,
    top_k=10,
):
    query = np.asarray(query_vector, dtype=np.float32)

    if query.size != dimensions:
        raise ValueError("Query dimensions mismatch")

    query_norm = np.linalg.norm(query)

    if query_norm == 0:
        raise ValueError("Zero query vector")

    # Filter by tenant, permissions and model BEFORE scoring.
    rows = store.conn.execute(
        """
        SELECT c.chunk_id,
               c.metadata_json,
               e.vector
        FROM chunks c
        JOIN documents d
          ON d.tenant_id = c.tenant_id
         AND d.document_id = c.document_id
        JOIN embedding_cache e
          ON e.cache_key = c.cache_key
        WHERE c.tenant_id = ?
          AND e.model_id = ?
          AND e.dimensions = ?
          AND EXISTS (
              SELECT 1
              FROM json_each(
                  d.metadata_json, '$.allowed_groups'
              ) AS g
              WHERE g.value IN (
                  SELECT value FROM json_each(?)
              )
          )
        """,
        (
            tenant_id,
            model_id,
            dimensions,
            json.dumps(allowed_groups),
        ),
    ).fetchall()

    results = []

    for chunk_id, metadata, vector_blob in rows:
        vector = np.frombuffer(
            vector_blob, dtype=np.float32
        )

        if vector.size != dimensions:
            raise ValueError("Corrupt stored embedding")

        denominator = (
            query_norm * np.linalg.norm(vector)
        )

        score = (
            float(np.dot(query, vector) / denominator)
            if denominator > 0 else 0.0
        )

        results.append({
            "chunk_id": chunk_id,
            "score": score,
            "chunk": json.loads(metadata),
        })

    results.sort(
        key=lambda item: (-item["score"], item["chunk_id"])
    )

    return results[:top_k]