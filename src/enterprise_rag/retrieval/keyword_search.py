import json
import re


def make_fts_query(question: str) -> str:
    tokens = re.findall(
        r"[\w]+",
        question,
        flags=re.UNICODE,
    )

    if not tokens:
        return ""

    # Quoted terms avoid interpreting words as FTS syntax.
    # Limit terms to avoid excessively large queries.
    terms = [f'"{token}"' for token in tokens[:20]]

    return " OR ".join(terms)


def keyword_search(
    store,
    question,
    tenant_id,
    allowed_groups,
    top_k=10,
):
    match_query = make_fts_query(question)

    if not match_query:
        return []

    rows = store.conn.execute(
        """
        SELECT c.chunk_id,
               c.metadata_json,
               bm25(chunks_fts) AS score
        FROM chunks_fts
        JOIN chunks c
          ON c.chunk_id = chunks_fts.chunk_id
        JOIN documents d
          ON d.tenant_id = c.tenant_id
         AND d.document_id = c.document_id
        WHERE chunks_fts MATCH ?
          AND c.tenant_id = ?
          AND EXISTS (
              SELECT 1
              FROM json_each(
                  d.metadata_json, '$.allowed_groups'
              ) AS g
              WHERE g.value IN (
                  SELECT value FROM json_each(?)
              )
          )
        ORDER BY score ASC, c.chunk_id ASC
        LIMIT ?
        """,
        (
            match_query,
            tenant_id,
            json.dumps(allowed_groups),
            top_k,
        ),
    ).fetchall()

    return [
        {
            "chunk_id": chunk_id,
            "score": score,
            "chunk": json.loads(metadata),
        }
        for chunk_id, metadata, score in rows
    ]