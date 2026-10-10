from enterprise_rag.storage.index import SQLiteIndex

store = SQLiteIndex()

# for table in ["documents", "chunks", "embedding_cache"]:
#     count = store.conn.execute(
#         f"SELECT * FROM {table}"
#     ).fetchone()[0]


#     print(table, count)

# store.conn.close()

from enterprise_rag.storage.index import SQLiteIndex

store = SQLiteIndex()

results = store.conn.execute("""
    SELECT chunk_id, bm25(chunks_fts) AS score
    FROM chunks_fts
    WHERE chunks_fts MATCH ?
    ORDER BY score
    LIMIT 5
""", ("database",)).fetchall()

for row in results:
    print(row)

store.conn.close()