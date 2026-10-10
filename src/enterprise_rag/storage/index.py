import sqlite3
from pathlib import Path

import numpy as np

from enterprise_rag.models import Document, Chunk


class SQLiteIndex:
    def __init__(self, db_path="/home/ubuntu/workspace/practice/enterprise-rag/data/index/rag.db"):
        Path(db_path).parent.mkdir(
            parents=True, exist_ok=True
        )

        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")

        self.create_tables()

    def create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                tenant_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                PRIMARY KEY (tenant_id, document_id)
            );

            CREATE TABLE IF NOT EXISTS embedding_cache (
                cache_key TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector BLOB NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                cache_key TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                FOREIGN KEY (tenant_id, document_id)
                    REFERENCES documents(tenant_id, document_id),
                FOREIGN KEY (cache_key)
                    REFERENCES embedding_cache(cache_key)
            );



            CREATE INDEX IF NOT EXISTS idx_chunks_tenant
            ON chunks(tenant_id, document_id);
        """)

        self.conn.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(
                chunk_id UNINDEXED,
                text,
                tokenize='unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS chunks_fts_delete
            AFTER DELETE ON chunks
            BEGIN
                DELETE FROM chunks_fts
                WHERE chunk_id = OLD.chunk_id;
            END;
        """)

    def get_embedding(self, cache_key: str):
        row = self.conn.execute(
            """
            SELECT vector, dimensions
            FROM embedding_cache
            WHERE cache_key = ?
            """,
            (cache_key,),
        ).fetchone()

        if row is None:
            return None

        vector = np.frombuffer(
            row[0], dtype=np.float32
        ).copy()

        if vector.size != row[1]:
            raise ValueError("Cached vector dimensions mismatch")

        return vector

    def put_embedding(
        self,
        cache_key: str,
        model_id: str,
        vector: np.ndarray,
    ):
        vector = np.asarray(vector, dtype=np.float32)

        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("Invalid embedding vector")

        with self.conn:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO embedding_cache
                (cache_key, model_id, dimensions, vector)
                VALUES (?, ?, ?, ?)
                """,
                (
                    cache_key,
                    model_id,
                    vector.size,
                    vector.tobytes(),
                ),
            )

    def replace_document(
        self,
        document: Document,
        chunks: list[tuple[Chunk, str]],
    ):
        with self.conn:
            self.conn.execute(
                """
                DELETE FROM chunks
                WHERE tenant_id = ? AND document_id = ?
                """,
                (document.tenant_id, document.document_id),
            )

            self.conn.execute(
                """
                INSERT INTO documents
                (tenant_id, document_id, version,
                 content_hash, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, document_id)
                DO UPDATE SET
                    version = excluded.version,
                    content_hash = excluded.content_hash,
                    metadata_json = excluded.metadata_json
                """,
                (
                    document.tenant_id,
                    document.document_id,
                    document.version,
                    document.content_hash,
                    document.model_dump_json(),
                ),
            )

            for chunk, cache_key in chunks:
                if (
                    chunk.document_id != document.document_id
                    or chunk.tenant_id != document.tenant_id
                ):
                    raise ValueError("Chunk parent mismatch")

                self.conn.execute(
                    """
                    INSERT INTO chunks
                    (chunk_id, tenant_id, document_id,
                     chunk_index, cache_key, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.tenant_id,
                        chunk.document_id,
                        chunk.chunk_index,
                        cache_key,
                        chunk.model_dump_json(),
                    ),
                )
                self.conn.execute(
                    """
                    INSERT INTO chunks_fts (chunk_id, text)
                    VALUES (?, ?)
                    """,
                    (chunk.chunk_id, chunk.text),
                )