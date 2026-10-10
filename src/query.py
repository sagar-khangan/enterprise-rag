import argparse

from enterprise_rag.storage.index import SQLiteIndex
from enterprise_rag.embeddings.provider import (
    FakeEmbeddingProvider,
    BedrockEmbeddingProvider,
)
from enterprise_rag.retrieval.hybrid import hybrid_search


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("question")
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument(
        "--provider",
        choices=["fake", "bedrock"],
        default="fake",
    )

    args = parser.parse_args()

    provider = (
        BedrockEmbeddingProvider()
        if args.provider == "bedrock"
        else FakeEmbeddingProvider()
    )

    store = SQLiteIndex()

    try:
        results = hybrid_search(
            store=store,
            provider=provider,
            question=args.question,
            tenant_id=args.tenant,
            allowed_groups=args.groups,
        )

        if not results:
            print("No matching authorized documents.")
            return

        for rank, result in enumerate(results, 1):
            chunk = result["chunk"]

            print(f"\n--- Result {rank} ---")
            print(f"Document: {chunk['document_id']}")
            print(f"Chunk: {chunk['chunk_id']}")
            print(f"RRF Score: {result['score']:.5f}")
            print(
                "Citation: "
                f"{chunk['document_id']}"
                f"[{chunk['start_offset']}:"
                f"{chunk['end_offset']}]"
            )
            print(chunk["text"][:400])

    finally:
        store.conn.close()


if __name__ == "__main__":
    main()