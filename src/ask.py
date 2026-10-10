
import argparse
import os

from enterprise_rag.storage.index import SQLiteIndex
from enterprise_rag.embeddings.provider import BedrockEmbeddingProvider
from enterprise_rag.generation.provider import BedrockLLMProvider
from enterprise_rag.generation.service import RAGService
from enterprise_rag.retrieval.reranker import BedrockReranker
from enterprise_rag.errors import ModelUnavailableError


def print_help():
    print("""
Available commands:
  /help     Show commands
  /clear    Clear terminal
  /exit     Exit
  /quit     Exit

Enter a question to query the knowledge base.
Each question is processed independently.
""")


def print_result(result):
    print("\n" + "=" * 55)
    print("ANSWER")
    print("=" * 55)
    print(result["answer"])

    citations = result.get("citations", [])

    if citations:
        print("\nCITATIONS:")

        for citation in citations:
            print(
                f"[{citation['source_id']}] "
                f"{citation['document_id']} "
                f"(offsets "
                f"{citation['start_offset']}:"
                f"{citation['end_offset']})"
            )

    else:
        print("\nCITATIONS: None")

    metrics = result.get("metrics") or {}
    usage = result.get("usage") or {}

    print("\nMETRICS:")
    print(f"Request ID      : {result.get('request_id', '-')}")
    print(f"Retrieval       : {metrics.get('retrieval_ms', 0) / 1000:.2f}s")
    print(f"Generation      : {metrics.get('generation_ms', 0) / 1000:.2f}s")
    print(f"Total latency   : {metrics.get('total_ms', 0) / 1000:.2f}s")
    print(f"Retrieved chunks: {metrics.get('candidate_count', 0)}")
    print(f"Input tokens    : {usage.get('inputTokens', 0)}")
    print(f"Output tokens   : {usage.get('outputTokens', 0)}")

    print("-" * 55)


def interactive_session(rag, tenant_id, groups):
    print("\n" + "=" * 55)
    print("           ENTERPRISE RAG ASSISTANT")
    print("=" * 55)
    print(f"Tenant    : {tenant_id}")
    print(f"Groups    : {', '.join(groups)}")
    print(f"LLM       : {rag.llm_provider.model_id}")
    print(
        f"Reranking : "
        f"{'Enabled' if rag.reranker else 'Disabled'}"
    )
    print("Type /help for commands or /exit to quit.")
    print("=" * 55)

    while True:
        try:
            question = input("\nrag> ").strip()

            if not question:
                continue

            command = question.lower()

            if command in ("/exit", "/quit"):
                print("\nGoodbye!")
                break

            if command == "/help":
                print_help()
                continue

            if command == "/clear":
                print("\033[2J\033[H", end="", flush=True)
                continue

            result = rag.ask(
                question=question,
                tenant_id=tenant_id,
                allowed_groups=groups,
            )

            print_result(result)

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        except ValueError as exc:
            print(f"\nInvalid request: {exc}")

        except ModelUnavailableError as exc:
            print(f"\nModel unavailable: {exc}")
            print(
                "Check your AWS permissions, model access "
                "and connectivity."
            )

        except Exception as exc:
            print(
                f"\nRequest failed: "
                f"{type(exc).__name__}"
            )
            print("Check the structured error logs.")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Enterprise RAG CLI"
    )

    parser.add_argument(
        "--tenant",
        required=True,
        help="Tenant identifier",
    )

    parser.add_argument(
        "--groups",
        nargs="+",
        required=True,
        help="Groups for this trusted local test session",
    )

    parser.add_argument(
        "--region",
        default=os.getenv("AWS_REGION", "us-east-1"),
    )

    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable Bedrock reranking",
    )

    args = parser.parse_args()

    model_id = os.getenv("RAG_LLM_MODEL_ID")

    if not model_id:
        parser.error(
            "RAG_LLM_MODEL_ID environment variable is missing"
        )

    store = SQLiteIndex()

    try:
        embedding_provider = BedrockEmbeddingProvider(
            region=args.region
        )

        llm_provider = BedrockLLMProvider(
            model_id=model_id,
            region=args.region,
        )

        reranker = (
            BedrockReranker(region=args.region)
            if args.rerank
            else None
        )

        rag = RAGService(
            store=store,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            reranker=reranker,
        )

        interactive_session(
            rag=rag,
            tenant_id=args.tenant,
            groups=args.groups,
        )

    finally:
        store.conn.close()


if __name__ == "__main__":
    main()
