
import argparse
import csv
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from enterprise_rag.storage.index import SQLiteIndex
from enterprise_rag.embeddings.provider import BedrockEmbeddingProvider
from enterprise_rag.generation.provider import BedrockLLMProvider
from enterprise_rag.generation.service import RAGService, ABSTAIN
from enterprise_rag.retrieval.reranker import BedrockReranker


DATASET = Path("data/evaluation/manual_generation.json")
REPORT_DIR = Path("reports")
REPORT_JSON = REPORT_DIR / "manual_generation.json"
REPORT_CSV = REPORT_DIR / "manual_generation.csv"


def ask_choice(label, choices):
    while True:
        value = input(label).strip().lower()

        if value in choices:
            return value

        print(f"Invalid choice. Use: {', '.join(choices)}")


def load_dataset():
    cases = json.loads(DATASET.read_text(encoding="utf-8"))

    ids = [case["id"] for case in cases]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evaluation case IDs")

    return cases


def load_previous_results():
    if REPORT_JSON.exists():
        return json.loads(
            REPORT_JSON.read_text(encoding="utf-8")
        )

    return []


def save_results(results):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Write JSON using a temporary file.
    temporary = REPORT_JSON.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    temporary.replace(REPORT_JSON)

    columns = [
        "id",
        "question",
        "expected",
        "should_abstain",
        "answer",
        "correctness",
        "citation_support",
        "abstention",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "citations",
        "notes",
    ]

    with REPORT_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=columns, extrasaction="ignore"
        )
        writer.writeheader()

        for result in results:
            row = result.copy()
            row["citations"] = json.dumps(
                row.get("citations", [])
            )
            writer.writerow(row)


def evaluate_answer(case, result, latency_ms):
    print("\n" + "=" * 65)
    print(f"QUESTION: {case['id']}")
    print("=" * 65)

    print(f"\nQUESTION:\n{case['question']}")
    print(f"\nEXPECTED:\n{case['expected']}")
    print(
        f"\nSHOULD ABSTAIN: "
        f"{case['should_abstain']}"
    )

    answer = result["answer"]

    print(f"\nGENERATED ANSWER:\n{answer}")

    print("\nCITATIONS:")
    citations = result.get("citations", [])

    if not citations:
        print("None")

    for citation in citations:
        print(
            f"[{citation['source_id']}] "
            f"{citation['document_id']} "
            f"offsets "
            f"{citation['start_offset']}:"
            f"{citation['end_offset']}"
        )

    print(f"\nLATENCY: {latency_ms:.0f} ms")

    print("\nMANUAL SCORING")
    print("Correctness: 0=wrong, 1=partial, 2=correct")
    correctness = int(
        ask_choice("Correctness (0/1/2): ", {"0", "1", "2"})
    )

    print("Citation: s=supported, p=partial, u=unsupported")
    citation_score = ask_choice(
        "Citation support (s/p/u/n): ",
        {"s", "p", "u", "n"},
    )

    print("Use n only when correct abstention needs no citation.")

    print("Abstention: c=correct, i=incorrect")
    abstention_score = ask_choice(
        "Abstention (c/i): ",
        {"c", "i"},
    )

    notes = input("Notes (optional): ").strip()

    usage = result.get("usage", {})

    return {
        "id": case["id"],
        "question": case["question"],
        "expected": case["expected"],
        "should_abstain": case["should_abstain"],
        "answer": answer,
        "correctness": correctness,
        "citation_support": citation_score,
        "abstention": abstention_score,
        "latency_ms": round(latency_ms, 2),
        "input_tokens": usage.get("inputTokens", 0),
        "output_tokens": usage.get("outputTokens", 0),
        "citations": citations,
        "notes": notes,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def print_summary(results):
    if not results:
        print("No completed evaluations.")
        return

    total = len(results)

    correctness = statistics.mean(
        r["correctness"] for r in results
    )

    correct_abstentions = sum(
        r["abstention"] == "c" for r in results
    )

    citation_applicable = [
        r for r in results
        if r["citation_support"] != "n"
    ]

    supported = sum(
        r["citation_support"] == "s"
        for r in citation_applicable
    )

    avg_latency = statistics.mean(
        r["latency_ms"] for r in results
    )

    print("\n" + "=" * 55)
    print("MANUAL EVALUATION SUMMARY")
    print("=" * 55)

    print(f"Evaluated questions : {total}")
    print(f"Mean correctness    : {correctness:.2f} / 2")
    print(
        f"Correctness %       : "
        f"{correctness / 2 * 100:.1f}%"
    )
    print(
        f"Abstention accuracy : "
        f"{correct_abstentions / total * 100:.1f}%"
    )

    if citation_applicable:
        print(
            f"Full citation support: "
            f"{supported / len(citation_applicable) * 100:.1f}%"
        )

    print(
        f"Average latency     : "
        f"{avg_latency / 1000:.2f}s"
    )

    print(f"\nJSON report : {REPORT_JSON}")
    print(f"CSV report  : {REPORT_CSV}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive manual RAG evaluation"
    )

    parser.add_argument("--tenant", default="acme")
    parser.add_argument("--groups", nargs="+", default=["sre"])
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Discard previous manual evaluation results",
    )

    args = parser.parse_args()

    if args.reset:
        previous = []
    else:
        previous = load_previous_results()

    model_id = os.getenv("RAG_LLM_MODEL_ID")

    if not model_id:
        parser.error("RAG_LLM_MODEL_ID must be set")

    cases = [
        c for c in load_dataset()
        if c.get("tenant_id", args.tenant) == args.tenant
    ]

    if args.limit > 0:
        cases = cases[:args.limit]

    # Only evaluate cases not already completed.
    completed = {r["id"] for r in previous}
    pending = [c for c in cases if c["id"] not in completed]

    if not pending:
        print("All selected questions already evaluated.")
        print_summary(previous)
        return

    store = SQLiteIndex()

    try:
        embedder = BedrockEmbeddingProvider(
            region=args.region
        )

        llm = BedrockLLMProvider(
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
            embedding_provider=embedder,
            llm_provider=llm,
            reranker=reranker,
        )

        print("\nRAG MANUAL EVALUATION")
        print(f"Pending questions : {len(pending)}")
        print(f"Tenant            : {args.tenant}")
        print(f"Groups            : {args.groups}")
        print(f"Reranking         : {args.rerank}")
        print("Press Ctrl+C to stop. Completed scores are saved.")

        for index, case in enumerate(pending, 1):
            print(
                f"\nRunning question {index}/{len(pending)}: "
                f"{case['id']}"
            )

            start = perf_counter()

            try:
                result = rag.ask(
                    question=case["question"],
                    tenant_id=args.tenant,
                    allowed_groups=args.groups,
                )
            except Exception as exc:
                print(
                    f"Question {case['id']} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                print("Skipping this question.")
                continue

            elapsed_ms = (perf_counter() - start) * 1000

            evaluation = evaluate_answer(
                case, result, elapsed_ms
            )

            previous.append(evaluation)
            save_results(previous)

            print(f"\nSaved evaluation: {case['id']}")

    except (KeyboardInterrupt, EOFError):
        print("\nEvaluation interrupted. Saved results retained.")

    finally:
        store.conn.close()

    print_summary(previous)


if __name__ == "__main__":
    main()
