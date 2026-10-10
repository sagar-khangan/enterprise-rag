
import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter

from enterprise_rag.storage.index import SQLiteIndex
from enterprise_rag.embeddings.provider import BedrockEmbeddingProvider
from enterprise_rag.retrieval.hybrid import hybrid_search
from enterprise_rag.retrieval.reranker import BedrockReranker
from enterprise_rag.evaluation.metrics import (
    unique_document_ids,
    recall_at_k,
    reciprocal_rank_at_k,
    ndcg_at_k,
)


def evaluate_case(store, provider, reranker, case):
    start = perf_counter()

    results = hybrid_search(
        store=store,
        provider=provider,
        question=case["question"],
        tenant_id=case["tenant_id"],
        allowed_groups=case["allowed_groups"],
        top_k=20,
        candidate_k=40,
        reranker=reranker,
    )
    retrieval_ms = (perf_counter() - start) * 1000

    log_stage(
        request_id,
        "retrieval",
        retrieval_ms,
        candidate_count=len(results),
    )

    latency_ms = (perf_counter() - start) * 1000

    ranked_ids = unique_document_ids(results)
    relevant = case["relevant_documents"]

    return {
        "id": case["id"],
        "question": case["question"],
        "ranked_documents": ranked_ids[:5],
        "recall_at_5": recall_at_k(
            ranked_ids, relevant
        ),
        "mrr_at_5": reciprocal_rank_at_k(
            ranked_ids, relevant
        ),
        "ndcg_at_5": ndcg_at_k(
            ranked_ids, relevant
        ),
        "latency_ms": round(latency_ms, 2),
    }


def average(rows, field):
    values = [
        row[field] for row in rows
        if row[field] is not None
    ]
    return statistics.mean(values) if values else None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="data/evaluation/retrieval.json",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
    )

    args = parser.parse_args()

    cases = json.loads(
        Path(args.dataset).read_text(encoding="utf-8")
    )

    store = SQLiteIndex()
    provider = BedrockEmbeddingProvider(region=args.region)

    reranker = (
        BedrockReranker(region=args.region)
        if args.rerank else None
    )

    results = []

    try:
        for case in cases:
            row = evaluate_case(
                store, provider, reranker, case
            )
            results.append(row)

            print(
                f"{case['id']}: "
                f"Recall@5={row['recall_at_5']} "
                f"MRR@5={row['mrr_at_5']}"
            )
    finally:
        store.conn.close()

    summary = {
        "mode": "rerank" if args.rerank else "baseline",
        "total_queries": len(results),
        "mean_recall_at_5": average(
            results, "recall_at_5"
        ),
        "mean_mrr_at_5": average(
            results, "mrr_at_5"
        ),
        "mean_ndcg_at_5": average(
            results, "ndcg_at_5"
        ),
        "mean_latency_ms": average(
            results, "latency_ms"
        ),
    }

    output = {
        "summary": summary,
        "queries": results,
    }

    Path("reports").mkdir(exist_ok=True)

    name = (
        "rerank.json" if args.rerank
        else "baseline.json"
    )

    Path("reports", name).write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("\nEVALUATION SUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
