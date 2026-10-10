from enterprise_rag.retrieval.vector_search import (
    vector_search,
)
from enterprise_rag.retrieval.keyword_search import (
    keyword_search,
)


def reciprocal_rank_fusion(
    result_lists,
    k=60,
    top_k=5,
):
    merged = {}

    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            chunk_id = item["chunk_id"]

            if chunk_id not in merged:
                merged[chunk_id] = {
                    "chunk_id": chunk_id,
                    "chunk": item["chunk"],
                    "score": 0.0,
                }

            merged[chunk_id]["score"] += (
                1.0 / (k + rank)
            )

    return sorted(
        merged.values(),
        key=lambda x: (-x["score"], x["chunk_id"]),
    )[:top_k]



def hybrid_search(
    store,
    provider,
    question,
    tenant_id,
    allowed_groups,
    top_k=5,
    candidate_k=20,
    reranker=None,
):
    query_vector = provider.embed(question)

    semantic = vector_search(
        store=store,
        query_vector=query_vector,
        tenant_id=tenant_id,
        allowed_groups=allowed_groups,
        model_id=provider.model_id,
        dimensions=provider.dimensions,
        top_k=candidate_k,
    )

    lexical = keyword_search(
        store=store,
        question=question,
        tenant_id=tenant_id,
        allowed_groups=allowed_groups,
        top_k=candidate_k,
    )

    candidates = reciprocal_rank_fusion(
        [semantic, lexical],
        top_k=candidate_k,
    )

    if reranker is None:
        return candidates[:top_k]

    return reranker.rerank(
        question=question,
        candidates=candidates,
        top_k=top_k,
    )
