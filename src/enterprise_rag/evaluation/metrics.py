
import math


def unique_document_ids(results):
    """Deduplicate chunk results by document ID, preserving rank."""
    seen = set()
    ids = []

    for result in results:
        doc_id = result["chunk"]["document_id"]

        if doc_id not in seen:
            seen.add(doc_id)
            ids.append(doc_id)

    return ids


def recall_at_k(ranked_ids, relevant_docs, k=5):
    relevant = {
        doc_id for doc_id, grade in relevant_docs.items()
        if grade > 0
    }

    if not relevant:
        return None

    retrieved = set(ranked_ids[:k])
    return len(relevant & retrieved) / len(relevant)


def reciprocal_rank_at_k(ranked_ids, relevant_docs, k=5):
    relevant = {
        doc_id for doc_id, grade in relevant_docs.items()
        if grade > 0
    }

    if not relevant:
        return None

    for position, doc_id in enumerate(ranked_ids[:k], 1):
        if doc_id in relevant:
            return 1.0 / position

    return 0.0


def ndcg_at_k(ranked_ids, relevant_docs, k=5):
    if not relevant_docs or not any(relevant_docs.values()):
        return None

    def dcg(grades):
        return sum(
            (2 ** grade - 1) / math.log2(index + 2)
            for index, grade in enumerate(grades)
        )

    actual_grades = [
        relevant_docs.get(doc_id, 0)
        for doc_id in ranked_ids[:k]
    ]

    ideal_grades = sorted(
        relevant_docs.values(), reverse=True
    )[:k]

    ideal = dcg(ideal_grades)

    return dcg(actual_grades) / ideal if ideal else None
