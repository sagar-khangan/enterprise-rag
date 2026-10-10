import re
import uuid

from time import perf_counter

from enterprise_rag.retrieval.hybrid import hybrid_search
from enterprise_rag.generation.prompt import (
    SYSTEM_PROMPT,
    build_prompt,
)
from enterprise_rag.observability import setup_logging


ABSTAIN = "I don't have enough information in the documents."

logger = setup_logging()


def log_stage(request_id, stage, duration_ms, **fields):
    logger.info(
        "rag_stage_completed",
        extra={
            "request_id": request_id,
            "stage": stage,
            "duration_ms": round(duration_ms, 2),
            **fields,
        },
    )


class RAGService:

    def __init__(
        self,
        store,
        embedding_provider,
        llm_provider,
        reranker=None,
    ):
        self.store = store
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.reranker = reranker

    def ask(
        self,
        question: str,
        tenant_id: str,
        allowed_groups: list[str],
    ) -> dict:

        if not question.strip():
            raise ValueError("Question cannot be empty")

        if len(question) > 2000:
            raise ValueError("Question is too long")

        if not tenant_id.strip():
            raise ValueError("Tenant ID is required")

        if not allowed_groups:
            raise ValueError("At least one group is required")

        request_id = uuid.uuid4().hex[:12]
        total_start = perf_counter()

        metrics = {
            "retrieval_ms": 0.0,
            "generation_ms": 0.0,
            "total_ms": 0.0,
            "candidate_count": 0,
        }

        try:
            # ------------------------------------
            # Stage 1: Hybrid Retrieval + Reranking
            # ------------------------------------

            start = perf_counter()

            results = hybrid_search(
                store=self.store,
                provider=self.embedding_provider,
                question=question,
                tenant_id=tenant_id,
                allowed_groups=allowed_groups,
                top_k=5,
                candidate_k=20,
                reranker=self.reranker,
            )

            metrics["retrieval_ms"] = round(
                (perf_counter() - start) * 1000,
                2,
            )

            metrics["candidate_count"] = len(results)

            log_stage(
                request_id,
                "retrieval",
                metrics["retrieval_ms"],
                candidate_count=len(results),
                status="success",
            )

            # ------------------------------------
            # No retrieved evidence
            # ------------------------------------

            if not results:
                metrics["total_ms"] = round(
                    (perf_counter() - total_start) * 1000,
                    2,
                )

                return {
                    "answer": ABSTAIN,
                    "citations": [],
                    "usage": {},
                    "request_id": request_id,
                    "metrics": metrics,
                }

            # ------------------------------------
            # Stage 2: Build grounded prompt
            # ------------------------------------

            prompt, sources = build_prompt(
                question,
                results,
            )

            # ------------------------------------
            # Stage 3: LLM Generation
            # ------------------------------------

            start = perf_counter()

            generated = self.llm_provider.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
            )

            metrics["generation_ms"] = round(
                (perf_counter() - start) * 1000,
                2,
            )

            usage = generated.get("usage") or {}

            log_stage(
                request_id,
                "generation",
                metrics["generation_ms"],
                input_tokens=usage.get("inputTokens", 0),
                output_tokens=usage.get("outputTokens", 0),
                status="success",
            )

            answer = generated["text"].strip()

            # ------------------------------------
            # Stage 4: Validate response
            # ------------------------------------

            if generated.get("stop_reason") != "end_turn":
                answer = ABSTAIN
                citations = []

            elif answer == ABSTAIN:
                citations = []

            else:
                cited_ids = list(dict.fromkeys(
                    re.findall(r"\[S\d+\]", answer)
                ))

                cited_ids = [
                    marker[1:-1]
                    for marker in cited_ids
                ]

                if (
                    not cited_ids
                    or any(
                        source_id not in sources
                        for source_id in cited_ids
                    )
                ):
                    answer = ABSTAIN
                    citations = []

                else:
                    citations = [
                        {
                            "source_id": source_id,
                            **sources[source_id],
                        }
                        for source_id in cited_ids
                    ]

            # ------------------------------------
            # Stage 5: Final metrics
            # ------------------------------------

            metrics["total_ms"] = round(
                (perf_counter() - total_start) * 1000,
                2,
            )

            logger.info(
                "rag_request_completed",
                extra={
                    "request_id": request_id,
                    "stage": "complete",
                    "duration_ms": metrics["total_ms"],
                    "candidate_count": len(results),
                    "citation_count": len(citations),
                    "status": "success",
                },
            )

            return {
                "answer": answer,
                "citations": citations,
                "usage": usage,
                "request_id": request_id,
                "metrics": metrics,
            }

        except Exception as exc:
            logger.error(
                "rag_request_failed",
                extra={
                    "request_id": request_id,
                    "stage": "request",
                    "duration_ms": round(
                        (perf_counter() - total_start) * 1000,
                        2,
                    ),
                    "error_type": type(exc).__name__,
                    "status": "failed",
                },
            )
            raise