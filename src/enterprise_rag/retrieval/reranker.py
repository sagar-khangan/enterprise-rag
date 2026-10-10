from botocore.exceptions import BotoCoreError, ClientError

from enterprise_rag.aws_config import create_aws_client
from enterprise_rag.errors import ModelUnavailableError


class BedrockReranker:

    def __init__(
        self,
        region: str = "us-east-1",
        model_id: str = "cohere.rerank-v3-5:0",
    ):
        self.model_id = model_id
        self.region = region

        self.client = create_aws_client(
            "bedrock-agent-runtime",
            region,
        )

        self.model_arn = (
            f"arn:aws:bedrock:{region}::"
            f"foundation-model/{model_id}"
        )

    def rerank(
        self,
        question: str,
        candidates: list,
        top_k: int = 5,
    ) -> list:

        if not candidates:
            return []

        if not question.strip():
            raise ValueError("Question cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        sources = [
            {
                "type": "INLINE",
                "inlineDocumentSource": {
                    "type": "TEXT",
                    "textDocument": {
                        "text": item["chunk"]["text"]
                    },
                },
            }
            for item in candidates
        ]

        try:
            response = self.client.rerank(
                queries=[
                    {
                        "type": "TEXT",
                        "textQuery": {
                            "text": question
                        },
                    }
                ],
                sources=sources,
                rerankingConfiguration={
                    "type": "BEDROCK_RERANKING_MODEL",
                    "bedrockRerankingConfiguration": {
                        "modelConfiguration": {
                            "modelArn": self.model_arn,
                        },
                        "numberOfResults": min(
                            top_k,
                            len(candidates),
                        ),
                    },
                },
            )

        except (ClientError, BotoCoreError) as exc:
            raise ModelUnavailableError(
                "Bedrock reranking request failed"
            ) from exc

        reranked = []
        seen = set()

        for result in response["results"]:
            index = result["index"]

            if not isinstance(index, int):
                raise ValueError(
                    "Reranker returned invalid index type"
                )

            if index < 0 or index >= len(candidates):
                raise ValueError(
                    "Reranker returned out-of-range index"
                )

            if index in seen:
                raise ValueError(
                    "Reranker returned duplicate index"
                )

            seen.add(index)

            item = candidates[index].copy()

            item["rrf_score"] = item["score"]
            item["rerank_score"] = result[
                "relevanceScore"
            ]

            reranked.append(item)

        reranked.sort(
            key=lambda item: -item["rerank_score"]
        )

        return reranked[:top_k]