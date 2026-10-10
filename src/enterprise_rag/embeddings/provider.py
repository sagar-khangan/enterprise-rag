import hashlib
import json

import numpy as np

from botocore.exceptions import BotoCoreError, ClientError

from enterprise_rag.aws_config import create_aws_client
from enterprise_rag.errors import ModelUnavailableError


class FakeEmbeddingProvider:
    """
    Deterministic embeddings for testing.

    These vectors are NOT semantically meaningful.
    """

    model_id = "fake-v1"
    dimensions = 256

    def embed(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("Text cannot be empty")

        digest = hashlib.shake_256(
            text.encode("utf-8")
        ).digest(self.dimensions * 4)

        values = np.frombuffer(
            digest,
            dtype=np.uint32,
        ).astype(np.float32)

        values = values / np.float32(2**32 - 1)
        values = values * 2 - 1

        norm = np.linalg.norm(values)

        if norm == 0:
            raise ValueError("Zero embedding norm")

        return (values / norm).astype(np.float32)


class BedrockEmbeddingProvider:

    model_id = "amazon.titan-embed-text-v2:0"
    dimensions = 256

    def __init__(self, region: str = "us-east-1"):
        self.region = region

        self.client = create_aws_client(
            "bedrock-runtime",
            region,
        )

    def embed(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "inputText": text,
                    "dimensions": self.dimensions,
                    "normalize": True,
                }),
                contentType="application/json",
                accept="application/json",
            )

            payload = json.loads(
                response["body"].read()
            )

        except (ClientError, BotoCoreError) as exc:
            raise ModelUnavailableError(
                "Bedrock embedding request failed"
            ) from exc

        vector = np.asarray(
            payload["embedding"],
            dtype=np.float32,
        )

        if vector.shape != (self.dimensions,):
            raise ValueError(
                "Unexpected embedding dimensions"
            )

        if not np.isfinite(vector).all():
            raise ValueError(
                "Embedding contains non-finite values"
            )

        return vector