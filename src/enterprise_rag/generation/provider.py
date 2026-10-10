from botocore.exceptions import BotoCoreError, ClientError

from enterprise_rag.aws_config import create_aws_client
from enterprise_rag.errors import ModelUnavailableError


class BedrockLLMProvider:

    def __init__(
        self,
        model_id: str,
        region: str = "us-east-1",
    ):
        self.model_id = model_id
        self.region = region

        self.client = create_aws_client(
            "bedrock-runtime",
            region,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:

        if not system_prompt.strip():
            raise ValueError("System prompt is empty")

        if not user_prompt.strip():
            raise ValueError("User prompt is empty")

        try:
            response = self.client.converse(
                modelId=self.model_id,
                system=[
                    {"text": system_prompt}
                ],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"text": user_prompt}
                        ],
                    }
                ],
                inferenceConfig={
                    "maxTokens": 700,
                    "temperature": 0,
                },
            )

        except (ClientError, BotoCoreError) as exc:
            raise ModelUnavailableError(
                "Bedrock LLM request failed"
            ) from exc

        content = (
            response
            .get("output", {})
            .get("message", {})
            .get("content", [])
        )

        text = "\n".join(
            block["text"]
            for block in content
            if "text" in block
        ).strip()

        if not text:
            raise ModelUnavailableError(
                "Bedrock returned an empty answer"
            )

        return {
            "text": text,
            "usage": response.get("usage", {}),
            "stop_reason": response.get("stopReason"),
        }