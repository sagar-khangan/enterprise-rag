import json
import logging
import sys

from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):

    def format(self, record):
        payload = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        fields = (
            "request_id",
            "stage",
            "duration_ms",
            "candidate_count",
            "citation_count",
            "input_tokens",
            "output_tokens",
            "error_type",
            "status",
        )

        for field in fields:
            value = getattr(record, field, None)

            if value is not None:
                payload[field] = value

        return json.dumps(payload, default=str)


def setup_logging():
    logger = logging.getLogger("enterprise_rag")

    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    return logger