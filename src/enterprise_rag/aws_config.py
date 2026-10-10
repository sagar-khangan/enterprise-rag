import boto3
from botocore.config import Config


AWS_CONFIG = Config(
    connect_timeout=5,
    read_timeout=60,
    retries={
        "mode": "standard",
        "total_max_attempts": 3,
    },
    max_pool_connections=10,
)


def create_aws_client(service, region="us-east-1"):
    return boto3.client(
        service,
        region_name=region,
        config=AWS_CONFIG,
    )