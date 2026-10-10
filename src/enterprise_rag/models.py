from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Optional


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_path: str = Field(min_length=1)

    allowed_groups: list[str] = Field(min_length=1)

    version: int = Field(default=1, ge=1)
    content_hash: str = Field(min_length=1)


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)

    text: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)

    allowed_groups: list[str] = Field(min_length=1)

    source_page: Optional[int] = Field(default=None, ge=1)

    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_offsets(self):
        if self.end_offset <= self.start_offset:
            raise ValueError(
                "end_offset must be greater than start_offset"
            )
        return self