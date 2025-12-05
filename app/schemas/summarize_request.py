from pydantic import BaseModel, Field
from typing import Literal, Optional

class SummarizeRequest(BaseModel):
    """
    Request model for the summarization endpoint.
    """
    text: str = Field(..., min_length=10, description="The text to summarize.")
    length: Literal["short", "medium", "long"] = Field(
        "medium", description="Desired summary length preset."
    )
    model: Optional[str] = Field(
        "distilbart", description="The model to use for summarization."
    )
