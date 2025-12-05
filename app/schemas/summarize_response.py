from pydantic import BaseModel

class SummarizeResponse(BaseModel):
    """
    Response model for the summarization endpoint.
    """
    summary: str
    original_word_count: int
    summary_word_count: int
    compression_ratio: float
