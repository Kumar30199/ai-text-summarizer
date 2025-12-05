from app.schemas.summarize_request import SummarizeRequest
from app.schemas.summarize_response import SummarizeResponse
from app.core.smart_summarizer import SmartSummarizer
from app.config import config

# Instantiate smart summarizer (Singleton)
_summarizer = SmartSummarizer()

# Safety limit
MAX_INPUT_CHARS = 4000 

async def summarize_text(request: SummarizeRequest) -> SummarizeResponse:
    """
    Service function to handle text summarization logic.
    Now Async to support timeouts.
    """
    # Get length constraints
    length_settings = config.LENGTH_MAP.get(request.length, config.LENGTH_MAP["medium"])
    min_length = length_settings["min_length"]
    max_length = length_settings["max_length"]

    # Truncate Input
    text_to_process = request.text
    if len(text_to_process) > MAX_INPUT_CHARS:
        text_to_process = text_to_process[:MAX_INPUT_CHARS]
        last_period = text_to_process.rfind('.')
        if last_period > 0:
            text_to_process = text_to_process[:last_period+1]

    # Perform summarization (Async)
    summary_text = await _summarizer.summarize_async(
        text_to_process, 
        max_length=max_length, 
        min_length=min_length,
        model_key=request.model
    )

    # Calculate statistics
    original_word_count = len(request.text.split())
    summary_word_count = len(summary_text.split())
    compression_ratio = summary_word_count / original_word_count if original_word_count > 0 else 0.0

    return SummarizeResponse(
        summary=summary_text,
        original_word_count=original_word_count,
        summary_word_count=summary_word_count,
        compression_ratio=round(compression_ratio, 2)
    )
