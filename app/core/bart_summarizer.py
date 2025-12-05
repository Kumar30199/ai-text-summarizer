import time
import torch
from transformers import pipeline
from app.core.summarizer_base import SummarizerBase
from app.config import config

# Log when model is created (for debugging)
print("🔁 Initializing summarization pipeline...")

# Auto-detect device: 0 for GPU (CUDA), -1 for CPU
_device = 0 if torch.cuda.is_available() else -1
_device_name = "GPU" if _device == 0 else "CPU"

print(f"Loading model: {config.MODEL_NAME} on {_device_name}...")

# Initialize the pipeline at module level (Singleton)
_summarization_pipeline = pipeline(
    "summarization",
    model=config.MODEL_NAME,
    tokenizer=config.MODEL_NAME,
    framework="pt",
    device=_device,
)

print("✅ Summarization pipeline initialized.")

class BARTSummarizer(SummarizerBase):
    """
    Lightweight wrapper that reuses a single global pipeline instance.
    """

    def summarize(self, text: str, max_length: int, min_length: int) -> str:
        """
        Summarize text using the global pipeline instance.
        """
        start = time.time()
        
        # The pipeline returns a list of dicts: [{'summary_text': '...'}]
        result = _summarization_pipeline(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )
        
        elapsed = time.time() - start
        print(f"⏱  summarize() took {elapsed:.2f}s")
        
        return result[0]['summary_text']
