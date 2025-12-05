import time
import logging
import torch
import asyncio
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline
from app.config import config
from app.core.summarizer_base import SummarizerBase
from app.core.language_utils import detect_language

logger = logging.getLogger(__name__)

class SmartSummarizer(SummarizerBase):
    """
    Summarizer implementation that handles model routing and timeouts.
    Uses eager loading for fast models and a hybrid approach for long text.
    """
    _instance = None
    _pipelines = {}
    _executor = ThreadPoolExecutor(max_workers=1) # Dedicated thread for model inference

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SmartSummarizer, cls).__new__(cls)
            
            # Auto-detect device
            cls._device = 0 if torch.cuda.is_available() else -1
            cls._device_name = "GPU" if cls._device == 0 else "CPU"
            logger.info(f"SmartSummarizer initialized. Device: {cls._device_name}")
            
            # Load fast models immediately to avoid delay on first request
            cls._instance._eager_load_fast_models()
            
        return cls._instance

    def _eager_load_fast_models(self):
        """
        Loads fast models (DistilBART, T5) during startup.
        """
        logger.info("Eager loading fast models...")
        try:
            self._get_pipeline("distilbart")
            self._get_pipeline("t5-small")
            logger.info("Fast models loaded.")
        except Exception as e:
            logger.error(f"Failed to eager load models: {e}")

    def _get_pipeline(self, model_key: str):
        """
        Loads the requested model pipeline if it's not already in memory.
        Checks if advanced models are allowed on CPU.
        """
        # Safety Check for Advanced Models on CPU
        if model_key in config.SLOW_MODELS and self._device == -1:
            if not config.ENABLE_ADVANCED_MODELS_ON_CPU:
                logger.warning(f"Advanced model '{model_key}' disabled on CPU. Routing to DistilBART.")
                return self._get_pipeline("distilbart")

        model_name = config.AVAILABLE_MODELS.get(model_key)
        
        if not model_name:
            logger.warning(f"Model key '{model_key}' not found. Falling back to default.")
            model_name = config.ENGLISH_MODEL_NAME

        if model_name not in self._pipelines:
            logger.info(f"Loading Model ({model_name})...")
            start = time.time()
            try:
                # Standard pipeline initialization
                self._pipelines[model_name] = pipeline(
                    "summarization",
                    model=model_name,
                    tokenizer=model_name,
                    framework="pt",
                    device=self._device
                )
                logger.info(f"Model ({model_name}) loaded in {time.time() - start:.2f}s")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                # Fallback to DistilBART if specific model fails
                if model_key != "distilbart":
                     logger.info("Fallback to DistilBART due to load failure.")
                     return self._get_pipeline("distilbart")
                raise e
        
        return self._pipelines[model_name]

    def _extractive_summarize(self, text: str, max_sentences: int = 5) -> str:
        """
        Basic frequency-based extractive summarization.
        Helps reduce very long text before passing it to the model.
        """
        try:
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
            if len(sentences) <= max_sentences:
                return text
                
            words = re.findall(r'\w+', text.lower())
            word_freq = Counter(words)
            
            # Score sentences based on word frequency
            sentence_scores = {}
            for sent in sentences:
                for word in re.findall(r'\w+', sent.lower()):
                    if word in word_freq:
                        if len(sent.split()) < 30: # Skip very long sentences
                            if sent not in sentence_scores:
                                sentence_scores[sent] = word_freq[word]
                            else:
                                sentence_scores[sent] += word_freq[word]
                                
            # Select top N sentences
            import heapq
            summary_sentences = heapq.nlargest(max_sentences, sentence_scores, key=sentence_scores.get)
            return ' '.join(summary_sentences)
        except Exception as e:
            logger.error(f"Extractive summarization failed: {e}")
            return text[:2000] # Fallback to simple truncation

    def _run_inference(self, pipe, text, max_length, min_length):
        """
        Runs inference in a separate thread.
        """
        return pipe(
            text,
            max_length=max_length,
            min_length=min_length,
            truncation=True,
            do_sample=False
        )[0]['summary_text']

    async def summarize_async(self, text: str, max_length: int, min_length: int, model_key: str = "auto") -> str:
        """
        Async summarization with timeout protection and auto-selection logic.
        """
        start = time.time()
        word_count = len(text.split())
        lang = detect_language(text)
        
        logger.info(f"Request: {word_count} words. Lang: {lang}. Model: {model_key}")

        # 1. Auto-Detect Logic
        target_model_key = model_key
        if model_key == "auto":
            if lang != "en":
                # Prefer T5 for non-English text
                target_model_key = "t5-small" 
            elif word_count < config.THRESHOLD_SHORT:
                target_model_key = "t5-small"
            else:
                target_model_key = "distilbart"
        
        # 2. Handle Long Text (Hybrid Approach)
        processed_text = text
        if word_count > config.THRESHOLD_LONG and lang == "en":
            logger.info("Text > 500 words. Running extractive summarization first.")
            processed_text = self._extractive_summarize(text, max_sentences=8)
            logger.info(f"Reduced to {len(processed_text.split())} words.")

        # 3. Select Pipeline
        try:
            pipe = self._get_pipeline(target_model_key)
        except Exception as e:
            logger.error(f"Failed to get pipeline for {target_model_key}: {e}")
            # Ultimate fallback
            target_model_key = "distilbart"
            pipe = self._get_pipeline("distilbart")

        # 4. Run with Timeout
        try:
            # 6-second timeout for model inference
            loop = asyncio.get_event_loop()
            summary = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor, 
                    self._run_inference, 
                    pipe, processed_text, max_length, min_length
                ),
                timeout=6.0
            )
            
            elapsed = time.time() - start
            logger.info(f"Summarization finished in {elapsed:.2f}s")
            return summary

        except asyncio.TimeoutError:
            logger.warning("Model inference timed out (>6s).")
            
            # Fallback strategy
            if target_model_key != "distilbart":
                logger.info("Falling back to DistilBART (Fastest)...")
                try:
                    fallback_pipe = self._get_pipeline("distilbart")
                    summary = await asyncio.wait_for(
                        loop.run_in_executor(
                            self._executor,
                            self._run_inference,
                            fallback_pipe, processed_text, max_length, min_length
                        ),
                        timeout=3.0 # Give fallback 3 seconds
                    )
                    return summary
                except Exception:
                    pass
            
            # Ultimate fallback: Return extractive summary
            logger.warning("Fallback failed. Returning extractive summary.")
            return self._extractive_summarize(text, max_sentences=3)

        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            # Return a friendly error message or extractive summary instead of crashing
            return f"Error generating summary: {str(e)}. Here is a brief extract: " + self._extractive_summarize(text, max_sentences=2)

    def summarize(self, text: str, max_length: int, min_length: int, model_key: str = "auto") -> str:
        # Wrapper to run async code synchronously if needed
        return asyncio.run(self.summarize_async(text, max_length, min_length, model_key))
