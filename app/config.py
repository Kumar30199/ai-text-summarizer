import torch

class Config:
    """
    Application configuration.
    """
    # Default English Model (DistilBART)
    ENGLISH_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
    
    # Default Multilingual Model (mT5)
    MULTILINGUAL_MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"

    # Map of user-selectable models
    AVAILABLE_MODELS = {
        "distilbart": "sshleifer/distilbart-cnn-12-6",
        "t5-small": "t5-small",
        "pegasus": "google/pegasus-xsum",
        "bart-large": "facebook/bart-large-cnn"
    }
    
    # Models categorized by speed/performance
    FAST_MODELS = ["distilbart", "t5-small"]
    SLOW_MODELS = ["pegasus", "bart-large"]

    # Thresholds for Auto-Detect Logic
    THRESHOLD_SHORT = 200
    THRESHOLD_LONG = 500

    # Min/Max length mappings for each preset
    LENGTH_MAP = {
        "short": {"min_length": 20, "max_length": 60},
        "medium": {"min_length": 40, "max_length": 120},
        "long": {"min_length": 60, "max_length": 180}
    }

    # Advanced Model Safety
    # If True, allows loading Pegasus/BART-Large on CPU.
    # If False, they will be blocked or routed to a fast model.
    ENABLE_ADVANCED_MODELS_ON_CPU = False

config = Config()
