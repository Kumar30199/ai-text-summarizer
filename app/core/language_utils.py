from langdetect import detect, LangDetectException

def detect_language(text: str) -> str:
    """
    Detects the language of the given text.
    Returns 'en' for English, 'hi' for Hindi, etc.
    Defaults to 'en' if detection fails.
    """
    try:
        # langdetect is fast and reliable for reasonable length text
        lang = detect(text)
        return lang
    except LangDetectException:
        # Fallback to English if detection fails (e.g. empty or weird text)
        return "en"
