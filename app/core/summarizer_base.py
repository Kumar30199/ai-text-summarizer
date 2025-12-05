from abc import ABC, abstractmethod

class SummarizerBase(ABC):
    """
    Abstract base class for all summarizers.
    """

    @abstractmethod
    def summarize(self, text: str, max_length: int, min_length: int) -> str:
        """
        Summarize the given text.

        Args:
            text: The text to summarize.
            max_length: The maximum length of the summary (in tokens).
            min_length: The minimum length of the summary (in tokens).

        Returns:
            The generated summary string.
        """
        pass
