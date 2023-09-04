from abc import ABC, abstractmethod


class BaseTaskHelper(ABC):
    """
    Base helper class to handle basic steps of the pipeline (e.g., tokenization, inference).
    """

    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def _check_target(self, target, **kwargs):
        """Validate the specific target requested for the explanation"""
        pass

    @abstractmethod
    def _check_sample(self, input, **kwargs):
        """Validate the specific input requested for the explanation"""
        pass

    def _prepare_sample(self, sample, **kwargs):
        """Format the input before the explanation"""
        return sample

    def format_target(self, target, **kwargs):
        """Format the target variable

        In all our current explainers, 'target' must be a positional integer for the
        logits matrix. Default: leave target unchanged.
        """
        return target

    def _postprocess_logits(self, logits, **kwargs):
        """Process the logits before computing the explanation"""
        return logits
