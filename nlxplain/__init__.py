"""Top-level package for nlxplain."""

__author__ = """Giuseppe Attanasio"""
__email__ = "giuseppeattanasio6@gmail.com"
__version__ = "0.1.0"

# from .explainer import Explainer
from .evaluation.evaluator import Evalutator

from .explainers.shap import SHAPExplainer
from .explainers.gradient import GradientExplainer, IntegratedGradientExplainer
from .explainers.dummy import DummyExplainer

from .benchmark import Benchmark