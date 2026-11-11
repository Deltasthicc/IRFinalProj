"""Evaluation toolkit (Role B): metrics, judgments, pooling, significance testing."""
from .metrics import METRICS
from .qrels import load_qrels, weighted_kappa
from .stats import paired_bootstrap

__all__ = ["METRICS", "load_qrels", "weighted_kappa", "paired_bootstrap"]
