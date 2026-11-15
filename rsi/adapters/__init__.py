"""Dataset adapters beyond the official packages.

``private``   evaluate a private/hidden-style JSONL file without training on it
``synthetic`` generate deterministic drift probes from the official public splits
``schema``    reuse the official strict JSONL reader and summarise its schema
"""
from .private import evaluate_private, evaluate_private_file
from .schema import SchemaError, describe_dataset, load_private_dataset
from .synthetic import synthesize, write_synthetic

__all__ = ["evaluate_private", "evaluate_private_file", "SchemaError", "describe_dataset", "load_private_dataset",
           "synthesize", "write_synthetic"]
