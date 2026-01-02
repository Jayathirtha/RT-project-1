"""Utility functions for the RAG project."""
import os
import random
import logging

logger = logging.getLogger(__name__)


def set_random_seed(seed: int = None):
    """
    Set random seeds for reproducibility.
    
    Sets seeds for Python's random module, NumPy, and PyTorch (if available).
    The seed value can be set via RANDOM_SEED environment variable or passed directly.
    
    Args:
        seed: Random seed value. If None, will try to get from RANDOM_SEED env var, 
              or default to 42.
    """
    if seed is None:
        seed = int(os.getenv("RANDOM_SEED", 42))
    
    # Set Python's random seed
    random.seed(seed)
    logger.debug(f"Set Python random seed to {seed}")
    
    # Set NumPy seed (used by sentence-transformers)
    try:
        import numpy as np
        np.random.seed(seed)
        logger.debug(f"Set NumPy random seed to {seed}")
    except ImportError:
        logger.debug("NumPy not available, skipping NumPy seed setting")
    
    # Set PyTorch seed (used by sentence-transformers and transformers)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # For reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug(f"Set PyTorch random seed to {seed}")
    except ImportError:
        logger.debug("PyTorch not available, skipping PyTorch seed setting")
    
    logger.info(f"Random seeds set to {seed} for reproducibility")

