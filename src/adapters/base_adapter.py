# Re-export shim — BaseAdapter now lives in src/core/base_adapter.py.
# This file is kept for backward compatibility so any existing import of
# src.adapters.base_adapter continues to work without changes.
from src.core.base_adapter import BaseAdapter  # noqa: F401

__all__ = ["BaseAdapter"]
