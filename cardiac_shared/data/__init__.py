"""
Data management module for cardiac imaging projects

Provides:
- IntermediateResultsRegistry: Cross-project data discovery and sharing
- PathConverter: Windows/WSL path conversion
"""

from .registry import (
    IntermediateResultsRegistry,
    RegistryEntry,
    get_registry,
)

__all__ = [
    'IntermediateResultsRegistry',
    'RegistryEntry',
    'get_registry',
]
