"""Backbone factory for the comparative-study protocol.

The vendored implementations deliberately retain their upstream operations.
"""

from .factory import BACKBONES, build_model, parameter_count

__all__ = ["BACKBONES", "build_model", "parameter_count"]
