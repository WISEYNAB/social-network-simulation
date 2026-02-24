"""Clustering module for coalition detection"""
try:
    from .coalition_detector import CoalitionDetector
except ImportError:
    from coalition_detector import CoalitionDetector

__all__ = ['CoalitionDetector']