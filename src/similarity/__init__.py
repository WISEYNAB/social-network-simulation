"""Similarity module"""
try:
    from .similarity_matrix import SimilarityMatrix
except ImportError:
    from similarity_matrix import SimilarityMatrix

__all__ = ['SimilarityMatrix']