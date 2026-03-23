"""Analysis module for computing metrics"""
try:
    from .metrics import MetricsCalculator
    from .reporter import ResearchReporter
except ImportError:
    from metrics import MetricsCalculator
    from reporter import ResearchReporter

__all__ = ['MetricsCalculator', 'ResearchReporter']