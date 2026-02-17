"""GroupChat module"""
try:
    from .group_chat import GroupChat
except ImportError:
    from group_chat import GroupChat

__all__ = ['GroupChat']