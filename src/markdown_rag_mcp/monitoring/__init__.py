"""
Monitoring package for file system change detection.

This package provides components for monitoring file system changes
and automatically updating the document index when files are added,
modified, or deleted.
"""

from .file_watcher import FileChangeEvent, MarkdownFileWatcher
from .monitoring_coordinator import MonitoringCoordinator

__all__ = [
    "FileChangeEvent",
    "MarkdownFileWatcher",
    "MonitoringCoordinator",
]
