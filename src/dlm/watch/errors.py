"""Typed errors for `dlm train --watch`."""

from __future__ import annotations


class WatchError(Exception):
    """Base class for watch-mode errors."""


class WatchSetupError(WatchError):
    """Raised when the watcher can't start (missing path, permission, etc.)."""
