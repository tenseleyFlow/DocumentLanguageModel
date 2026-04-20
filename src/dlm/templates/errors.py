"""Typed errors for the template gallery."""

from __future__ import annotations


class TemplateError(Exception):
    """Base class for template gallery errors."""


class TemplateNotFoundError(TemplateError):
    """A template name was requested that isn't in any registry."""


class TemplateMetaError(TemplateError):
    """A template's `meta.yaml` failed to load or validate."""


class TemplateApplyError(TemplateError):
    """`apply_template` refused — e.g. target file already exists."""
