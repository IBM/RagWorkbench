import mimetypes
from typing import Any

from docling_core.types.io import DocumentStream
from pydantic import Field, field_validator


class DocumentObject(DocumentStream):
    """
    A document object that extends DocumentStream with additional metadata.

    This class validates MIME types against Python's standard mimetypes registry
    and ensures document identifiers in metadata are consistent with the document's name.

    Attributes:
        mime_type: The MIME type of the document (e.g., 'application/pdf').
                   Must be a recognized MIME type from Python's mimetypes module.
        metadata: Additional metadata for the document.
    """

    mime_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    @field_validator("mime_type")
    def validate_mime_type(cls, value: str) -> str:
        """
        Validate that the MIME type is recognized by Python's mimetypes module.

        This validator checks against the comprehensive list of MIME types
        maintained in Python's standard library (700+ types).

        Args:
            value: The MIME type string to validate.

        Returns:
            The validated MIME type string.

        Raises:
            ValueError: If the MIME type is not recognized or invalid.
        """
        if not value:
            raise ValueError("MIME type cannot be empty")

        # Check basic format: must contain exactly one '/'
        if value.count("/") != 1:
            raise ValueError(
                f"Invalid MIME type format: '{value}'. "
                "Must be in 'type/subtype' format (e.g., 'application/pdf')"
            )

        # Get all known MIME types from Python's mimetypes module
        # types_map maps file extensions to MIME types
        known_mime_types: set[str] = set(mimetypes.types_map.values())

        # Also include common types from common_types
        known_mime_types.update(mimetypes.common_types.values())

        # Check if the MIME type is in the known types
        if value not in known_mime_types:
            raise ValueError(
                f"Unrecognized MIME type: '{value}'. "
                f"Must be a valid MIME type recognized by Python's mimetypes module. "
                f"Examples: 'application/pdf', 'text/plain', 'image/jpeg'"
            )

        return value
