"""
Custom exception module for centralized error handling.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Optional


class CustomException(Exception):
    """
    Production-grade custom exception.

    Captures:
    - Original exception
    - Custom message
    - File name
    - Function name
    - Line number
    - Full traceback
    - Optional debugging context
    """

    def __init__(
        self,
        error: Exception,
        message: str,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)

        self.original_error = error
        self.message = message
        self.context = context or {}

        _, _, exc_tb = sys.exc_info()

        if exc_tb:
            self.filename = Path(exc_tb.tb_frame.f_code.co_filename).name
            self.function_name = exc_tb.tb_frame.f_code.co_name
            self.line_number = exc_tb.tb_lineno
        else:
            self.filename = "Unknown"
            self.function_name = "Unknown"
            self.line_number = -1

        self.traceback = traceback.format_exc()

    def __str__(self) -> str:
        return (
            "\n"
            "========== APPLICATION ERROR ==========\n"
            f"Message          : {self.message}\n"
            f"Exception Type   : {type(self.original_error).__name__}\n"
            f"Original Error   : {self.original_error}\n"
            f"File             : {self.filename}\n"
            f"Function         : {self.function_name}\n"
            f"Line             : {self.line_number}\n"
            f"Context          : {self.context}\n"
            "---------------------------------------\n"
            f"{self.traceback}"
            "=======================================\n"
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Structured representation useful for logging systems.
        """
        return {
            "message": self.message,
            "exception_type": type(self.original_error).__name__,
            "original_error": str(self.original_error),
            "file": self.filename,
            "function": self.function_name,
            "line": self.line_number,
            "context": self.context,
            "traceback": self.traceback,
        }