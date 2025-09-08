"""
Custom exceptions for the OMOAI pipeline.

This module defines custom exception classes for different types of errors
that can occur in the OMOAI pipeline.
"""

class OMOPipelineError(RuntimeError):
    """Base exception for OMOAI pipeline errors."""
    pass


class OMOAudioError(OMOPipelineError):
    """Exception raised for audio processing errors."""
    pass


class OMOASRError(OMOPipelineError):
    """Exception raised for ASR processing errors."""
    pass


class OMOPostprocessError(OMOPipelineError):
    """Exception raised for postprocessing errors."""
    pass


class OMOConfigError(OMOPipelineError):
    """Exception raised for configuration errors."""
    pass


class OMOModelError(OMOPipelineError):
    """Exception raised for model-related errors."""
    pass


class OMOIOError(OMOPipelineError):
    """Exception raised for input/output errors."""
    pass