"""Custom exceptions for the API."""
from typing import Optional


class OmoAIException(Exception):
    """Base exception for OmoAI API."""
    
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class AudioProcessingException(OmoAIException):
    """Exception raised when audio processing fails."""
    
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message, status_code)


class ModelInitializationException(OmoAIException):
    """Exception raised when model initialization fails."""
    
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message, status_code)


class FileNotFoundException(OmoAIException):
    """Exception raised when a required file is not found."""
    
    def __init__(self, message: str, status_code: int = 404):
        super().__init__(message, status_code)


class ValidationError(OmoAIException):
    """Exception raised when input validation fails."""
    
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message, status_code)