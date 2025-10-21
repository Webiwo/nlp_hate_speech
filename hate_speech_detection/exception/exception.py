import sys
import traceback
import linecache
from hate_speech_detection.logger.logger import logging


def error_message_detail(error):
    """
    Returns a detailed error message with the file name, function name, line number, source code, and exception text.
    """
    # stack[0] where the exception was created (deepest call)
    # stack[-1] where the exception was caught (catch / raise)
    tb_stack = traceback.TracebackException.from_exception(error).stack

    first_tb = None
    for tb in tb_stack:
        if tb.name != "<module>":
            first_tb = tb
            break

    if first_tb is None:
        # If all frames are '<module>', just take the first one
        first_tb = tb_stack[0]

    file_name = first_tb.filename
    func_name = first_tb.name
    line_no = first_tb.lineno

    code_line = linecache.getline(file_name, line_no).strip()

    error_message = (
        f"Error in [{file_name}], "
        f"function [{func_name}], "
        f"line [{line_no}], "
        f"(code: '{code_line}'), "
        f"error message: {str(error)}"
    )

    return error_message


class CustomException(Exception):
    """Base application exception (no logging)"""

    def __init__(self, error):
        self.error_message = (
            error if isinstance(error, str) else error_message_detail(error)
        )
        super().__init__(self.error_message)

    def __str__(self):
        return self.error_message


# Domain exceptions
class GCloudSyncError(CustomException):
    """Error syncing data with GCloud"""


class DataIngestionError(CustomException):
    """Error downloading or unpacking data"""


class DataTransformationError(CustomException):
    """Error transforming data"""


class DataValidationError(CustomException):
    """Data structure validation error"""


class ModelTrainingError(CustomException):
    """Error while training the model"""


class ModelEvaluationError(CustomException):
    """Error while evaluating the model"""


class PipelineExecutionError(CustomException):
    """General error in the pipeline run"""
