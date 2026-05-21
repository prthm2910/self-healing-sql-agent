# ### --- IMPORTS AND GLOBALS --- ###
import os
import sys
import logging
from typing import Dict
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler

# ##############################################################################
# [Elaborative Breakdown] Contextual Thread/Async Logging & ContextVars
# Why ContextVars?
# In multi-tenant, asynchronous, or multi-threaded environments, standard thread-local
# storage (threading.local) fails to propagate context correctly across asynchronous
# tasks (async/await) because event loops yield execution across different tasks on 
# the same OS thread. ContextVars provide a clean, native Python mechanism to safely
# propagate context (such as 'user_id' and 'thread_id') across asynchronous task 
# switching as well as standard thread boundaries.
#
# Trade-offs:
# 1. Performance Overhead: Fetching and resolving dictionary values from ContextVars
#    on every single log record incurs a minor CPU latency overhead. We mitigate this
#    by setting highly structured defaults and keeping the context dictionaries small.
# 2. Memory leak risks: If large dictionaries or nested structures are passed, they
#    might survive task lifecycles. We strictly scope context variables to simple 
#    metadata strings ('user_id' and 'thread_id').
#
# Mental Model:
# Think of ContextVars as a thread-safe, task-safe dictionary stack that automatically
# follows the execution flow. When a log record is processed, our custom filter grabs the 
# current context snapshot and decorates the LogRecord object so the Formatter can 
# cleanly output it without having to pass 'extra' parameters on every log call.
# ##############################################################################

# Context-safe storage for logging metadata
log_context: ContextVar[Dict[str, str]] = ContextVar(
    "log_context", 
    default={"user_id": "system", "thread_id": "global"}
)


# ### --- CONTEXT FILTER SECTION --- ###

class ContextFilter(logging.Filter):
    """
    Injects context variables (user_id, thread_id) into log records dynamically.
    
    This ensures that downstream log formatters have access to key tenant metadata.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Intercepts the log record and injects contextual values from the ContextVar.
        
        Args:
            record: The standard logging.LogRecord object being processed.
            
        Returns:
            True to indicate the record should be logged.
        """
        # 1. Fetch current ContextVar map: Retrieves standard user/thread context dictionary
        # registered by the active calling node.
        context: Dict[str, str] = log_context.get()
        
        # 2. Decorate standard LogRecord: Dynamically inject properties so they are automatically
        # formatted into standard log formats without needing to pass 'extra' parameters.
        record.user_id = context.get("user_id", "system")  # type: ignore[attr-defined]
        record.thread_id = context.get("thread_id", "global")  # type: ignore[attr-defined]
        return True


# ### --- LOGGER SETUP SECTION --- ###

def setup_logger(name: str = "ai_assistant") -> logging.Logger:
    """
    Configures a professional, production-grade logger with metadata injection.
    
    Args:
        name: The name of the logger to retrieve or create. Defaults to "ai_assistant".
        
    Returns:
        A configured logging.Logger instance with Console and rotating File handlers.
    """
    logger: logging.Logger = logging.getLogger(name)
    
    # 1. Avoid duplicate handlers if setup has already run for this logger name to prevent double logging.
    if logger.hasHandlers():
        return logger

    # 2. Set base logging level to capture verbose debug markers for diagnostics.
    logger.setLevel(logging.DEBUG)
    
    # 3. Unified Contextual Format: timestamp, level, context parameters, file/line, and message.
    log_format: str = (
        '%(asctime)s | %(levelname)-8s | [%(user_id)s][%(thread_id)s] | '
        '%(name)s:%(funcName)s:%(lineno)d - %(message)s'
    )
    formatter: logging.Formatter = logging.Formatter(log_format)
    context_filter: ContextFilter = ContextFilter()

    # 4. Console Handler: Write high-priority INFO-level statements to stdout.
    console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(context_filter)
    
    # 5. Rotating File Handler: Write extremely verbose debug statements to local logs file.
    # We restrict maximum log capacity to 10MB per file and preserve up to 5 historical log backups,
    # ensuring disk allocations never grow out of control.
    os.makedirs("logs", exist_ok=True)
    file_handler: RotatingFileHandler = RotatingFileHandler(
        "logs/app.log", 
        maxBytes=10 * 1024 * 1024,  # 10 MB limit per file
        backupCount=5,              # Keep up to 5 historical log backups
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(context_filter)

    # 6. Attach configured handlers to the target logger instance
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger



# Shared global application logger instance
logger: logging.Logger = setup_logger()

