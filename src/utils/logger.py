import os
import sys
import logging
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler

# Context-safe storage for logging metadata
log_context: ContextVar[dict] = ContextVar("log_context", default={"user_id": "system", "thread_id": "global"})

class ContextFilter(logging.Filter):
    """
    Injects context variables (user_id, thread_id) into log records.
    """
    def filter(self, record):
        context = log_context.get()
        record.user_id = context.get("user_id", "system")
        record.thread_id = context.get("thread_id", "global")
        return True

def setup_logger(name: str = "ai_assistant"):
    """
    Configures a professional logger with contextual metadata injection.
    """
    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)
    
    # Unified Contextual Format
    log_format = '%(asctime)s | %(levelname)-8s | [%(user_id)s][%(thread_id)s] | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
    formatter = logging.Formatter(log_format)
    
    context_filter = ContextFilter()

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(context_filter)
    
    # File Handler (Rotating)
    os.makedirs("logs", exist_ok=True)
    file_handler = RotatingFileHandler(
        "logs/app.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(context_filter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Shared instance
logger = setup_logger()
