import logging
from functools import lru_cache
import sys


def setup_logging(level: str=None):

    formatter=logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] [%(message)s]",
        "%Y-%m-%d %H:%M:%S"
    )

    root_logger=logging.getLogger()
    root_logger.setLevel(level.upper())
    root_logger.handlers.clear()

    console_handler=logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler=logging.FileHandler("app.log")
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)


    @lru_cache
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)
    

    class LoggerMixin:

        @property
        def logger(self):
            return get_logger(self.__class__.__name__)