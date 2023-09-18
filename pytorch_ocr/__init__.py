import logging
import os


def _get_level() -> int:
    """Get logging level from environment variable."""
    level = os.getenv("LOGGING_LEVEL", "INFO")
    try:
        return int(level)
    except ValueError:
        return getattr(logging, level)


# Set the logging level (e.g., INFO, DEBUG, etc.)
logging.getLogger(__name__).setLevel(_get_level())

# Define a formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Create a handler and add the formatter
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# Add the handler to the root logger
logging.getLogger().addHandler(handler)
