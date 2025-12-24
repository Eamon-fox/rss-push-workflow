"""Date formatting utilities."""

from datetime import datetime
from typing import Optional


# Standard format constants
DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def today() -> str:
    """
    Get today's date as YYYY-MM-DD string.

    Returns:
        Date string in YYYY-MM-DD format
    """
    return datetime.now().strftime(DATE_FORMAT)


def now_iso() -> str:
    """
    Get current datetime as ISO format string.

    Returns:
        Datetime string in ISO format
    """
    return datetime.now().isoformat()


def now_formatted(fmt: str = DATETIME_FORMAT) -> str:
    """
    Get current datetime with custom format.

    Args:
        fmt: strftime format string

    Returns:
        Formatted datetime string
    """
    return datetime.now().strftime(fmt)


def parse_date(date_str: str, default: Optional[datetime] = None) -> Optional[datetime]:
    """
    Parse date string (YYYY-MM-DD) to datetime.

    Args:
        date_str: Date string to parse
        default: Default value if parsing fails

    Returns:
        datetime object or default
    """
    try:
        return datetime.strptime(date_str, DATE_FORMAT)
    except (ValueError, TypeError):
        return default
