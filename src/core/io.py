"""JSON I/O utilities with consistent error handling."""

import json
import logging
from pathlib import Path
from typing import Any, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


def load_json(
    path: Union[str, Path],
    default: T = None,
    *,
    encoding: str = "utf-8",
    log_errors: bool = True,
) -> Union[Any, T]:
    """
    Load JSON file with consistent error handling.

    Args:
        path: Path to JSON file
        default: Default value if file doesn't exist or is invalid
        encoding: File encoding (default: utf-8)
        log_errors: Whether to log errors (default: True)

    Returns:
        Parsed JSON data or default value
    """
    path = Path(path)

    if not path.exists():
        return default

    try:
        content = path.read_text(encoding=encoding).strip()
        if not content:
            return default
        return json.loads(content)
    except (json.JSONDecodeError, IOError) as e:
        if log_errors:
            logger.warning(f"Failed to load {path}: {e}")
        return default


def save_json(
    data: Any,
    path: Union[str, Path],
    *,
    encoding: str = "utf-8",
    indent: int = 2,
    ensure_ascii: bool = False,
    mkdir: bool = True,
    default: Any = str,
) -> bool:
    """
    Save data to JSON file with consistent formatting.

    Args:
        data: Data to serialize
        path: Output file path
        encoding: File encoding (default: utf-8)
        indent: JSON indentation (default: 2)
        ensure_ascii: Whether to escape non-ASCII (default: False)
        mkdir: Create parent directories if needed (default: True)
        default: Default serializer for non-JSON types (default: str)

    Returns:
        True if successful, False otherwise
    """
    path = Path(path)

    try:
        if mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)

        content = json.dumps(data, ensure_ascii=ensure_ascii, indent=indent, default=default)
        path.write_text(content, encoding=encoding)
        return True
    except Exception as e:
        logger.error(f"Failed to save {path}: {e}")
        return False


def load_json_or_empty_dict(path: Union[str, Path]) -> dict:
    """Convenience wrapper: load JSON as dict, default to {}."""
    return load_json(path, default={})


def load_json_or_empty_list(path: Union[str, Path]) -> list:
    """Convenience wrapper: load JSON as list, default to []."""
    return load_json(path, default=[])
