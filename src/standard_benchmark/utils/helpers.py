"""Miscellaneous helper functions."""

import hashlib
import json
from pathlib import Path
from typing import Any


def hash_text(text: str) -> str:
    """Generate SHA256 hash of text.

    Args:
        text: Text to hash

    Returns:
        Hex digest of hash
    """
    return hashlib.sha256(text.encode()).hexdigest()


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(file_path: Path) -> Any:
    """Load JSON from file.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data
    """
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data: Any, file_path: Path, indent: int = 2) -> None:
    """Save data to JSON file.

    Args:
        data: Data to save
        file_path: Path to save to
        indent: JSON indentation
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=indent, default=str)


def load_jsonl(file_path: Path) -> list[dict]:
    """Load JSON Lines file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of parsed JSON objects
    """
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], file_path: Path) -> None:
    """Save data to JSON Lines file.

    Args:
        data: List of dictionaries to save
        file_path: Path to save to
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item, default=str) + "\n")


def append_jsonl(item: dict, file_path: Path) -> None:
    """Append item to JSON Lines file.

    Args:
        item: Dictionary to append
        file_path: Path to file
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "a") as f:
        f.write(json.dumps(item, default=str) + "\n")
