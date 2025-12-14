import re
import uuid
from typing import List
from datetime import datetime


def generate_uuid() -> str:
    """
    Generate a UUID string.
    """
    return str(uuid.uuid4())


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing.
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def split_text_by_sentences(text: str, max_length: int = 400) -> List[str]:
    """
    Split text into chunks by sentences, keeping chunks under max_length.
    """
    sentences = re.split(r'[.!?]+\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def extract_chapter_info_from_path(file_path: str) -> tuple:
    """
    Extract chapter number and section title from file path.
    """
    # Example: docs/3-vla/chapter3.md -> chapter 3, section title from filename
    import os
    directory, filename = os.path.split(file_path)
    dir_name = os.path.basename(directory)

    # Extract chapter number from directory name like "3-vla"
    chapter_match = re.search(r'^(\d+)', dir_name)
    chapter_number = int(chapter_match.group(1)) if chapter_match else 0

    # Extract section title from filename
    section_title = os.path.splitext(filename)[0]

    return chapter_number, section_title


def validate_language_code(language: str) -> bool:
    """
    Validate if the language code is supported.
    """
    from .constants import SUPPORTED_LANGUAGES
    return language in SUPPORTED_LANGUAGES


def format_timestamp(dt: datetime) -> str:
    """
    Format datetime to ISO 8601 string.
    """
    return dt.isoformat()


def calculate_similarity_score(distance: float, max_distance: float = 1.0) -> float:
    """
    Calculate similarity score from distance (0-1 scale, where 1 is most similar).
    """
    return max(0.0, min(1.0, 1.0 - (distance / max_distance)))