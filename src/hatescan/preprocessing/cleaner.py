import re

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str, preserve_case: bool = False) -> str:
    """Clean text with option to preserve case information for toxicity detection."""
    if text is None:
        return ""

    text = str(text)
    # Remove URLs, mentions, hashtags
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    # Convert to lowercase only if preserve_case is False
    if not preserve_case:
        text = text.lower()

    return text


def extract_case_features(text: str) -> dict:
    """Extract case-related features that might indicate toxicity."""
    if text is None:
        return {"has_uppercase": False, "all_caps_ratio": 0.0, "exclamation_count": 0}

    words = str(text).split()
    if not words:
        return {"has_uppercase": False, "all_caps_ratio": 0.0, "exclamation_count": 0}

    uppercase_words = sum(1 for word in words if word.isupper() and len(word) > 1)
    all_caps_words = sum(1 for word in words if word.isupper() and len(word) > 2)
    exclamation_count = str(text).count('!')

    return {
        "has_uppercase": uppercase_words > 0,
        "all_caps_ratio": all_caps_words / len(words),
        "exclamation_count": exclamation_count
    }

