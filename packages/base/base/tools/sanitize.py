"""Strip invisible characters that carry instructions a reader cannot see."""

# Unicode tag characters encode ASCII in a block that renders as nothing.
TAG_BLOCK = range(0xE0000, 0xE0080)

# Invisible and non-shaping. U+200D (ZWJ) is excluded on purpose: it composes
# emoji sequences and shapes Arabic and Indic scripts, as do variation
# selectors and bidirectional controls.
INVISIBLE = frozenset({0x200B, 0x200C, 0x2060, 0xFEFF})

_STRIPPED = frozenset(TAG_BLOCK) | INVISIBLE


def strip_invisible(text: str) -> str:
    """Remove invisible instruction-carrying characters from `text`."""
    if not text or not any(ord(c) in _STRIPPED for c in text):
        return text
    return "".join(c for c in text if ord(c) not in _STRIPPED)


def has_invisible(text: str) -> bool:
    """Whether `text` contains anything `strip_invisible` would remove."""
    return any(ord(c) in _STRIPPED for c in text or "")
