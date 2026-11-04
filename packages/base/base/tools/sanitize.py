"""Remove invisible characters that carry instructions a reader cannot see.

Prompt injection has no reliable interceptive defence, but one narrow class of
it is worth closing because the fix is deterministic and cheap: text that is
invisible when rendered yet still reaches the model. Unicode tag characters
are the clearest case — they encode ASCII in a block that renders as nothing,
so a comment, filename or retrieved document can carry instructions no
reviewer will ever see.

What is deliberately *not* stripped matters as much as what is:

- **Zero-width joiner** (U+200D) composes emoji sequences and shapes Arabic
  and Indic scripts. This project supports Arabic; removing it would corrupt
  real content.
- **Variation selectors** (U+FE00–U+FE0F) select emoji presentation. U+FE0F
  is what makes an emoji render as an emoji.
- **Bidirectional controls** (U+202A–U+202E, U+2066–U+2069) are how
  right-to-left text embeds left-to-right runs. They can be abused to reorder
  displayed text, but they are load-bearing for legitimate Arabic content, so
  a blanket strip trades a real bug for a theoretical one.

So this removes only characters with no legitimate role in natural language:
the tag block, and the zero-width characters that carry no shaping meaning.
It is a narrow, honest control, not a solution to prompt injection.
"""

# Unicode tag characters. Encode ASCII invisibly; no legitimate use in text.
TAG_BLOCK = range(0xE0000, 0xE0080)

# Invisible and non-shaping. U+200D (ZWJ) is excluded on purpose: it joins
# emoji sequences and shapes Arabic and Indic scripts.
INVISIBLE = frozenset(
    {
        0x200B,  # zero-width space
        0x200C,  # zero-width non-joiner
        0x2060,  # word joiner
        0xFEFF,  # zero-width no-break space / BOM
    }
)

_STRIPPED = frozenset(TAG_BLOCK) | INVISIBLE


def strip_invisible(text: str) -> str:
    """Remove invisible instruction-carrying characters from `text`.

    Returns the text unchanged when there is nothing to remove, which is the
    overwhelmingly common case, so this is cheap to call on every message.
    """
    if not text:
        return text
    if not any(ord(character) in _STRIPPED for character in text):
        return text
    return "".join(c for c in text if ord(c) not in _STRIPPED)


def has_invisible(text: str) -> bool:
    """Whether `text` contains characters `strip_invisible` would remove.

    Useful for logging or rejecting rather than silently rewriting, when a
    caller would rather know that someone tried.
    """
    return any(ord(character) in _STRIPPED for character in text or "")
