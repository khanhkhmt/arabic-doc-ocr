"""Text helpers: Arabic code-point fixes, Hebrew detection, word-level comparison of two readings."""

from __future__ import annotations

import difflib
import re

# --- Arabic code points ---------------------------------------------------------------------------------------------

_PERSIAN = str.maketrans({"ی": "ي", "ک": "ك", "ۃ": "ة", "ہ": "ه", "ۀ": "ة",
                          **{chr(0x06F0 + i): chr(0x0660 + i) for i in range(10)}})
_LETTERS = "ء-يٮ-ۓ"
_LONE_WAW = re.compile(rf"(?<![{_LETTERS}ً-ْ])و([ً-ْ]*)\s+(?=[{_LETTERS}])")


def fix_arabic(s: str) -> str:
    """Persian code points → the Arabic letters with the same shape (ی→ي, ک→ك, ۸→٨ …); a lone "و" is joined to the
    next word ("و" is never a word on its own). Nothing else is changed.

    dots.mocr on an enlarged page often emits Persian code points; the line models occasionally do too."""
    return _LONE_WAW.sub(r"و\1", (s or "").translate(_PERSIAN))


def clean_repeated_substrings(text: str) -> str:
    """From the Baseer-Nakba authors' inference code: cut a tail repeated ≥ 10 times (generation loop)."""
    n = len(text)
    if n < 200:
        return text
    for length in range(2, n // 10 + 1):
        cand, count, i = text[-length:], 0, n - length
        while i >= 0 and text[i:i + length] == cand:
            count += 1
            i -= length
        if count >= 10:
            return text[:n - length * (count - 1)]
    return text


# --- Hebrew blocks (notary stamps / certificates) -------------------------------------------------------------------

_HE = re.compile(r"[א-ת]")
_AR = re.compile(r"[ء-ي]")


def is_hebrew(s: str) -> bool:
    """Mostly Hebrew: ≥ 4 Hebrew letters and not fewer Hebrew than Arabic letters (digits / dates do not count)."""
    he = len(_HE.findall(s or ""))
    return he >= 4 and he >= len(_AR.findall(s or ""))


# --- comparing two readings -----------------------------------------------------------------------------------------

_HARAKAT = re.compile(r"[ً-ْٰـ]")
_ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"})
_PUNCT = re.compile(r"[\s\.\,\:\;\!\?\(\)\[\]\"'«»،؛؟\-–—#\*_|]+")  # + Markdown marks from dots


def _key(w: str) -> str:
    """Ignore differences that do not change the reading: harakat, hamza/alef, ى/ي, ة/ه, attached punctuation."""
    return _PUNCT.sub("", _HARAKAT.sub("", w)).translate(_ALEF)


def compare_words(main: str, check: str) -> list[tuple[str, bool, str]]:
    """→ [(word of the main reading, suspicious?, what the check reading has there)].
    Suspicious = the word lies in a span where the two readings differ."""
    a, b = main.split(), check.split()
    sm = difflib.SequenceMatcher(None, [_key(w) for w in a], [_key(w) for w in b], autojunk=False)
    out: list[tuple[str, bool, str]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        alt = " ".join(b[j1:j2])
        for w in a[i1:i2]:
            out.append((w, tag != "equal", "" if tag == "equal" else (alt or "(nothing)")))
    return out


def agreement(main: str, check: str) -> float:
    """Share of words of `main` that `check` reads the same way (0–1); 1.0 for an empty line."""
    words = compare_words(main, check)
    return 1.0 if not words else sum(not s for _, s, _ in words) / len(words)
