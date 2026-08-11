"""
Greek-aware text normalization, shared by build_index.py and search.py.

The SAME normalization must run at index time and query time, or terms
won't match. Keep this module as the single source of truth.

Normalization steps:
  1. lowercase
  2. strip diacritics/accents (ά -> α, ϊ -> ι, 934 forms), and fold final
     sigma ς -> σ so "συνεδρίασης" and "συνεδριασησ" share a stem-ish form
  3. tokenize on non-letter boundaries (Greek + Latin letters + digits)
  4. drop very short tokens and Greek/English stopwords
"""

import re
import unicodedata
from pathlib import Path

MIN_TOKEN_LEN = 2

# External stopword files (one word per line). Placed next to this module.
_HERE = Path(__file__).resolve().parent
STOPWORD_FILES = [_HERE / "stopwordsGR.txt", _HERE / "stopwordsEn.txt"]

# Fallback lists, used only if the files above are missing.
_FALLBACK_GR = {
    "και", "να", "το", "τον", "την", "της", "του", "των", "τα", "τη", "τις",
    "ο", "η", "οι", "στο", "στη", "στην", "στον", "σε", "με", "για", "απο",
    "αν", "αλλα", "οτι", "που", "ειναι", "ηταν", "θα", "δεν", "ως", "τους",
}
_FALLBACK_EN = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "by", "at", "as", "from", "that",
}

# Greek + Latin letters and digits; everything else is a separator.
_TOKEN_RE = re.compile(r"[0-9a-zα-ω\u0370-\u03ff\u1f00-\u1fff]+", re.IGNORECASE)


def strip_accents(text: str) -> str:
    """Remove diacritics via NFD decomposition, then fold final sigma."""
    nfd = unicodedata.normalize("NFD", text)
    no_marks = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", no_marks).replace("\u03c2", "\u03c3")  # ς -> σ


def normalize(text: str) -> str:
    return strip_accents(text.lower())


def _load_stopwords() -> set[str]:
    """Load stopwords from the external files, normalized to match tokens.
    Each word is normalized (lowercased, accents/final-sigma stripped) so a
    file entry like 'από' matches the token 'απο'. Falls back to built-ins
    if a file is missing or unreadable."""
    words: set[str] = set()
    loaded_any = False
    for fp in STOPWORD_FILES:
        try:
            for line in fp.read_text(encoding="utf-8").splitlines():
                w = normalize(line.strip())
                if w:
                    words.add(w)
            loaded_any = True
        except OSError:
            pass  # file missing — handled by fallback below
    if not loaded_any:
        words = {normalize(w) for w in (_FALLBACK_GR | _FALLBACK_EN)}
    return words


STOPWORDS = _load_stopwords()


# --------------------------------------------------------------------------
# Light Greek stemmer (suffix stripping)
# --------------------------------------------------------------------------
# Greek is heavily inflected, so the same word appears in many forms
# (δημοτικος / δημοτικου / δημοτικο / δημοτικων ...). Collapsing them to a common
# stem lets a query in one form match documents written in another — a big recall
# win. This is a CONSERVATIVE suffix stripper (not full morphology): it removes a
# common inflectional ending only when at least MIN_STEM characters remain, so
# short words are left intact. Must run at BOTH index and query time (it does,
# because both call tokenize()).
_STEM_MIN = 3
_SUFFIXES = sorted({
    "ιοντουσαν", "οντουσαν", "ουσαμε", "ουσατε", "ουσαν", "ησαμε", "ησατε",
    "ησαν", "ηθηκαν", "στηκαν", "οντας", "ωντας", "μενος", "μενης", "μενου",
    "μενοι", "μενες", "μενων", "μενα", "μενη", "μενο", "αδες", "αδων", "ματος",
    "ματα", "ματων", "ουμε", "ουσε", "ουσα", "ονται", "ονταν", "οταν", "θηκα",
    "θηκε", "στηκα", "στηκε", "ησω", "ηξω", "ησα", "ησε", "εων", "εως", "ειας",
    "εια", "εις", "ιου", "ιων", "ιας", "ιες", "ιοι", "ους", "ουν", "ετε", "ον",
    "οι", "ων", "ης", "ες", "ας", "ει", "ια", "ιο", "ος", "ου",
    "α", "ε", "η", "ι", "ο", "υ",
}, key=len, reverse=True)


def stem(token: str) -> str:
    """Strip one common inflectional suffix, keeping >= _STEM_MIN chars."""
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= _STEM_MIN:
            return token[: -len(suf)]
    return token


def tokenize(text: str) -> list[str]:
    """Full pipeline: normalize -> split -> drop stopwords/short -> stem.

    Stemming is ON. Measured tradeoff (see evaluate.py): it helps the realistic
    CROSS-FORM eval where queries use different word forms than the doc
    (Hit@1 0.235 -> 0.294) while slightly hurting the exact-snippet eval
    (0.513 -> 0.470). Real users rarely paste verbatim text, so the cross-form
    gain is what counts. A gentler stemmer/lemmatizer could improve both."""
    norm = normalize(text)
    tokens = _TOKEN_RE.findall(norm)
    return [stem(t) for t in tokens
            if len(t) >= MIN_TOKEN_LEN and t not in STOPWORDS]



