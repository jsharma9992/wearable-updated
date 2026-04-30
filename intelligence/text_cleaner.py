"""
Text Cleaner
────────────
Post-processes raw OCR output so it sounds natural through TTS.
Removes garbage, fixes common OCR artefacts, and chunks long
text at sentence boundaries.
"""

import re
import logging
from config import REMOVE_SPECIAL_CHARS

logger = logging.getLogger(__name__)

_GARBAGE_RE = [
    re.compile(r"^[^a-zA-Z0-9]+$"),
    re.compile(r"^[|\\/{}\[\]<>~`^]+$"),
    re.compile(r"^[^\w\s]{2,}$"), # only punctuation
]


class TextCleaner:

    def clean(self, text):
        """Full cleaning pipeline → ready-for-TTS string."""
        if not text:
            return ""
        text = self._strip_control(text)
        
        # Split by newlines to preserve paragraphs if multiple newlines exist
        paragraphs = re.split(r'\n\s*\n', text)
        cleaned_paragraphs = []
        
        for p in paragraphs:
            lines = p.split("\n")
            # filter garbage lines
            lines = [l.strip() for l in lines if self._valid(l.strip())]
            if not lines: continue
            
            # Intelligently join lines
            joined = lines[0]
            for i in range(1, len(lines)):
                prev = lines[i-1]
                curr = lines[i]
                # If prev ends with hyphen, remove hyphen and join
                if prev.endswith("-"):
                    joined = joined[:-1] + curr
                else:
                    joined += " " + curr
            
            cleaned_paragraphs.append(joined)
            
        text = "\n\n".join(cleaned_paragraphs)
        text = self._fix_ocr_errors(text)
        text = self._sentence_structure(text)
        text = self._remove_repeated_text(text)
        return re.sub(r"[ \t]+", " ", text).strip()

    def chunk_for_speech(self, text, max_len=120):
        """Split cleaned text at sentence boundaries for TTS."""
        if len(text) <= max_len:
            return [text] if text else []
        parts = re.split(r"(?<=[.!?])\s+", text)
        chunks, cur = [], ""
        for s in parts:
            if len(cur) + len(s) + 1 <= max_len:
                cur = f"{cur} {s}".strip()
            else:
                if cur:
                    chunks.append(cur)
                cur = s
        if cur:
            chunks.append(cur)
        return chunks

    # ── internals ───────────────────────────────────────────

    @staticmethod
    def _strip_control(t):
        t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", t)
        if REMOVE_SPECIAL_CHARS:
            # avoid over-cleaning numbers and punctuation
            t = re.sub(r"[^\w\s.,!?;:'\-/()&@#$%+\-=*]", "", t)
        return t

    @staticmethod
    def _valid(line):
        if not line:
            return False
        for pat in _GARBAGE_RE:
            if pat.match(line):
                return False
        alpha = sum(c.isalpha() for c in line)
        return alpha / max(len(line), 1) >= 0.3

    @staticmethod
    def _fix_ocr_errors(t):
        t = re.sub(r"\bl\b", "I", t)       # isolated 'l' → 'I'
        t = re.sub(r"\s{2,}", " ", t)
        return t

    @staticmethod
    def _sentence_structure(t):
        t = re.sub(r"\.(?=[A-Z])", ". ", t)
        words = t.split()
        if len(words) > 15 and not any(c in t for c in ".!?,;:"):
            out = []
            for i, w in enumerate(words):
                out.append(w)
                if (i + 1) % 10 == 0 and i < len(words) - 1:
                    out.append(",")
            t = " ".join(out)
        return t

    @staticmethod
    def _remove_repeated_text(t):
        """Removes duplicated sentences that might result from fusion"""
        sentences = re.split(r'(?<=[.!?])\s+', t)
        seen = set()
        out = []
        for s in sentences:
            s_clean = re.sub(r'\W+', '', s.lower())
            if s_clean and s_clean in seen:
                continue
            seen.add(s_clean)
            out.append(s)
        return " ".join(out)