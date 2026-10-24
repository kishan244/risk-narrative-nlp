"""Turn a 10-K HTML file into clean Risk Factors text and chunks."""
import re

import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

START = re.compile(r"item\s*1a\.?\s*[\-–—:.]?\s*risk\s+factors", re.I)
END = re.compile(
    r"item\s*1b\.?\s*[\-–—:.]?\s*unresolved\s+staff\s+comments"
    r"|item\s*1c\.?\s*[\-–—:.]?\s*cybersecurity"
    r"|item\s*2\.?\s*[\-–—:.]?\s*properties",
    re.I,
)


def html_to_text(html):
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    # hidden XBRL header blocks hold metadata, not readable text
    for tag in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        tag.decompose()
    for tag in soup.find_all(["p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4"]):
        tag.append("\n")
    text = soup.get_text()
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text


def extract_risk_section(text):
    """Pick the Item 1A heading that gives the longest section.

    The table of contents also says "Item 1A Risk Factors", but the section
    after it is tiny, so the longest candidate is the real one.
    """
    starts = [m.end() for m in START.finditer(text)]
    best = ""
    for end in END.finditer(text):
        # the closest "Item 1A" heading before this "Item 1B/1C/2" heading
        before = [s for s in starts if s < end.start()]
        if not before:
            continue
        candidate = text[before[-1]: end.start()]
        if len(candidate) > len(best):
            best = candidate
    return best.strip()


def _is_noise(par):
    words = par.split()
    if len(words) < 8:
        return True  # headings, page numbers
    if re.fullmatch(r"[\d\s\|\-–—.]+", par):
        return True
    return False


def split_into_chunks(text, target_words=180, min_words=60):
    """Paragraph-aware chunks, so one chunk does not cut two risks in half."""
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in text.split("\n")]
    paragraphs = [p for p in paragraphs if p and not _is_noise(p)]

    # very long paragraphs are split at sentence ends
    sentences = []
    for p in paragraphs:
        if len(p.split()) > target_words:
            sentences.extend(re.split(r"(?<=[.;])\s+(?=[A-Z])", p))
        else:
            sentences.append(p)

    chunks, current = [], []
    for s in sentences:
        current.extend(s.split())
        if len(current) >= target_words:
            chunks.append(" ".join(current))
            current = []
    if len(current) >= min_words:
        chunks.append(" ".join(current))
    return chunks
