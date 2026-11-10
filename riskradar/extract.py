"""Turn a 10-K HTML file into the text of its "Item 1A. Risk Factors" section."""
from __future__ import annotations

import re

from lxml import html as lxml_html

_START = r"item\s*1a\.?\s*[\-–—:.]?\s*risk\s+factors"
_END = (
    r"item\s*1b\.?\s*[\-–—:.]?\s*unresolved\s+staff\s+comments"
    r"|item\s*1c\.?\s*[\-–—:.]?\s*cybersecurity"
    r"|item\s*2\.?\s*[\-–—:.]?\s*(?:description\s+of\s+)?propert(?:y|ies)"
)
# A heading starts its own line. A cross-reference ("see Item 1A. Risk Factors") does not.
_LINE_START = r"^[^\w\n]{0,8}(?:part\s+i\b[^\w\n]{0,8})?"
START = re.compile(_LINE_START + "(?:" + _START + ")", re.I | re.M)
END = re.compile(_LINE_START + "(?:" + _END + ")", re.I | re.M)
_BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "ul", "ol", "hr", "title"}
_CELL = {"td", "th"}
_HIDDEN = re.compile(r"display\s*:\s*none", re.I)

MIN_SECTION_WORDS = 1500  # a real risk section is thousands of words long


class ExtractionError(ValueError):
    """The filing has no usable Item 1A section."""


def html_to_text(raw: bytes | str) -> str:
    """Readable text with one line per block element."""
    doc = lxml_html.fromstring(raw)
    for el in doc.xpath("//script | //style | //head"):
        el.drop_tree()
    # hidden inline-XBRL blocks hold metadata, not readable text
    for el in doc.xpath("//*[@style]"):
        if _HIDDEN.search(el.get("style", "")):
            el.drop_tree()
    for el in doc.iter():
        if not isinstance(el.tag, str):
            continue
        if el.tag in _BLOCK:
            el.tail = "\n" + (el.tail or "")
        elif el.tag in _CELL:
            el.tail = " " + (el.tail or "")
    text = doc.text_content().replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f]+", " ", text)
    return re.sub(r" ?\n[ \n]*", "\n", text)


def extract_risk_section(text: str) -> str:
    """Return the Item 1A text, or "" if it cannot be found.

    The words "Item 1A. Risk Factors" appear several times in a 10-K: in the
    table of contents, in cross-references and as the real heading. Only
    matches at the start of a line count as headings. Each heading runs to the
    next "Item 1B / 1C / 2" heading (or the next Item 1A heading). The table of
    contents gives a tiny candidate, so the longest candidate is the section.
    """
    starts = list(START.finditer(text))
    best = ""
    for i, m in enumerate(starts):
        end = END.search(text, m.end())
        if not end:
            continue
        stop = end.start()
        if i + 1 < len(starts):
            stop = min(stop, starts[i + 1].start())
        candidate = text[m.end(): stop]
        if len(candidate) > len(best):
            best = candidate
    return best.strip()


def risk_section_from_html(raw: bytes | str) -> str:
    """HTML in, validated Item 1A text out."""
    section = extract_risk_section(html_to_text(raw))
    words = len(section.split())
    if words < MIN_SECTION_WORDS:
        raise ExtractionError(
            f"No full 'Item 1A. Risk Factors' section found ({words} words). "
            "Some companies only refer to another document for their risk factors."
        )
    return section
