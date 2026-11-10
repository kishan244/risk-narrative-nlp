import pytest

from riskradar.extract import ExtractionError, extract_risk_section, html_to_text, risk_section_from_html
from tests.conftest import make_10k_html


def test_finds_real_section_not_table_of_contents(ten_k_html):
    section = risk_section_from_html(ten_k_html)
    assert section.startswith("Our business depends")
    assert "Unresolved" not in section and "We own a large building" not in section


def test_cross_reference_is_not_a_heading(ten_k_html):
    assert "We make phones" not in risk_section_from_html(ten_k_html)
    assert "sell services to many customers" not in risk_section_from_html(ten_k_html)


def test_hidden_xbrl_is_dropped(ten_k_html):
    assert "hidden xbrl metadata" not in html_to_text(ten_k_html)


def test_heading_split_over_table_cells_is_one_line():
    text = html_to_text("<table><tr><td>Item 1A.</td><td>Risk Factors</td></tr></table>")
    assert "Item 1A. Risk Factors" in text


def test_short_section_raises_clear_error():
    with pytest.raises(ExtractionError, match="Item 1A"):
        risk_section_from_html(make_10k_html(risk_paragraphs=2))


def test_no_section_returns_empty_string():
    assert extract_risk_section("Nothing useful here") == ""


def test_section_can_end_at_cybersecurity_item():
    risks = "A real risk sentence that is long enough. " * 50
    text = "Item 1A. Risk Factors\n" + risks + "\nItem 1C. Cybersecurity\nOur program."
    assert "Our program" not in extract_risk_section(text)
