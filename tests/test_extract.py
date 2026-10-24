from src.extract import extract_risk_section, split_into_chunks

TOC = "Item 1A. Risk Factors 5\nItem 1B. Unresolved Staff Comments 14\nItem 2. Properties 14\n"
BODY = "Item 1A. Risk Factors\n\n" + ("Our business faces many serious risks that could hurt results. " * 60) + "\n\nItem 1B. Unresolved Staff Comments\nNone."


def test_skips_table_of_contents():
    section = extract_risk_section(TOC + "\n\nItem 1. Business\nWe make phones.\n\n" + BODY)
    assert section.startswith("Our business faces")
    assert "Unresolved" not in section
    assert "We make phones" not in section


def test_cross_reference_does_not_swallow_business_section():
    text = TOC + "Item 1. Business\nSee Item 1A Risk Factors for details. We make phones.\n\n" + BODY
    section = extract_risk_section(text)
    assert "We make phones" not in section


def test_chunks_have_reasonable_size():
    chunks = split_into_chunks(BODY.split("\n\n", 1)[1].rsplit("\n\n", 1)[0])
    assert chunks
    assert all(60 <= len(c.split()) <= 400 for c in chunks)


def test_empty_when_no_section():
    assert extract_risk_section("Nothing here") == ""
