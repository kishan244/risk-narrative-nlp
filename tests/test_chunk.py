from riskradar.chunk import split_into_chunks

SENTENCE = "This is one plain sentence about a risk that we face every single year. "


def test_chunks_fit_the_model_window():
    chunks = split_into_chunks("\n".join(SENTENCE * 4 for _ in range(40)))
    assert len(chunks) > 5
    assert all(50 <= len(c.split()) <= 300 for c in chunks)


def test_headings_and_page_numbers_are_dropped():
    text = "Risks Related to Our Business\n23\n" + SENTENCE * 20
    chunks = split_into_chunks(text)
    assert not any("Risks Related to Our Business" in c for c in chunks)


def test_no_words_are_invented_or_reordered():
    text = "\n".join(SENTENCE * 3 for _ in range(10))
    joined = " ".join(split_into_chunks(text))
    assert joined in " ".join(text.split())


def test_very_long_sentence_is_cut_by_length():
    chunks = split_into_chunks("word " * 2000)
    assert max(len(c.split()) for c in chunks) <= 300


def test_short_text_gives_no_chunks():
    assert split_into_chunks("Too short to be a passage.") == []
