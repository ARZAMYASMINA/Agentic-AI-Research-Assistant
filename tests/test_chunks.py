from rag.chunks import chunk_text


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world.\n\nSecond paragraph.") == ["Hello world.\n\nSecond paragraph."]


def test_empty_text():
    assert chunk_text("  \n\n  ") == []


def test_long_text_is_split_and_nothing_is_lost():
    paras = [f"Paragraph {i}. " + ("word " * 120).strip() for i in range(10)]  # ~600 caractères chacun
    chunks = chunk_text("\n\n".join(paras), max_chars=1500, overlap=200)
    assert len(chunks) > 1
    assert all(len(c) <= 1500 + 200 + 2 for c in chunks)
    for p in paras:
        assert any(p in c for c in chunks)  # chaque paragraphe reste entier dans au moins un passage


def test_overlap_between_consecutive_chunks():
    paras = [f"P{i} " + ("alpha beta gamma " * 40).strip() for i in range(6)]
    chunks = chunk_text("\n\n".join(paras), max_chars=1000, overlap=100)
    assert len(chunks) >= 2
    tail_words = chunks[0].split()[-3:]
    assert all(w in chunks[1].split()[:40] for w in tail_words)


def test_text_without_punctuation_is_hard_split():
    chunks = chunk_text("a" * 4000, max_chars=1500, overlap=200)
    assert len(chunks) >= 3
    assert all(len(c) <= 1500 + 200 + 2 for c in chunks)


def test_heading_stays_with_following_text():
    text = "## Methods\n\nWe recruited 100 women."
    chunks = chunk_text(text)
    assert "## Methods" in chunks[0] and "We recruited 100 women." in chunks[0]
