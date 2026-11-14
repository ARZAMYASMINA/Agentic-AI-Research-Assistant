from rag.fusion import rrf_fuse, tokenize


def test_tokenize_removes_stopwords_and_punctuation():
    assert tokenize("The Deep-Learning of AI, in mammography!") == ["deep", "learning", "ai", "mammography"]


def test_rrf_rewards_agreement():
    dense = ["a", "b", "c"]
    sparse = ["c", "b", "d"]
    fused = [pid for pid, _ in rrf_fuse([dense, sparse])]
    # b est 2e dans les deux listes, c est 3e puis 1er, a est 1er puis absent
    assert fused[0] in ("b", "c")
    assert set(fused) == {"a", "b", "c", "d"}
    assert fused.index("b") < fused.index("d")


def test_rrf_single_list_keeps_order():
    assert [p for p, _ in rrf_fuse([["x", "y", "z"]])] == ["x", "y", "z"]


def test_rrf_deterministic_ties():
    out = rrf_fuse([["b"], ["a"]])
    assert [p for p, _ in out] == ["a", "b"]  # scores égaux -> ordre par identifiant
