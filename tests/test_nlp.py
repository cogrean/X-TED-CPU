"""Tests for the optional NLP functions (requires pip install x-ted[nlp] and en_core_web_sm)."""

import pytest
import spacy
from xted import x_ted_compute, x_ted_util_transfer, x_ted_compute_from_text


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


# ── x_ted_util_transfer ─────────────────────────────────────────────────────

class TestUtilTransfer:
    def test_returns_parent_and_labels(self, nlp):
        parent, labels = x_ted_util_transfer("The cat sat.", nlp=nlp)
        assert isinstance(parent, list)
        assert isinstance(labels, list)
        assert len(parent) == len(labels)

    def test_root_is_minus_one(self, nlp):
        parent, labels = x_ted_util_transfer("The cat sat.", nlp=nlp)
        assert parent.count(-1) == 1
        assert parent[0] == -1

    def test_parent_indices_in_range(self, nlp):
        parent, labels = x_ted_util_transfer("The big brown dog ran quickly.", nlp=nlp)
        n = len(parent)
        for i, p in enumerate(parent):
            if i == 0:
                assert p == -1
            else:
                assert 0 <= p < n

    def test_labels_are_strings(self, nlp):
        parent, labels = x_ted_util_transfer("Hello world.", nlp=nlp)
        assert all(isinstance(l, str) for l in labels)

    def test_single_word(self, nlp):
        parent, labels = x_ted_util_transfer("Hello", nlp=nlp)
        assert len(parent) >= 1
        assert parent[0] == -1

    def test_dfs_preorder(self, nlp):
        """Parent index must always be less than child index in DFS preorder."""
        parent, labels = x_ted_util_transfer(
            "The quick brown fox jumps over the lazy dog.", nlp=nlp
        )
        for i in range(1, len(parent)):
            assert parent[i] < i


# ── x_ted_compute_from_text ──────────────────────────────────────────────────

class TestComputeFromText:
    def test_identical_text_is_zero(self, nlp):
        assert x_ted_compute_from_text("The cat sat.", "The cat sat.", nlp=nlp) == 0

    def test_different_text_is_positive(self, nlp):
        d = x_ted_compute_from_text("The cat sat.", "A dog ran.", nlp=nlp)
        assert d > 0

    def test_symmetric(self, nlp):
        a = "The cat sat on the mat."
        b = "A dog ran through the park."
        assert x_ted_compute_from_text(a, b, nlp=nlp) == \
               x_ted_compute_from_text(b, a, nlp=nlp)

    def test_consistent_with_manual_pipeline(self, nlp):
        """compute_from_text should give the same result as util_transfer + compute."""
        a = "I like apples."
        b = "She likes oranges."
        parent1, labels1 = x_ted_util_transfer(a, nlp=nlp)
        parent2, labels2 = x_ted_util_transfer(b, nlp=nlp)
        expected = x_ted_compute(parent1, labels1, parent2, labels2)
        assert x_ted_compute_from_text(a, b, nlp=nlp) == expected

    def test_multithreaded(self, nlp):
        a = "The cat sat on the mat."
        b = "A dog ran through the park."
        d1 = x_ted_compute_from_text(a, b, nlp=nlp, num_threads=1)
        d4 = x_ted_compute_from_text(a, b, nlp=nlp, num_threads=4)
        assert d1 == d4


# ── Graceful errors without spaCy ────────────────────────────────────────────

class TestMissingDependencyErrors:
    def test_util_transfer_error_without_spacy(self, monkeypatch):
        """Simulates spaCy not being installed."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "spacy":
                raise ImportError("No module named 'spacy'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="pip install x-ted"):
            x_ted_util_transfer("test")

    def test_compute_from_text_error_without_spacy(self, monkeypatch):
        """Simulates spaCy not being installed."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "spacy":
                raise ImportError("No module named 'spacy'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="pip install x-ted"):
            x_ted_compute_from_text("test", "test")
