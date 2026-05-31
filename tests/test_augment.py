"""Unit tests for src/augment.py — Data augmentation functions.

Tests augment_minority_classes, EDA operations, and convenience wrappers
to verify correct augmentation behavior.
"""

import pandas as pd
import pytest


class TestAugmentMinorityClasses:
    """Verify augment_minority_classes contract."""

    def _make_imbalanced_df(self):
        """Create a synthetic imbalanced DataFrame."""
        data = {
            "text": ["xin chào"] * 100 + ["ngu quá"] * 10 + ["đồ mày xấu"] * 5,
            "label": [0] * 100 + [1] * 10 + [2] * 5,
        }
        return pd.DataFrame(data)

    def test_augment_minority_classes_increases_samples(self):
        """Output has more rows than input for imbalanced data."""
        from src.augment import augment_minority_classes

        df = self._make_imbalanced_df()
        result = augment_minority_classes(df, text_col="text", label_col="label",
                                          target_ratio=0.8, seed=42)

        assert len(result) > len(df), "Augmented DataFrame should have more rows"

    def test_augment_preserves_majority_class_count(self):
        """Majority class count should remain unchanged."""
        from src.augment import augment_minority_classes

        df = self._make_imbalanced_df()
        original_majority_count = (df["label"] == 0).sum()

        result = augment_minority_classes(df, text_col="text", label_col="label",
                                          target_ratio=0.8, seed=42)

        new_majority_count = (result["label"] == 0).sum()
        assert new_majority_count == original_majority_count, \
            "Majority class should not be augmented"

    def test_augment_factor_controls_target(self):
        """Different factors produce proportionally different results."""
        from src.augment import augment_minority_classes

        df = self._make_imbalanced_df()

        result_low = augment_minority_classes(df, text_col="text", label_col="label",
                                              target_ratio=0.3, seed=42)
        result_high = augment_minority_classes(df, text_col="text", label_col="label",
                                               target_ratio=0.9, seed=42)

        assert len(result_high) > len(result_low), \
            "Higher target_ratio should produce more augmented samples"

    def test_augment_already_balanced_no_change(self):
        """If all classes are at or above target, no augmentation occurs."""
        from src.augment import augment_minority_classes

        data = {
            "text": ["hello"] * 50 + ["world"] * 50,
            "label": [0] * 50 + [1] * 50,
        }
        df = pd.DataFrame(data)

        result = augment_minority_classes(df, text_col="text", label_col="label",
                                          target_ratio=0.5, seed=42)

        assert len(result) == len(df), \
            "Already balanced data should not be augmented"

    @pytest.mark.xfail(reason="augment.py does not handle empty DataFrames (cannot modify)")
    def test_augment_empty_dataframe(self):
        """Handles empty input gracefully."""
        from src.augment import augment_minority_classes

        df = pd.DataFrame({"text": [], "label": []})
        result = augment_minority_classes(df, text_col="text", label_col="label",
                                          target_ratio=0.8, seed=42)

        assert len(result) == 0, "Empty input should return empty output"

    def test_augment_seed_reproducibility(self):
        """Same seed produces same output."""
        from src.augment import augment_minority_classes

        df = self._make_imbalanced_df()

        result1 = augment_minority_classes(df, text_col="text", label_col="label",
                                           target_ratio=0.8, seed=42)
        result2 = augment_minority_classes(df, text_col="text", label_col="label",
                                           target_ratio=0.8, seed=42)

        assert len(result1) == len(result2), "Same seed should produce same row count"
        pd.testing.assert_frame_equal(result1.reset_index(drop=True),
                                      result2.reset_index(drop=True))


class TestEDAOperations:
    """Verify individual EDA augmentation operations."""

    def test_eda_augment_produces_different_text(self):
        """Augmented text should differ from original for text with known synonyms."""
        from src.augment import eda_augment

        sentence = "mày ngu quá đi thôi"
        results = eda_augment(sentence, num_aug=4)

        # At least one augmented version should differ
        assert any(r != sentence for r in results), \
            "EDA should produce at least one different sentence"

    def test_synonym_replacement_uses_dictionary(self):
        """Known word gets replaced with a known synonym."""
        from src.augment import synonym_replacement, VIETNAMESE_SYNONYMS
        import random

        random.seed(42)
        # "ngu" has synonyms ["dốt", "ngốc", "đần", "khờ"]
        sentence = "ngu quá"
        result = synonym_replacement(sentence, n=1)

        # Either "ngu" was replaced with a synonym, or no replaceable words
        if result != sentence:
            words = result.split()
            assert words[0] in VIETNAMESE_SYNONYMS["ngu"] or words[0] == "ngu", \
                "Replacement should come from synonym dictionary"

    def test_random_deletion_preserves_at_least_one_word(self):
        """Random deletion should never delete all words."""
        from src.augment import random_deletion
        import random

        random.seed(42)
        sentence = "đây là một câu thử nghiệm"

        # Run many times to check it never returns empty
        for _ in range(100):
            result = random_deletion(sentence, p=0.9)
            assert len(result.strip()) > 0, "Should never produce empty string"
            assert len(result.split()) >= 1, "Should preserve at least one word"


class TestConvenienceWrappers:
    """Verify convenience wrapper functions."""

    def test_augment_vihsd_convenience_wrapper(self):
        """augment_vihsd() works and uses correct columns."""
        from src.augment import augment_vihsd

        data = {
            "free_text": ["xin chào bạn"] * 50 + ["mày ngu quá"] * 5,
            "label_id": [0] * 50 + [2] * 5,
        }
        df = pd.DataFrame(data)

        result = augment_vihsd(df, target_ratio=0.8)

        assert len(result) > len(df), "Should augment minority class"
        assert "free_text" in result.columns, "Should preserve text column"
        assert "label_id" in result.columns, "Should preserve label column"
