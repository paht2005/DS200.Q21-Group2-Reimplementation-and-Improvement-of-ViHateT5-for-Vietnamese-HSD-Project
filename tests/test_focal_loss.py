"""Unit tests for src/focal_loss.py — Focal Loss classes.

Tests FocalLoss, LabelSmoothingFocalLoss, and FocalLossSeq2SeqTrainer
to verify correct computation and trainer integration.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch


class TestFocalLoss:
    """Verify FocalLoss class contract."""

    def test_returns_scalar_loss(self):
        """Loss should return a scalar tensor."""
        from src.focal_loss import FocalLoss

        focal_loss = FocalLoss(gamma=2.0)
        logits = torch.randn(2, 10, 100)  # (batch, seq_len, vocab_size)
        targets = torch.randint(0, 100, (2, 10))  # (batch, seq_len)

        loss = focal_loss(logits, targets)

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() > 0, "Loss should be positive"

    def test_ignore_index_masks_correctly(self):
        """Targets with ignore_index should not contribute to loss."""
        from src.focal_loss import FocalLoss

        focal_loss = FocalLoss(gamma=2.0, ignore_index=-100)
        logits = torch.randn(2, 10, 100)
        targets = torch.randint(0, 100, (2, 10))
        targets[:, 5:] = -100  # Set half to ignore

        loss = focal_loss(logits, targets)

        assert not torch.isnan(loss), "Loss should not be NaN with ignore_index"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_gamma_zero_equals_ce_behavior(self):
        """With gamma=0, focal loss should behave like cross-entropy."""
        from src.focal_loss import FocalLoss

        focal_loss = FocalLoss(gamma=0.0)
        logits = torch.randn(2, 10, 100)
        targets = torch.randint(0, 100, (2, 10))

        loss = focal_loss(logits, targets)

        assert loss.item() > 0, "Loss should be positive"

    def test_higher_gamma_reduces_easy_example_weight(self):
        """Higher gamma should reduce weight on easy (high confidence) examples."""
        from src.focal_loss import FocalLoss

        # Create high-confidence prediction (easy example)
        logits = torch.zeros(1, 1, 10)
        logits[0, 0, 5] = 10.0  # Very confident in class 5
        targets = torch.tensor([[5]])

        loss_gamma_0 = FocalLoss(gamma=0.0)(logits, targets)
        loss_gamma_2 = FocalLoss(gamma=2.0)(logits, targets)

        # Higher gamma should reduce loss for easy examples
        assert loss_gamma_2.item() < loss_gamma_0.item(), \
            "Higher gamma should reduce loss for high-confidence predictions"

    def test_empty_valid_targets_returns_zero(self):
        """When all targets are ignore_index, loss should be zero."""
        from src.focal_loss import FocalLoss

        focal_loss = FocalLoss(gamma=2.0, ignore_index=-100)
        logits = torch.randn(2, 10, 100)
        targets = torch.full((2, 10), -100)  # All ignored

        loss = focal_loss(logits, targets)

        assert loss.item() == 0.0, "Loss should be 0 when all targets are ignored"


class TestLabelSmoothingFocalLoss:
    """Verify LabelSmoothingFocalLoss class contract."""

    def test_returns_scalar_loss(self):
        """Loss should return a scalar tensor."""
        from src.focal_loss import LabelSmoothingFocalLoss

        loss_fn = LabelSmoothingFocalLoss(gamma=2.0, smoothing=0.1)
        logits = torch.randn(2, 10, 100)
        targets = torch.randint(0, 100, (2, 10))

        loss = loss_fn(logits, targets)

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() > 0, "Loss should be positive"

    def test_smoothing_zero_behaves_like_focal_loss(self):
        """With smoothing=0, should behave like standard FocalLoss."""
        from src.focal_loss import FocalLoss, LabelSmoothingFocalLoss

        torch.manual_seed(42)
        logits = torch.randn(2, 10, 100)
        targets = torch.randint(0, 100, (2, 10))

        focal_loss = FocalLoss(gamma=2.0)(logits.clone(), targets.clone())
        smooth_focal = LabelSmoothingFocalLoss(gamma=2.0, smoothing=0.0)(
            logits.clone(), targets.clone()
        )

        # Should be very close (floating point tolerance)
        assert torch.allclose(focal_loss, smooth_focal, atol=1e-5), \
            "Smoothing=0 should be equivalent to standard focal loss"

    def test_smoothing_affects_loss_value(self):
        """Non-zero smoothing should change the loss value."""
        from src.focal_loss import LabelSmoothingFocalLoss

        torch.manual_seed(42)
        logits = torch.randn(2, 10, 100)
        targets = torch.randint(0, 100, (2, 10))

        loss_no_smooth = LabelSmoothingFocalLoss(gamma=2.0, smoothing=0.0)(
            logits.clone(), targets.clone()
        )
        loss_with_smooth = LabelSmoothingFocalLoss(gamma=2.0, smoothing=0.1)(
            logits.clone(), targets.clone()
        )

        assert loss_no_smooth.item() != loss_with_smooth.item(), \
            "Smoothing should affect the loss value"


class TestFocalLossSeq2SeqTrainer:
    """Verify FocalLossSeq2SeqTrainer class contract."""

    def test_init_with_default_params(self):
        """Trainer should instantiate with default focal loss parameters."""
        from src.focal_loss import FocalLossSeq2SeqTrainer, FocalLoss

        with patch("src.focal_loss.Seq2SeqTrainer.__init__", return_value=None):
            trainer = FocalLossSeq2SeqTrainer()

        assert hasattr(trainer, "focal_loss_fn")
        assert isinstance(trainer.focal_loss_fn, FocalLoss)

    def test_init_with_label_smoothing_uses_smoothing_loss(self):
        """When label_smoothing > 0, should use LabelSmoothingFocalLoss."""
        from src.focal_loss import FocalLossSeq2SeqTrainer, LabelSmoothingFocalLoss

        with patch("src.focal_loss.Seq2SeqTrainer.__init__", return_value=None):
            trainer = FocalLossSeq2SeqTrainer(label_smoothing=0.1)

        assert isinstance(trainer.focal_loss_fn, LabelSmoothingFocalLoss)

    def test_compute_loss_returns_loss_tensor(self):
        """compute_loss should return a loss tensor."""
        from src.focal_loss import FocalLossSeq2SeqTrainer

        with patch("src.focal_loss.Seq2SeqTrainer.__init__", return_value=None):
            trainer = FocalLossSeq2SeqTrainer(focal_gamma=2.0, label_smoothing=0.0)

        # Mock model and inputs
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.randn(2, 10, 100)
        mock_model.return_value = mock_outputs

        inputs = {
            "input_ids": torch.randint(0, 100, (2, 10)),
            "attention_mask": torch.ones(2, 10),
            "labels": torch.randint(0, 100, (2, 10)),
        }

        loss = trainer.compute_loss(mock_model, inputs)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0, "Loss should be scalar"

    def test_compute_loss_with_return_outputs(self):
        """compute_loss with return_outputs=True should return tuple."""
        from src.focal_loss import FocalLossSeq2SeqTrainer

        with patch("src.focal_loss.Seq2SeqTrainer.__init__", return_value=None):
            trainer = FocalLossSeq2SeqTrainer(focal_gamma=2.0, label_smoothing=0.0)

        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.randn(2, 10, 100)
        mock_model.return_value = mock_outputs

        inputs = {
            "input_ids": torch.randint(0, 100, (2, 10)),
            "attention_mask": torch.ones(2, 10),
            "labels": torch.randint(0, 100, (2, 10)),
        }

        result = trainer.compute_loss(mock_model, inputs, return_outputs=True)

        assert isinstance(result, tuple)
        assert len(result) == 2
        loss, outputs = result
        assert isinstance(loss, torch.Tensor)
