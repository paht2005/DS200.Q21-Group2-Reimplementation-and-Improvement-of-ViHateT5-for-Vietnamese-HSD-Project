"""Tests for ensemble CLI (scripts/run_ensemble.py) and label_remap logic."""
import sys
import os
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from run_ensemble import detect_model_type, detect_num_labels, get_default_models, get_all_models, parse_args


class TestDetectModelType:
    """Tests for detect_model_type function."""

    def test_detect_model_type_t5(self, tmp_path):
        """T5 model detected from architectures field."""
        config = {"architectures": ["T5ForConditionalGeneration"], "model_type": "t5"}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        model_type, returned_config = detect_model_type(str(tmp_path))
        assert model_type == "t5"
        assert returned_config["model_type"] == "t5"

    def test_detect_model_type_bert(self, tmp_path):
        """BERT/XLMRoberta model detected from architectures field."""
        config = {
            "architectures": ["XLMRobertaForSequenceClassification"],
            "model_type": "xlm-roberta",
            "num_labels": 2,
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        model_type, returned_config = detect_model_type(str(tmp_path))
        assert model_type == "bert"
        assert returned_config["num_labels"] == 2

    def test_detect_model_type_missing_config(self, tmp_path):
        """Raises FileNotFoundError when config.json missing."""
        with pytest.raises(FileNotFoundError):
            detect_model_type(str(tmp_path))

    def test_detect_model_type_fallback_model_type(self, tmp_path):
        """Falls back to model_type field when architectures empty."""
        config = {"architectures": [], "model_type": "t5"}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        model_type, _ = detect_model_type(str(tmp_path))
        assert model_type == "t5"


class TestDetectNumLabels:
    """Tests for detect_num_labels function."""

    def test_returns_num_labels(self):
        assert detect_num_labels({"num_labels": 3}) == 3

    def test_defaults_to_2(self):
        assert detect_num_labels({}) == 2


class TestGetDefaultModels:
    """Tests for get_default_models function."""

    def test_returns_three_models(self):
        models = get_default_models()
        assert len(models) == 3

    def test_all_paths_contain_models(self):
        models = get_default_models()
        for m in models:
            assert "models/" in m


class TestGetAllModels:
    """Tests for get_all_models function."""

    def test_excludes_pretrain(self):
        models = get_all_models()
        for m in models:
            assert "pretrain" not in m


class TestParseArgs:
    """Tests for parse_args function."""

    def test_defaults(self):
        args = parse_args([])
        assert args.task == "vihsd"
        assert args.batch_size == 8
        assert args.output == "results/ensemble_results.csv"
        assert args.models is None
        assert args.all_models is False
        assert args.no_optimize is False

    def test_custom_models(self):
        args = parse_args(["--models", "a", "b", "c"])
        assert args.models == ["a", "b", "c"]

    def test_task_victsd(self):
        args = parse_args(["--task", "victsd"])
        assert args.task == "victsd"

    def test_weights(self):
        args = parse_args(["--weights", "0.4", "0.3", "0.3"])
        assert args.weights == [0.4, 0.3, 0.3]


class TestLabelRemap:
    """Tests for label_remap functionality in ensemble."""

    def test_label_remap_applied_in_predict_vihsd(self):
        """BERT predictions are remapped correctly for ViHSD task."""
        from ensemble import HateSpeechEnsemble

        ensemble = HateSpeechEnsemble(device="cpu")

        # Mock a BERT model with label_remap
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        ensemble.models["test_bert"] = {
            "type": "bert",
            "model": mock_model,
            "tokenizer": mock_tokenizer,
            "label_remap": {0: 0, 1: 2},
        }
        ensemble.weights["test_bert"] = 1.0

        # Mock predict_bert to return class 0 and class 1
        with patch.object(ensemble, "predict_bert") as mock_pred:
            mock_pred.return_value = (np.array([0, 1, 1, 0]), np.array([]))

            result = ensemble.predict_vihsd(["a", "b", "c", "d"])
            # After remap: [0, 2, 2, 0] — with weighted vote, these become the predictions
            assert list(result) == [0, 2, 2, 0]

    def test_label_remap_not_applied_when_none(self):
        """No remapping when label_remap is None."""
        from ensemble import HateSpeechEnsemble

        ensemble = HateSpeechEnsemble(device="cpu")

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        ensemble.models["test_bert"] = {
            "type": "bert",
            "model": mock_model,
            "tokenizer": mock_tokenizer,
            "label_remap": None,
        }
        ensemble.weights["test_bert"] = 1.0

        with patch.object(ensemble, "predict_bert") as mock_pred:
            mock_pred.return_value = (np.array([0, 1, 1, 0]), np.array([]))

            result = ensemble.predict_vihsd(["a", "b", "c", "d"])
            # No remap: [0, 1, 1, 0] stay as-is
            assert list(result) == [0, 1, 1, 0]


class TestMPSDeviceDetection:
    """Tests for MPS device detection."""

    def test_device_is_valid(self):
        """HateSpeechEnsemble device is one of mps/cuda/cpu."""
        from ensemble import HateSpeechEnsemble

        e = HateSpeechEnsemble()
        assert e.device in ("mps", "cuda", "cpu")

    def test_explicit_device(self):
        """Explicit device parameter is respected."""
        from ensemble import HateSpeechEnsemble

        e = HateSpeechEnsemble(device="cpu")
        assert e.device == "cpu"
