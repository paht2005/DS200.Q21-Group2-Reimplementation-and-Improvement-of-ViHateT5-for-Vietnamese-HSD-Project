"""
Data loading utilities for Vietnamese hate speech datasets.
"""

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from typing import Tuple, Dict, Any

try:
    from underthesea import word_tokenize
except ImportError:
    word_tokenize = None  # Will raise later if PhoBERT requires it


class TextDataset(Dataset):
    """PyTorch Dataset for text classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length, use_word_seg=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_word_seg = use_word_seg

        if self.use_word_seg and word_tokenize is None:
            raise ImportError(
                "PhoBERT requires Vietnamese word segmentation. Install underthesea to proceed."
            )
    
    def __len__(self):
        return len(self.texts)
    
    def _maybe_segment(self, text: str) -> str:
        if not self.use_word_seg:
            return text
        return " ".join(word_tokenize(text))
    
    def __getitem__(self, idx):
        text = self._maybe_segment(str(self.texts[idx]))
        label = int(self.labels[idx])
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_vihsd() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load ViHSD dataset.
    
    Returns:
        Tuple of (train_df, val_df, test_df, metadata)
    """
    vihsd = load_dataset("sonlam1102/vihsd")
    
    # Check available splits
    available_splits = list(vihsd.keys())
    print(f"  Available splits: {available_splits}")
    
    # Try to get train/val/test splits directly
    if "train" in available_splits:
        train_df = vihsd["train"].to_pandas().dropna()
    else:
        raise ValueError(f"ViHSD dataset does not have 'train' split. Available: {available_splits}")
    
    if "validation" in available_splits:
        val_df = vihsd["validation"].to_pandas().dropna()
    elif "val" in available_splits:
        val_df = vihsd["val"].to_pandas().dropna()
    else:
        # No validation split, create empty DataFrame
        val_df = pd.DataFrame(columns=train_df.columns)
        print("  ⚠️  No validation split found, using empty validation set")
    
    if "test" in available_splits:
        test_df = vihsd["test"].to_pandas().dropna()
    else:
        # No test split, create empty DataFrame
        test_df = pd.DataFrame(columns=train_df.columns)
        print("  ⚠️  No test split found, using empty test set")
    
    # Calculate num_labels from all available data
    all_labels = train_df["label_id"]
    if len(val_df) > 0:
        all_labels = pd.concat([all_labels, val_df["label_id"]])
    if len(test_df) > 0:
        all_labels = pd.concat([all_labels, test_df["label_id"]])
    
    metadata = {
        "name": "ViHSD",
        "text_col": "free_text",
        "label_col": "label_id",
        "num_labels": int(all_labels.nunique())
    }
    
    return train_df, val_df, test_df, metadata


def load_victsd() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load ViCTSD dataset.
    
    Returns:
        Tuple of (train_df, val_df, test_df, metadata)
    """
    train_set = load_dataset("tarudesu/ViCTSD", split="train")
    val_set = load_dataset("tarudesu/ViCTSD", split="validation")
    test_set = load_dataset("tarudesu/ViCTSD", split="test")
    
    train_df = train_set.to_pandas().dropna()
    val_df = val_set.to_pandas().dropna()
    test_df = test_set.to_pandas().dropna()
    
    metadata = {
        "name": "ViCTSD",
        "text_col": "Comment",
        "label_col": "Toxicity",
        "num_labels": 2  # Binary: 0=NONE, 1=TOXIC
    }
    
    return train_df, val_df, test_df, metadata


def load_vihos() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load ViHOS dataset.
    
    Returns:
        Tuple of (train_df, val_df, test_df, metadata)
    """
    base = "https://raw.githubusercontent.com/phusroyal/ViHOS/master/"
    
    data_files = {
        "train": base + "data/Span_Extraction_based_version/train.csv",
        "validation": base + "data/Span_Extraction_based_version/dev.csv",
        "test": base + "data/Test_data/test.csv",
    }
    
    vihos = load_dataset("csv", data_files=data_files)
    
    train_df = vihos["train"].to_pandas().dropna()
    val_df = vihos["validation"].to_pandas().dropna()
    test_df = vihos["test"].to_pandas().dropna()
    
    # Create binary label: has hate span or not
    def has_hate_span(spans_str):
        if pd.isna(spans_str) or spans_str == "[]" or spans_str == "":
            return 0  # CLEAN
        return 1  # HAS HATE
    
    train_df["has_hate"] = train_df["index_spans"].apply(has_hate_span)
    val_df["has_hate"] = val_df["index_spans"].apply(has_hate_span)
    test_df["has_hate"] = test_df["index_spans"].apply(has_hate_span)
    
    metadata = {
        "name": "ViHOS",
        "text_col": "content",
        "label_col": "has_hate",
        "num_labels": 2  # Binary: 0=CLEAN, 1=HAS_HATE_SPANS
    }
    
    return train_df, val_df, test_df, metadata


def load_vihsd_processed() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load ViHSD processed dataset (trinhtrantran122/ViHSD_processed).
    
    Returns:
        Tuple of (train_df, val_df, test_df, metadata)
    """
    base_url = "https://huggingface.co/datasets/trinhtrantran122/ViHSD_processed/resolve/main/"
    data_files = {
        "train": base_url + "train_processed.csv",
        "validation": base_url + "dev_processed.csv",
        "test": base_url + "test_processed.csv",
    }
    
    dataset = load_dataset("csv", data_files=data_files)
    
    train_df = dataset["train"].to_pandas().dropna()
    val_df = dataset["validation"].to_pandas().dropna()
    test_df = dataset["test"].to_pandas().dropna()
    
    # Map string labels to integers
    # Based on inspection: 'none' -> 0, 'hate' -> 1
    label_map = {"none": 0, "hate": 1}
    
    def map_label(label):
        # Default to -1 if unknown, but we expect only none/hate
        return label_map.get(str(label).strip(), -1)
        
    train_df["label_id"] = train_df["label"].apply(map_label)
    val_df["label_id"] = val_df["label"].apply(map_label)
    test_df["label_id"] = test_df["label"].apply(map_label)
    
    metadata = {
        "name": "ViHSD_processed",
        "text_col": "free_text",
        "label_col": "label_id",
        "num_labels": 2
    }
    
    return train_df, val_df, test_df, metadata


def load_voz_hsd_2m(split_name: str = "balanced", dev_ratio: float = 0.1, max_samples: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load VOZ-HSD 2M dataset from HuggingFace.
    
    Args:
        split_name: "full" (data.csv), "balanced" (data_balance.csv), or "hate_only" (data_full_date.csv)
        dev_ratio: Validation split ratio (default: 0.1)
        max_samples: Maximum number of samples to use (limit BEFORE splitting). If None, use all samples.
    
    Returns:
        Tuple of (train_df, val_df, test_df, metadata)
    """
    from sklearn.model_selection import train_test_split
    
    # Load the correct CSV file based on split_name
    if split_name == "hate_only":
        file_url = "https://huggingface.co/datasets/NCPhat2005/re_VOZ-HSD/resolve/main/data_full_date.csv"
        print(f"  Loading data_full_date.csv (hate_only) from HuggingFace...")
        full_df = pd.read_csv(file_url).dropna()
    elif split_name == "balanced":
        file_url = "https://huggingface.co/datasets/NCPhat2005/re_VOZ-HSD/resolve/main/data_balance.csv"
        print(f"  Loading data_balance.csv (balanced) from HuggingFace...")
        full_df = pd.read_csv(file_url).dropna()
    elif split_name == "full":
        file_url = "https://huggingface.co/datasets/NCPhat2005/re_VOZ-HSD/resolve/main/data.csv"
        print(f"  Loading data.csv (full dataset) from HuggingFace...")
        full_df = pd.read_csv(file_url).dropna()
    else:
        # Default: load balanced dataset
        file_url = "https://huggingface.co/datasets/NCPhat2005/re_VOZ-HSD/resolve/main/data_balance.csv"
        print(f"  Loading default (balanced) dataset from HuggingFace...")
        full_df = pd.read_csv(file_url).dropna()
    
    print(f"  Total samples: {len(full_df)}")
    print(f"  Class distribution: {full_df['labels'].value_counts().to_dict()}")
    
    # Limit samples BEFORE splitting (to avoid wasting time on val/test samples)
    if max_samples is not None and len(full_df) > max_samples:
        original_size = len(full_df)
        print(f"  📊 Limiting dataset to {max_samples} samples (before splitting)...")
        # Stratified sampling to maintain class distribution
        full_df, _ = train_test_split(
            full_df,
            train_size=max_samples,
            random_state=42,
            stratify=full_df["labels"]
        )
        print(f"  ✅ Reduced from {original_size} to {len(full_df)} samples (stratified)")
    
    # Split based on dev_ratio
    # test_ratio = dev_ratio (same as validation)
    test_ratio = dev_ratio
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        full_df, test_size=(dev_ratio + test_ratio), random_state=42, stratify=full_df["labels"]
    )
    # Second split: val vs test (50-50 of remaining)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["labels"]
    )
    
    metadata = {
        "name": f"VOZ-HSD_{split_name}",
        "text_col": "texts",
        "label_col": "labels",  # Use 'labels' column (not 'predicted_labels')
        "num_labels": 2  # Binary: 0=non-hate, 1=hate
    }
    
    return train_df, val_df, test_df, metadata


def load_from_huggingface(dataset_name: str, dev_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load dataset directly from HuggingFace Hub.
    
    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "username/dataset_name")
        dev_ratio: Validation split ratio (default: 0.1)
    
    Returns:
        Tuple of (train_df, val_df, test_df, metadata)
    
    Raises:
        Exception: If dataset cannot be loaded from HuggingFace
    """
    from sklearn.model_selection import train_test_split
    
    print(f"  Attempting to load '{dataset_name}' from HuggingFace Hub...")
    
    try:
        # Try to load with standard splits first
        try:
            dataset = load_dataset(dataset_name)
            
            # Check available splits
            available_splits = list(dataset.keys())
            print(f"  Available splits: {available_splits}")
            
            # Try to get train/val/test splits
            train_df = None
            val_df = None
            test_df = None
            
            if "train" in available_splits:
                train_df = dataset["train"].to_pandas().dropna()
            if "validation" in available_splits or "val" in available_splits:
                val_df = dataset.get("validation", dataset.get("val", None))
                if val_df is not None:
                    val_df = val_df.to_pandas().dropna()
            if "test" in available_splits:
                test_df = dataset["test"].to_pandas().dropna()
            
            # If train exists but val/test are missing, split train into train/val/test
            if train_df is not None and (val_df is None or test_df is None):
                print(f"  ⚠️  Dataset only has 'train' split. Splitting into train/val/test using dev_ratio={dev_ratio}...")
                # Use train_df as full dataset to split
                full_df = train_df.copy()
                train_df = None  # Reset to split it
                
                # Try to detect label column for stratification
                label_col_for_split = None
                for col in ["label", "labels", "Label", "Labels", "toxicity", "Toxicity", "label_id"]:
                    if col in full_df.columns:
                        label_col_for_split = col
                        break
                
                # Split into train/val/test
                test_ratio = dev_ratio
                if label_col_for_split is not None:
                    # Use stratified split if label column found
                    train_df, temp_df = train_test_split(
                        full_df, test_size=(dev_ratio + test_ratio), random_state=42,
                        stratify=full_df[label_col_for_split]
                    )
                    val_df, test_df = train_test_split(
                        temp_df, test_size=0.5, random_state=42,
                        stratify=temp_df[label_col_for_split]
                    )
                else:
                    # Regular split without stratification
                    train_df, temp_df = train_test_split(
                        full_df, test_size=(dev_ratio + test_ratio), random_state=42
                    )
                    val_df, test_df = train_test_split(
                        temp_df, test_size=0.5, random_state=42
                    )
                print(f"  ✅ Split complete: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
            
            # If no train split at all, use the first available split and split it
            elif train_df is None and len(available_splits) > 0:
                full_df = dataset[available_splits[0]].to_pandas().dropna()
                # Try to detect label column for stratification
                label_col_for_split = None
                for col in ["label", "labels", "Label", "Labels", "toxicity", "Toxicity", "label_id"]:
                    if col in full_df.columns:
                        label_col_for_split = col
                        break
                
                # Split into train/val/test
                test_ratio = dev_ratio
                if label_col_for_split is not None:
                    # Use stratified split if label column found
                    train_df, temp_df = train_test_split(
                        full_df, test_size=(dev_ratio + test_ratio), random_state=42,
                        stratify=full_df[label_col_for_split]
                    )
                    val_df, test_df = train_test_split(
                        temp_df, test_size=0.5, random_state=42,
                        stratify=temp_df[label_col_for_split]
                    )
                else:
                    # Regular split without stratification
                    train_df, temp_df = train_test_split(
                        full_df, test_size=(dev_ratio + test_ratio), random_state=42
                    )
                    val_df, test_df = train_test_split(
                        temp_df, test_size=0.5, random_state=42
                    )
            
            if train_df is None:
                raise ValueError(f"Could not find train split in dataset {dataset_name}")
            
            # Auto-detect text and label columns
            text_col = None
            label_col = None
            
            # Common text column names
            text_candidates = ["text", "texts", "comment", "Comment", "content", "free_text", "sentence", "input"]
            # Common label column names
            label_candidates = ["label", "labels", "Label", "Labels", "toxicity", "Toxicity", "label_id", "target"]
            
            for col in text_candidates:
                if col in train_df.columns:
                    text_col = col
                    break
            
            for col in label_candidates:
                if col in train_df.columns:
                    label_col = col
                    break
            
            # Fallback: use first string column as text, last numeric column as label
            if text_col is None:
                for col in train_df.columns:
                    if train_df[col].dtype == 'object' or train_df[col].dtype == 'string':
                        text_col = col
                        break
            
            if label_col is None:
                for col in reversed(train_df.columns):
                    if train_df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        label_col = col
                        break
            
            if text_col is None or label_col is None:
                raise ValueError(
                    f"Could not auto-detect text/label columns. "
                    f"Available columns: {list(train_df.columns)}"
                )
            
            print(f"  Auto-detected columns: text='{text_col}', label='{label_col}'")
            
            # Ensure label is integer
            train_df[label_col] = train_df[label_col].astype(int)
            if val_df is not None:
                val_df[label_col] = val_df[label_col].astype(int)
            if test_df is not None:
                test_df[label_col] = test_df[label_col].astype(int)
            
            # Create metadata
            num_labels = int(pd.concat([
                train_df[label_col],
                val_df[label_col] if val_df is not None else pd.Series(),
                test_df[label_col] if test_df is not None else pd.Series()
            ]).nunique())
            
            metadata = {
                "name": dataset_name.split("/")[-1],
                "text_col": text_col,
                "label_col": label_col,
                "num_labels": num_labels
            }
            
            # Create empty DataFrames if missing
            if val_df is None:
                val_df = pd.DataFrame(columns=train_df.columns)
            if test_df is None:
                test_df = pd.DataFrame(columns=train_df.columns)
            
            return train_df, val_df, test_df, metadata
            
        except Exception as e:
            raise ValueError(f"Failed to load dataset '{dataset_name}' from HuggingFace: {str(e)}")
            
    except Exception as e:
        raise ValueError(f"Error loading from HuggingFace: {str(e)}")


def load_dataset_by_name(dataset_name: str, split_name: str = None, dev_ratio: float = 0.1, max_samples: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load dataset by name. Falls back to HuggingFace Hub if not found in predefined list.
    
    Args:
        dataset_name: One of "ViHSD", "ViCTSD", "ViHOS", "ViHSD_processed", "NCPhat2005/VOZ-HSD_2M",
                      or any HuggingFace dataset identifier (e.g., "username/dataset_name")
        split_name: For VOZ-HSD_2M: "balanced" or "hate_only" (default: "balanced")
        dev_ratio: Validation split ratio for datasets that need splitting (default: 0.1)
    
    Returns:
        Tuple of (train_df, val_df, test_df, metadata)
    
    Raises:
        ValueError: If dataset_name cannot be loaded from any source
    """
    loaders = {
        "ViHSD": load_vihsd,
        "ViCTSD": load_victsd,
        "ViHOS": load_vihos,
        "ViHSD_processed": load_vihsd_processed,
    }
    
    # Handle VOZ-HSD 2M
    if "VOZ-HSD_2M" in dataset_name or dataset_name == "NCPhat2005/VOZ-HSD_2M":
        split_to_use = split_name if split_name else "balanced"
        return load_voz_hsd_2m(split_to_use, dev_ratio)
    
    # Handle NCPhat2005/re_VOZ-HSD (different from VOZ-HSD_2M)
    if dataset_name == "NCPhat2005/re_VOZ-HSD":
        # Use load_voz_hsd_2m with split_name parameter
        split_to_use = split_name if split_name else "balanced"
        return load_voz_hsd_2m(split_to_use, dev_ratio, max_samples=max_samples)
    
    # Try predefined loaders first
    if dataset_name in loaders:
        return loaders[dataset_name]()
    
    # Fallback: Try loading from HuggingFace Hub
    if "/" in dataset_name or dataset_name.count("/") == 1:
        print(f"  Dataset '{dataset_name}' not found in predefined list, trying HuggingFace Hub...")
        return load_from_huggingface(dataset_name, dev_ratio)
    
    # If all else fails, raise error
    raise ValueError(
        f"Unknown dataset: {dataset_name}. "
        f"Available predefined datasets: {list(loaders.keys()) + ['NCPhat2005/VOZ-HSD_2M']}. "
        f"Or use HuggingFace dataset identifier (e.g., 'username/dataset_name')"
    )


def build_torch_dataset(df: pd.DataFrame, text_col: str, label_col: str, 
                        tokenizer, max_length: int) -> TextDataset:
    """
    Build PyTorch Dataset from DataFrame.
    
    Args:
        df: Input DataFrame
        text_col: Name of text column
        label_col: Name of label column
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
    
    Returns:
        TextDataset instance
    """
    use_word_seg = "phobert" in str(getattr(tokenizer, "name_or_path", "")).lower()

    return TextDataset(
        df[text_col].tolist(),
        df[label_col].tolist(),
        tokenizer,
        max_length,
        use_word_seg=use_word_seg,
    )
