"""
T5 multi-task Seq2Seq fine-tuning for Vietnamese hate speech detection.
"""

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from datasets import Dataset
import pandas as pd
import numpy as np
import argparse
import nltk
import os
import torch
import unicodedata
import ast
from tqdm import tqdm
from pathlib import Path

# Download required NLTK packages
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from data_loader import load_dataset_by_name
from focal_loss import FocalLossSeq2SeqTrainer
from augment import augment_minority_classes

# Set tokenizer parallelism to false
os.environ["TOKENIZERS_PARALLELISM"] = 'False'

# Create argument parser
parser = argparse.ArgumentParser(description='Fine-tune T5 model')
parser.add_argument('--save_model_name', type=str, help='Name of fine-tuned model')
parser.add_argument('--pre_trained_ckpt', type=str, help='Path to pre-trained checkpoint on HuggingFace')
parser.add_argument('--output_dir', type=str, help='Path to output directory for saving fine-tuned model')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=4, help='Number of epochs')
parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
parser.add_argument('--warmup_ratio', type=float, default=0.0, help='Warmup ratio')
parser.add_argument('--lr_scheduler_type', type=str, default='constant', help='Learning rate scheduler type')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU number')
parser.add_argument('--use_focal_loss', action='store_true', help='Use focal loss instead of standard CrossEntropy')
parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma parameter (focusing strength)')
parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing factor (0.0 = no smoothing)')
parser.add_argument('--augment_minority', action='store_true', help='Augment minority classes (HATE, OFFENSIVE) using EDA techniques')
parser.add_argument('--augment_factor', type=float, default=0.8, help='Target ratio for augmentation (0.0-1.0, where 1.0 = fully balanced with majority class)')

# Parse arguments
bash_args = parser.parse_args()

# Validate focal loss arguments
if not bash_args.use_focal_loss and (bash_args.focal_gamma != 2.0 or bash_args.label_smoothing != 0.0):
    parser.error("--focal_gamma and --label_smoothing require --use_focal_loss flag")
if bash_args.focal_gamma < 0:
    parser.error("--focal_gamma must be >= 0")
if bash_args.label_smoothing < 0.0 or bash_args.label_smoothing > 1.0:
    parser.error("--label_smoothing must be between 0.0 and 1.0")
if bash_args.augment_factor < 0.0 or bash_args.augment_factor > 1.0:
    parser.error("--augment_factor must be between 0.0 and 1.0")

# Define GPU
os.environ["CUDA_VISIBLE_DEVICES"]=bash_args.gpu


def map_data_vozhsd(set_df):
    map_labels = {
        0: "NONE",
        1: "HATE",
    }

    set_df["source"] = set_df["texts"].apply(lambda x: "hate-speech-detection: " + x)
    set_df["target"] = set_df["labels"].apply(lambda x: map_labels[x])

def map_data_vihsd(set_df):
  map_labels = {
      0: "CLEAN",
      1: "OFFENSIVE",
      2: "HATE",
  }

  set_df["source"] = set_df["free_text"].apply(lambda x: "hate-speech-detection: " + x)
  set_df["target"] = set_df["label_id"].apply(lambda x: map_labels[x])

  set_df = set_df[["source", "target"]]

  return set_df

def map_data_victsd(set_df):
    map_labels = {
        0: "NONE",
        1: "TOXIC",
    }

    set_df["source"] = set_df["Comment"].apply(lambda x: "toxic-speech-detection: " + x)
    set_df["target"] = set_df["Toxicity"].apply(lambda x: map_labels[x])

    set_df = set_df[["source", "target"]]

    return set_df

def process_spans(lst):

    lst = [int(x) for x in lst.strip("[]'").split(', ')]
    result = []
    temp = [lst[0]]

    for i in range(1, len(lst)):
        if lst[i] == lst[i-1] + 1:
            temp.append(lst[i])
        else:
            result.append(temp)
            temp = [lst[i]]

    result.append(temp)
    return result

def add_tags(text, indices):

  if indices == '[]':
    return text

  indices = process_spans(indices)


  for i in range(0, len(indices)):

    # Insert "[HATE]" at index X
    text = text[:indices[i][0]] + "[HATE]" + text[indices[i][0]:]

    # Insert "[\HATE]" at index Y
    text = text[:indices[i][-1]+7] + "[HATE]" + text[indices[i][-1]+7:]

    for j in range(i + 1, len(indices)):
      indices[j] = [x + 12 for x in indices[j]]

  return text

def map_data_vihos(set_df):
    set_df["source"] = set_df["content"].apply(lambda x: "hate-spans-detection: " + x)

    target = []

    for i in range(0, len(set_df['content'])):
        target.append(add_tags(set_df['content'][i], set_df['index_spans'][i]))

    set_df["target"] = target
    set_df = set_df[["source", "target"]]

    return set_df

# Evaluation functions (from evaluate.py)
def generate_output_batch(test_df, model, tokenizer, batch_size=32):
    """Generate outputs for test dataframe (optimized with batch processing)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    output_texts = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_df), batch_size), desc="Generating outputs"):
            batch_sources = test_df['source'].iloc[i:i+batch_size].tolist()
            
            # Tokenize batch
            enc = tokenizer(
                batch_sources,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=bash_args.max_length
            ).to(device)
            
            # Generate outputs for batch (following author's approach: max_length=256)
            output_ids = model.generate(**enc, max_length=64) # Labels are short
            
            # Decode batch
            decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            output_texts.extend(decoded)

    # Add the generated outputs to the DataFrame
    test_df['output'] = output_texts

    return test_df


def find_and_extract_substrings(original_str, input_str):
    """Extract character indices from generated text with [HATE] ... [HATE] tags (following author's approach)."""
    start_tag = '[hate]'
    end_tag = '[hate]'

    input_str = unicodedata.normalize('NFC', input_str.lower())
    original_str = unicodedata.normalize('NFC', original_str.lower())

    # Extract substrings
    substrings = []
    start_index = input_str.find(start_tag)
    while start_index != -1:
        end_index = input_str.find(end_tag, start_index + len(start_tag))
        if end_index != -1:
            substrings.append(input_str[start_index + len(start_tag):end_index])
            start_index = input_str.find(start_tag, end_index + len(end_tag))
        else:
            break

    if not substrings:
        return '[]'

    # Find indices in the original string and merge into one list
    indices_list = []
    for substring in substrings:
        start_index = original_str.find(substring)
        while start_index != -1:
            indices_list.extend(list(range(start_index, start_index + len(substring))))
            start_index = original_str.find(substring, start_index + 1)

    deduplicated_sorted_indices_list = sorted(set(indices_list))

    return str(deduplicated_sorted_indices_list)


def process_output_spans(lst_output, text_input):
    """Process output spans for ViHOS (following author's approach)."""
    processed_lst = []
    for i in range(len(lst_output)):
        processed_lst.append(find_and_extract_substrings(text_input[i], lst_output[i]))
    return processed_lst


def digitize_spans(vihos_test_df, vihos_results):
    """Convert predicted span strings to binary vectors (following author's approach)."""
    # Convert string representations of lists to actual lists using ast.literal_eval
    vihos_preds = [ast.literal_eval(x) for x in vihos_results['output_spans']]
    vihos_labels = [ast.literal_eval(x) for x in vihos_test_df['index_spans']]

    # Calculate the lengths of the content in vihos_test_df
    vihos_lengths = [len(x) for x in vihos_test_df['content']]

    preds = []
    labels = []

    for i in range(len(vihos_lengths)):
        pred = []
        label = []
        for idx in range(vihos_lengths[i]):
            if idx in vihos_preds[i]:
                pred.append(1)
            else:
                pred.append(0)

            if idx in vihos_labels[i]:
                label.append(1)
            else:
                label.append(0)

        preds.append(pred)
        labels.append(label)

    return labels, preds

print('PROGRESS|Loading datasets...')
# Load datasets using data_loader
vihsd_train_df, vihsd_val_df, vihsd_test_df, _ = load_dataset_by_name('ViHSD')
victsd_train_df, victsd_val_df, victsd_test_df, _ = load_dataset_by_name('ViCTSD')
vihos_train_df, vihos_val_df, vihos_test_df, _ = load_dataset_by_name('ViHOS')

# Apply data augmentation to minority classes if requested
if bash_args.augment_minority:
    print(f'PROGRESS|Augmenting minority classes (target_ratio={bash_args.augment_factor})...')
    vihsd_train_df = augment_minority_classes(
        vihsd_train_df, text_col='free_text', label_col='label_id',
        target_ratio=bash_args.augment_factor, seed=bash_args.seed
    )
    victsd_train_df = augment_minority_classes(
        victsd_train_df, text_col='Comment', label_col='Toxicity',
        target_ratio=bash_args.augment_factor, seed=bash_args.seed
    )
    # ViHOS is NOT augmented — span annotations (index_spans) would be
    # invalidated by text transformations. Only classification tasks are augmented.
    print(f'PROGRESS|Augmentation complete. ViHOS skipped (span-based annotations).')

# Keep original test datasets for evaluation (before mapping)
vihsd_test_df_orig = vihsd_test_df.copy()
victsd_test_df_orig = victsd_test_df.copy()
vihos_test_df_orig = vihos_test_df.copy()

# Map datasets to source-target format
vihsd_train_df = map_data_vihsd(vihsd_train_df)
vihsd_val_df = map_data_vihsd(vihsd_val_df)
vihsd_test_df = map_data_vihsd(vihsd_test_df)

victsd_train_df = map_data_victsd(victsd_train_df)
victsd_val_df = map_data_victsd(victsd_val_df)
victsd_test_df = map_data_victsd(victsd_test_df)

vihos_train_df = map_data_vihos(vihos_train_df)
vihos_val_df = map_data_vihos(vihos_val_df)
vihos_test_df = map_data_vihos(vihos_test_df)

print('PROGRESS|Concatenating datasets...')
list_of_train_dfs = [vihsd_train_df, victsd_train_df, vihos_train_df]
list_of_val_dfs = [vihsd_val_df, victsd_val_df, vihos_val_df]
list_of_test_dfs = [vihsd_test_df, victsd_test_df, vihos_test_df]

final_train_df = pd.concat(list_of_train_dfs, axis=0)
final_val_df = pd.concat(list_of_val_dfs, axis=0)
final_test_df = pd.concat(list_of_test_dfs, axis=0)

train_data = Dataset.from_pandas(final_train_df)
val_data = Dataset.from_pandas(final_val_df)
test_data = Dataset.from_pandas(final_test_df)

print('PROGRESS|Tokenizing datasets...')
model_id=bash_args.pre_trained_ckpt
tokenizer = AutoTokenizer.from_pretrained(model_id)

if "ViHateT5" in model_id:
    model = T5ForConditionalGeneration.from_pretrained(model_id, from_flax=True) # Because ViHateT5 is a Flax model
else:
    model = T5ForConditionalGeneration.from_pretrained(model_id)

def preprocess_function(sample,padding="max_length"):
    model_inputs = tokenizer(sample["source"], max_length=bash_args.max_length, padding=padding, truncation=True)
    labels = tokenizer(sample["target"], max_length=64, padding=padding, truncation=True) # Labels are short for classification
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print('PROGRESS|Tokenizing train, val, test datasets...')
train_tokenized_dataset = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names)
val_tokenized_dataset = val_data.map(preprocess_function, batched=True, remove_columns=val_data.column_names)
test_tokenized_dataset = test_data.map(preprocess_function, batched=True, remove_columns=test_data.column_names)
print(f"Keys of tokenized dataset: {list(train_tokenized_dataset.features)}")

output_dir=bash_args.output_dir
training_args = Seq2SeqTrainingArguments(
    overwrite_output_dir=True,
    output_dir=output_dir,
    per_device_train_batch_size=bash_args.batch_size,
    per_device_eval_batch_size=bash_args.batch_size,
    gradient_accumulation_steps=bash_args.gradient_accumulation_steps,
    learning_rate=bash_args.learning_rate,
    weight_decay=bash_args.weight_decay,
    num_train_epochs=bash_args.num_epochs,
    warmup_ratio=bash_args.warmup_ratio,
    lr_scheduler_type=bash_args.lr_scheduler_type,
    seed=bash_args.seed,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="epoch",
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to="none",  # Disable wandb
    load_best_model_at_end=True,
    save_total_limit=1,
    do_train=True,
    do_eval=True,
    predict_with_generate=True,
)

model.config.use_cache = False

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Safety: handle tuple/3D array (following VIHOS_T5.py approach)
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    predictions = np.array(predictions)
    
    # If 3D (logits), take argmax
    if predictions.ndim == 3:
        predictions = predictions.argmax(axis=-1)
    
    # Clip to valid vocab range to avoid overflow errors
    vocab_size = len(tokenizer)
    predictions = np.clip(predictions, 0, vocab_size - 1).astype(np.int64)
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Clip labels to valid vocab range
    labels = np.clip(labels, 0, vocab_size - 1).astype(np.int64)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Calculate F1 macro
    f1_macro = f1_score(decoded_labels, decoded_preds, average='macro', zero_division=0)

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    gen_len_mean = np.mean(prediction_lens)

    return {'f1_macro': round(f1_macro * 100, 4), 'gen_len': round(gen_len_mean, 4)}

# Create trainer with conditional focal loss support
if bash_args.use_focal_loss:
    print(f'PROGRESS|Using Focal Loss (gamma={bash_args.focal_gamma}, smoothing={bash_args.label_smoothing})')
    trainer = FocalLossSeq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_tokenized_dataset,
        eval_dataset=val_tokenized_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        focal_gamma=bash_args.focal_gamma,
        label_smoothing=bash_args.label_smoothing,
    )
else:
    print('PROGRESS|Using standard CrossEntropy loss')
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_tokenized_dataset,
        eval_dataset=val_tokenized_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

print('PROGRESS|Training...')
trainer.train()

print('PROGRESS|Push to hub...')
trainer.push_to_hub()

# Evaluation on test set (following evaluate.py approach)
print('\nPROGRESS|Evaluating on test set...')

# Reload test datasets and map them
vihsd_test_df_eval = map_data_vihsd(vihsd_test_df_orig.copy())
victsd_test_df_eval = map_data_victsd(victsd_test_df_orig.copy())
vihos_test_df_eval = map_data_vihos(vihos_test_df_orig.copy())

# Keep original vihos_test_df for span extraction
vihos_test_df_orig_eval = vihos_test_df_orig.copy()

# Get the trained model from trainer
trained_model = trainer.model

# Inferencing (following author's approach)
print('\nPROGRESS|Generating predictions...')
vihsd_results = generate_output_batch(vihsd_test_df_eval.copy(), trained_model, tokenizer, batch_size=bash_args.batch_size)
victsd_results = generate_output_batch(victsd_test_df_eval.copy(), trained_model, tokenizer, batch_size=bash_args.batch_size)
vihos_results = generate_output_batch(vihos_test_df_eval.copy(), trained_model, tokenizer, batch_size=bash_args.batch_size)

# Process ViHOS spans
vihos_results['output_spans'] = process_output_spans(vihos_results['output'], vihos_test_df_orig_eval['content'])

# ViHSD: Hate Speech Detection
# Turn outputs into labels, 'CLEAN', 'OFFENSIVE', 'HATE' to 0, 1, 2
vihsd_results['output'] = vihsd_results['output'].apply(
    lambda x: 0 if x == 'CLEAN' or x == 'clean' else (1 if x == 'OFFENSIVE' or x == 'offensive' else 2)
)
vihsd_test_df_eval['target'] = vihsd_test_df_eval['target'].apply(
    lambda x: 0 if x == 'CLEAN' or x == 'clean' else (1 if x == 'OFFENSIVE' or x == 'offensive' else 2)
)

# Compute accuracy, weighted f1 score, and macro f1 score
vihsd_accuracy = accuracy_score(vihsd_results['output'], vihsd_test_df_eval['target'])
vihsd_f1_weighted = f1_score(vihsd_results['output'], vihsd_test_df_eval['target'], average='weighted')
vihsd_f1_macro = f1_score(vihsd_results['output'], vihsd_test_df_eval['target'], average='macro')

# ViCTSD: Toxic Speech Detection
# Turn outputs into labels, 'NONE', 'TOXIC' to 0, 1
victsd_results['output'] = victsd_results['output'].apply(lambda x: 0 if x == 'NONE' or x == 'none' else 1)
victsd_test_df_eval['target'] = victsd_test_df_eval['target'].apply(lambda x: 0 if x == 'NONE' or x == 'none' else 1)

# Compute accuracy, weighted f1 score, and macro f1 score
victsd_accuracy = accuracy_score(victsd_results['output'], victsd_test_df_eval['target'])
victsd_f1_weighted = f1_score(victsd_results['output'], victsd_test_df_eval['target'], average='weighted')
victsd_f1_macro = f1_score(victsd_results['output'], victsd_test_df_eval['target'], average='macro')

# ViHOS: Hate Spans Detection
vihos_labels_digits, vihos_preds_digits = digitize_spans(vihos_test_df_orig_eval, vihos_results)

# Check for length mismatches
mismatches = []
for i in range(len(vihos_labels_digits)):
    if len(vihos_labels_digits[i]) != len(vihos_preds_digits[i]):
        mismatches.append(i)

if mismatches:
    print(f"  ⚠️  Found {len(mismatches)} length mismatches in ViHOS predictions")

# Initialize lists for storing accuracy, weighted f1 score, and macro f1 score
vihos_accuracy = []
vihos_f1_weighted = []
vihos_f1_macro = []

# Compute accuracy, weighted f1 score, and macro f1 score for vihos_labels_digits and vihos_preds_digits
for i in range(len(vihos_labels_digits)):
    preds_temp = list(vihos_preds_digits[i])
    labels_temp = list(vihos_labels_digits[i])
    
    # Skip if lengths don't match
    if len(preds_temp) != len(labels_temp):
        continue

    vihos_accuracy.append(accuracy_score(labels_temp, preds_temp))
    vihos_f1_weighted.append(f1_score(labels_temp, preds_temp, average='weighted', zero_division=0))
    vihos_f1_macro.append(f1_score(labels_temp, preds_temp, average='macro', zero_division=0))

# Compute the average accuracy, weighted f1 score, and macro f1 score
vihos_accuracy = np.mean(vihos_accuracy) if vihos_accuracy else 0.0
vihos_f1_weighted = np.mean(vihos_f1_weighted) if vihos_f1_weighted else 0.0
vihos_f1_macro = np.mean(vihos_f1_macro) if vihos_f1_macro else 0.0

# Print the results in tabular format
print('\n' + "=" * 80)
print('Test Set Results:')
print("=" * 80)
print('ViHSD:')
print(f'  Accuracy: {vihsd_accuracy:.4f}')
print(f'  Weighted F1 Score: {vihsd_f1_weighted:.4f}')
print(f'  Macro F1 Score: {vihsd_f1_macro:.4f}')
print('ViCTSD:')
print(f'  Accuracy: {victsd_accuracy:.4f}')
print(f'  Weighted F1 Score: {victsd_f1_weighted:.4f}')
print(f'  Macro F1 Score: {victsd_f1_macro:.4f}')
print('ViHOS:')
print(f'  Accuracy: {vihos_accuracy:.4f}')
print(f'  Weighted F1 Score: {vihos_f1_weighted:.4f}')
print(f'  Macro F1 Score: {vihos_f1_macro:.4f}')
print("=" * 80)

# Save the results to a CSV file
result_filepath = 'results/evaluation_results.csv'
results = {
    'Model': bash_args.save_model_name if bash_args.save_model_name else bash_args.pre_trained_ckpt,
    'Task': ['ViHSD', 'ViCTSD', 'ViHOS'],
    'Accuracy': [round(vihsd_accuracy, 4), round(victsd_accuracy, 4), round(vihos_accuracy, 4)],
    'Weighted F1 Score': [round(vihsd_f1_weighted, 4), round(victsd_f1_weighted, 4), round(vihos_f1_weighted, 4)],
    'Macro F1 Score': [round(vihsd_f1_macro, 4), round(victsd_f1_macro, 4), round(vihos_f1_macro, 4)]
}

results_df = pd.DataFrame(results)

# Create output directory if needed
output_file = Path(result_filepath)
output_file.parent.mkdir(parents=True, exist_ok=True)

# Append the results to the existing results.csv file
results_df.to_csv(output_file, mode='a', header=not output_file.exists(), index=False)
print(f'\n💾 Saved results to {output_file}')

print('PROGRESS|Done!')