# Load preprocessed data and setup XLM-RoBERTa model
import os
import pickle
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report as seq_classification_report

print("NER Model")
print("=" * 50)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Load preprocessed data
preprocessed_dir = './results'

print("\nLoading preprocessed data...")

# Load tokenized datasets
with open(os.path.join(preprocessed_dir, 'train_tokenized.pkl'), 'rb') as f:
    train_dataset = pickle.load(f)
    
with open(os.path.join(preprocessed_dir, 'val_tokenized.pkl'), 'rb') as f:
    val_dataset = pickle.load(f)
    
with open(os.path.join(preprocessed_dir, 'test_tokenized.pkl'), 'rb') as f:
    test_dataset = pickle.load(f)

# Load metadata and label mappings
with open(os.path.join(preprocessed_dir, 'preprocessing_metadata.json'), 'r') as f:
    metadata = json.load(f)
    
with open(os.path.join(preprocessed_dir, 'label2id.json'), 'r') as f:
    label2id = json.load(f)
    
with open(os.path.join(preprocessed_dir, 'id2label.json'), 'r') as f:
    id2label = {int(k): v for k, v in json.load(f).items()}

print(f"✓ Loaded {len(train_dataset)} training samples")
print(f"✓ Loaded {len(val_dataset)} validation samples")
print(f"✓ Loaded {len(test_dataset)} test samples")
print(f"✓ Number of labels: {len(label2id)}")
print(f"✓ Model name: {metadata['model_name']}")
print(f"✓ Max length: {metadata['max_length']}")

# Custom Dataset class for NER
class NERDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long)
        }

# Create dataset objects
train_torch_dataset = NERDataset(train_dataset)
val_torch_dataset = NERDataset(val_dataset)
test_torch_dataset = NERDataset(test_dataset)

print(f"\n✓ Created PyTorch datasets")
print(f"  Training: {len(train_torch_dataset)} samples")
print(f"  Validation: {len(val_torch_dataset)} samples")
print(f"  Test: {len(test_torch_dataset)} samples")

# Load model and tokenizer
model_name = metadata['model_name']  # 'xlm-roberta-base'
print(f"\nLoading {model_name} for token classification...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"✓ Tokenizer loaded - Vocab size: {len(tokenizer)}")

# Load model for token classification
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# Move model to device
model.to(device)
print(f"✓ Model loaded and moved to {device}")
print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"  Number of labels: {model.num_labels}")

# Data collator for token classification
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt"
)
# Metrics computation function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Get predictions (argmax of logits)
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens)
    true_predictions = []
    true_labels = []
    
    for prediction, label in zip(predictions, labels):
        # Only include tokens where label != -100
        valid_indices = label != -100
        
        pred_labels = [id2label[p] for p, valid in zip(prediction, valid_indices) if valid]
        true_label_list = [id2label[l] for l, valid in zip(label, valid_indices) if valid]
        
        if len(pred_labels) > 0 and len(true_label_list) > 0:
            true_predictions.append(pred_labels)
            true_labels.append(true_label_list)
    
    # Compute seqeval metrics
    try:
        precision = precision_score(true_labels, true_predictions)
        recall = recall_score(true_labels, true_predictions)
        f1 = f1_score(true_labels, true_predictions)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

# Training arguments
output_dir = './models/xlm_roberta_ner_results'
logging_dir = './models/cache/xlm_roberta_ner_logs'

training_args = TrainingArguments(
    output_dir=output_dir,
    
    # Training hyperparameters
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    
    # Logging and evaluation
    logging_dir=logging_dir,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    
    # Other settings
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to=None,  # Disable wandb
    
    # Performance
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    
    # For debugging
    max_steps=100,  # Uncomment for quick testing
)

print(f"\n✓ Training arguments configured:")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Train batch size: {training_args.per_device_train_batch_size}")
print(f"  Eval batch size: {training_args.per_device_eval_batch_size}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Output directory: {output_dir}")

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_torch_dataset,
    eval_dataset=val_torch_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print(f"\n✓ Trainer created successfully")
print(f"\nTraining Setup Summary:")
print(f"{'='*50}")
print(f"Model: {model_name}")
print(f"Total labels: {len(label2id)}")
print(f"Training samples: {len(train_torch_dataset):,}")
print(f"Validation samples: {len(val_torch_dataset):,}")
print(f"Test samples: {len(test_torch_dataset):,}")
print(f"Device: {device}")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Batch size (train): {training_args.per_device_train_batch_size}")
print(f"Learning rate: {training_args.learning_rate}")
print(f"{'='*50}")
print(f"\nReady to start training!")

# Clean up memory before training
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"\nGPU Memory after cleanup: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

print("\nMemory cleaned up - ready to start training!")
print("Run the next cell to start training...")

# Load preprocessed data
preprocessed_dir = './results'

print("\nLoading preprocessed data...")

# Load tokenized datasets
with open(os.path.join(preprocessed_dir, 'train_tokenized.pkl'), 'rb') as f:
    train_dataset = pickle.load(f)
    
with open(os.path.join(preprocessed_dir, 'val_tokenized.pkl'), 'rb') as f:
    val_dataset = pickle.load(f)
    
with open(os.path.join(preprocessed_dir, 'test_tokenized.pkl'), 'rb') as f:
    test_dataset = pickle.load(f)

# Load metadata and label mappings
with open(os.path.join(preprocessed_dir, 'preprocessing_metadata.json'), 'r') as f:
    metadata = json.load(f)
    
with open(os.path.join(preprocessed_dir, 'label2id.json'), 'r') as f:
    label2id = json.load(f)
    
with open(os.path.join(preprocessed_dir, 'id2label.json'), 'r') as f:
    id2label = {int(k): v for k, v in json.load(f).items()}

print(f"✓ Loaded {len(train_dataset)} training samples")
print(f"✓ Loaded {len(val_dataset)} validation samples")
print(f"✓ Loaded {len(test_dataset)} test samples")
print(f"✓ Number of labels: {len(label2id)}")
print(f"✓ Model name: {metadata['model_name']}")
print(f"✓ Max length: {metadata['max_length']}")

# Custom Dataset class for NER
class NERDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long)
        }

# Create dataset objects
train_torch_dataset = NERDataset(train_dataset)
val_torch_dataset = NERDataset(val_dataset)
test_torch_dataset = NERDataset(test_dataset)

print(f"\n✓ Created PyTorch datasets")
print(f"  Training: {len(train_torch_dataset)} samples")
print(f"  Validation: {len(val_torch_dataset)} samples")
print(f"  Test: {len(test_torch_dataset)} samples")


# Load model and tokenizer
model_name = metadata['model_name']  # 'xlm-roberta-base'
print(f"\nLoading {model_name} for token classification...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"✓ Tokenizer loaded - Vocab size: {len(tokenizer)}")

# Load model for token classification
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# Move model to device
model.to(device)
print(f"✓ Model loaded and moved to {device}")
print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"  Number of labels: {model.num_labels}")

# Start training
print("Starting Model Training...")
print("=" * 60)

import time
start_time = time.time()

try:
    # Train the model
    train_result = trainer.train()
    
    # Training completed
    end_time = time.time()
    training_duration = end_time - start_time
    
    print(f"\n✓ Training completed successfully!")
    print(f"Training duration: {training_duration/60:.2f} minutes")
    print(f"Final training loss: {train_result.training_loss:.4f}")
    
    # Save the model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✓ Model and tokenizer saved to: {output_dir}")
    
except Exception as e:
    print(f"\n❌ Training failed with error: {e}")
    import traceback
    traceback.print_exc()
    

# Evaluate on validation set
print("\nEvaluating on validation set...")
print("=" * 40)

try:
    eval_results = trainer.evaluate()
    
    print(f"\n✓ Validation Results:")
    print(f"  Validation Loss: {eval_results['eval_loss']:.4f}")
    print(f"  Precision: {eval_results['eval_precision']:.4f}")
    print(f"  Recall: {eval_results['eval_recall']:.4f}")
    print(f"  F1-Score: {eval_results['eval_f1']:.4f}")
    
except Exception as e:
    print(f"\n❌ Validation failed with error: {e}")
    import traceback
    traceback.print_exc()


# Detailed evaluation on test set
print("\nEvaluating on test set...")
print("=" * 40)

try:
    # Get predictions on test set
    test_predictions = trainer.predict(test_torch_dataset)
    
    # Process predictions
    predictions = np.argmax(test_predictions.predictions, axis=2)
    labels = test_predictions.label_ids
    
    # Convert to label names (excluding special tokens)
    true_predictions = []
    true_labels = []
    
    for prediction, label in zip(predictions, labels):
        valid_indices = label != -100
        
        pred_labels = [id2label[p] for p, valid in zip(prediction, valid_indices) if valid]
        true_label_list = [id2label[l] for l, valid in zip(label, valid_indices) if valid]
        
        if len(pred_labels) > 0 and len(true_label_list) > 0:
            true_predictions.append(pred_labels)
            true_labels.append(true_label_list)
    
    # Compute detailed metrics
    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)
    
    print(f"\n✓ Test Set Results:")
    print(f"  Test Loss: {test_predictions.metrics['test_loss']:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print("=" * 60)
    report = seq_classification_report(true_labels, true_predictions, digits=4)
    print(report)
    
except Exception as e:
    print(f"\n❌ Test evaluation failed with error: {e}")
    import traceback
    traceback.print_exc()