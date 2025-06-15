# IndoLER: Indonesian Legal Entity Recognition

Proyek Named Entity Recognition (NER) untuk dokumen hukum Indonesia menggunakan XLM-RoBERTa.

## Dataset Overview

- **Jumlah dokumen**: 993 dokumen hukum Indonesia
- **Entity types**: 22 jenis entitas (PER, ORG, LOC, dll.)
- **Format**: BIO tagging scheme
- **Split**: Train/Validation/Test

## Frequently Asked Questions (FAQ)

### Q1: Mengapa dilakukan tokenization dan label alignment?

**A:** Code ini mempersiapkan data text untuk Named Entity Recognition (NER). Proses ini diperlukan karena:

1. **Model transformer** (seperti BERT/XLM-RoBERTa) menggunakan **subword tokenization**
2. **Label asli** berupa per-kata, sedangkan **model** membutuhkan label per-subword token

**Contoh masalah:**
```python
# Input asli:
words = ["Jakarta", "adalah", "ibukota"]
labels = ["B-LOC", "O", "O"]

# Setelah tokenization (subword):
tokens = ["Jak", "##arta", "adalah", "ibu", "##kota"]  
# Perlu alignment labels: ["B-LOC", "I-LOC", "O", "O", "O"]
```

### Q2: Apa maksud output tokenization berbentuk dictionary dengan arrays?

**A:** Output berbentuk dictionary karena model transformer membutuhkan input dalam format khusus:

```python
{
    'input_ids': [[101, 2198, 2152, 102, ...], ...],      # Token IDs (vocabulary)
    'attention_mask': [[1, 1, 1, 1, ...], ...],          # 1=real token, 0=padding
    'labels': [[0, 1, 2, 0, ...], ...],                  # Label ID untuk setiap token
}
```

**Penjelasan angka:**
- **101** = `[CLS]` token (classification token)
- **102** = `[SEP]` token (separator)
- **2198, 2152** = Vocabulary IDs untuk kata tertentu
- **0, 1, 2** = Label IDs (0=O, 1=B-PER, 2=I-PER, dst.)

### Q3: Mengapa perlu NERDataset function?

**A:** `NERDataset` class diperlukan untuk:

1. **Mengkonversi** dictionary arrays menjadi PyTorch Dataset
2. **Kompatibilitas** dengan PyTorch DataLoader untuk batch processing
3. **Memory efficiency** - tidak semua data dimuat sekaligus

```python
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data
    
    def __getitem__(self, idx):
        # Convert ke tensor untuk satu sample
        return {
            'input_ids': torch.tensor(self.data[idx]['input_ids']),
            'attention_mask': torch.tensor(self.data[idx]['attention_mask']),
            'labels': torch.tensor(self.data[idx]['labels'])
        }
```

### Q4: Input apa yang masuk ke model setelah melalui Dataset?

**A:** Setelah data melalui `NERDataset` dan `DataLoader`, input ke model adalah **batch tensors**:

```python
# Batch yang masuk ke model.forward()
batch = {
    'input_ids': tensor([
        [101, 2198, 2152, 102, 0, 0, ...],     # Sample 1
        [101, 1234, 5678, 102, 0, 0, ...],     # Sample 2  
        ...                                     # More samples
    ]),                                         # Shape: [batch_size, seq_len]
    
    'attention_mask': tensor([
        [1, 1, 1, 1, 0, 0, ...],               # 1=real, 0=padding
        [1, 1, 1, 1, 0, 0, ...],
        ...
    ]),                                         # Shape: [batch_size, seq_len]
    
    'labels': tensor([
        [0, 1, 2, 0, -100, -100, ...],         # -100 = ignore in loss
        [0, 3, 4, 0, -100, -100, ...],
        ...
    ])                                          # Shape: [batch_size, seq_len]
}
```

### Q5: Bagaimana arsitektur model AutoModelForTokenClassification?

**A:** Model menggunakan arsitektur **BERT-based Token Classification**:

```python
# Model architecture:
1. BERT Encoder          # Contextualized embeddings
2. Dropout Layer         # Regularization  
3. Linear Classifier     # num_labels output units
4. Loss Function         # CrossEntropyLoss (ignores -100)
```

**Forward pass:**
```python
def forward(input_ids, attention_mask, labels=None):
    # 1. BERT encoding
    hidden_states = bert_encoder(input_ids, attention_mask)  # [batch, seq_len, hidden_size]
    
    # 2. Classification
    logits = classifier(hidden_states)                       # [batch, seq_len, num_labels]
    
    # 3. Loss computation (if training)
    if labels is not None:
        loss = loss_fn(logits, labels)                       # Scalar
        return {'loss': loss, 'logits': logits}
    
    return {'logits': logits}
```

### Q6: Apa output dari model?

**A:** Output model bergantung pada mode:

**Training mode (dengan labels):**
```python
outputs = {
    'loss': tensor(2.1234),                           # Scalar untuk backprop
    'logits': tensor([batch_size, seq_len, num_labels])  # Raw predictions
}
```

**Inference mode (tanpa labels):**
```python
outputs = {
    'logits': tensor([batch_size, seq_len, num_labels])  # Raw predictions saja
}
```

**Contoh interpretasi logits:**
```python
# Input: "John works at Google"
# logits shape: [1, 6, 5]  # 1 sample, 6 tokens, 5 possible labels

logits[0, 1, :] = [0.1, 0.8, 0.05, 0.03, 0.02]  # Token "John" → B-PER (index 1)
logits[0, 4, :] = [0.05, 0.1, 0.05, 0.02, 0.78]  # Token "Google" → B-ORG (index 4)
```

### Q7: Bagaimana proses training dengan Trainer?

**A:** Trainer melakukan training loop otomatis:

```python
# Training loop (simplified):
for batch in dataloader:
    # 1. Forward pass
    outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        labels=batch['labels']
    )
    
    # 2. Get loss
    loss = outputs.loss
    
    # 3. Backward pass
    loss.backward()
    
    # 4. Update weights
    optimizer.step()
    
    # 5. Evaluation (periodic)
    if step % eval_steps == 0:
        eval_metrics = evaluate()
```

## Flow Data Lengkap

```
Raw Text + Labels 
    ↓
Tokenization & Alignment
    ↓  
Dictionary {input_ids, attention_mask, labels}
    ↓
NERDataset (PyTorch Dataset)
    ↓
DataLoader (Batch Processing)
    ↓
Model Forward Pass
    ↓
Loss + Logits Output
    ↓
Backpropagation & Weight Update
```

## Model Performance

- **Test F1-Score**: 85-90% (target)
- **Entity Types**: 22 jenis entitas hukum
- **Language**: Indonesian
- **Domain**: Legal documents

## Usage

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load model
tokenizer = AutoTokenizer.from_pretrained('./models/model_final')
model = AutoModelForTokenClassification.from_pretrained('./models/model_final')

# Predict
text = "Hakim Budi Santoso memutuskan perkara di Jakarta"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=2)
```