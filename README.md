# Indonesian Legal Documents Named Entity Recognition (IndoLER)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.0%2B-yellow)](https://huggingface.co/transformers)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

**IndoLER** (Indonesian Legal Entity Recognition) is a comprehensive Named Entity Recognition (NER) system specifically designed for Indonesian legal documents. This project implements a state-of-the-art NER model using XLM-RoBERTa Large architecture, trained on 993 annotated Indonesian Supreme Court decision documents.

The system is capable of identifying and classifying various legal entities within Indonesian court documents, including person names, legal institutions, case numbers, laws, and other domain-specific entities critical for legal document analysis.

## 📊 Dataset

### Dataset Characteristics
- **Name**: IndoLER Dataset
- **Size**: 993 annotated court decision documents
- **Source**: Indonesian Supreme Court (Mahkamah Agung Indonesia)
- **Language**: Indonesian (Bahasa Indonesia)
- **Annotation Format**: IOB (Inside-Outside-Begin) tagging scheme
- **Domain**: Legal documents (criminal court decisions)

### Dataset Composition
- **Training Set**: 80% of documents
- **Validation Set**: 10% of documents  
- **Test Set**: 10% of documents

### Document Metadata
Each document contains the following metadata:
- **Verdict Types**: 
  - `guilty` (bersalah)
  - `bebas` (acquitted)
  - `lepas` (discharged)
- **Indictment Types**:
  - `tunggal` (single)
  - `subsider` (subsidiary)
  - `alternatif` (alternative)
  - `kumulatif` (cumulative)
- **Lawyer Presence**: Boolean indicator
- **Annotator Information**: Multiple annotators (agree, dhipa, fariz, jafar)

### Entity Types
The dataset contains 22 specific entity types commonly found in Indonesian legal court decisions, following the IOB (Inside-Outside-Begin) tagging scheme. Based on the actual annotation distribution, the entity types are:

#### Legal Personnel
- **Nama Terdakwa** (Defendant Name): Names of defendants in legal cases
- **Nama Hakim Ketua** (Chief Judge Name): Names of presiding judges
- **Nama Hakim Anggota** (Associate Judge Name): Names of associate judges
- **Nama Jaksa** (Prosecutor Name): Names of public prosecutors
- **Nama Saksi** (Witness Name): Names of witnesses in cases
- **Nama Panitera** (Court Clerk Name): Names of court clerks
- **Nama Pengacara** (Lawyer Name): Names of defense attorneys

#### Legal Case Information
- **Nomor Putusan** (Decision Number): Court decision reference numbers
- **Nama Pengadilan** (Court Name): Names of courts handling cases
- **Jenis Perkara** (Case Type): Types of legal cases
- **Jenis Dakwaan** (Indictment Type): Types of indictments
- **Tingkat Kasus** (Case Level): Court level (first instance, appeal, cassation)

#### Legal Content & Decisions
- **Melanggar UU (Dakwaan)** (Law Violation - Indictment): Legal articles violated according to indictment
- **Melanggar UU (Tuntutan)** (Law Violation - Prosecution): Legal articles cited in prosecution
- **Melanggar UU (Pertimbangan Hukum)** (Law Violation - Legal Consideration): Legal articles in court's consideration
- **Tuntutan Hukuman** (Prosecution Demand): Sentences demanded by prosecution
- **Putusan Hukuman** (Court Sentence): Final sentences given by court
- **Jenis Amar** (Verdict Type): Types of court verdicts

#### Temporal Information
- **Tanggal Kejadian** (Incident Date): Dates when incidents occurred
- **Tanggal Putusan** (Decision Date): Dates when court decisions were made

#### Dataset Statistics
- **Total Entities**: 22 unique entity types
- **Total Annotations**: 5,896,509 tokens (including O tags)
- **Entity Coverage**: 2.1% of tokens are labeled entities (non-O tags)
- **Most Common Entities**: Legal violations, personnel names, and case information

### Tag Distribution Analysis
The following table shows the distribution of NER tags in the complete dataset, demonstrating the class imbalance typical in NER tasks:

| Tag Type | Count | Percentage | Description |
|----------|--------|-----------|-------------|
| **O (Outside)** | 5,769,685 | 97.85% | Non-entity tokens |
| **I-Melanggar UU (Dakwaan)** | 12,798 | 0.22% | Law violations in indictment |
| **I-Melanggar UU (Pertimbangan Hukum)** | 11,712 | 0.20% | Law violations in legal consideration |
| **I-Putusan Hukuman** | 10,444 | 0.18% | Court sentences |
| **I-Tuntutan Hukuman** | 8,662 | 0.15% | Prosecution demands |
| **I-Melanggar UU (Tuntutan)** | 8,467 | 0.14% | Law violations in prosecution |
| **I-Nama Saksi** | 8,446 | 0.14% | Witness names |
| **I-Jenis Perkara** | 8,400 | 0.14% | Case types |
| **I-Nama Hakim Anggota** | 4,943 | 0.08% | Associate judge names |
| **I-Nama Terdakwa** | 4,182 | 0.07% | Defendant names |

**Key Insights:**
- **Class Imbalance**: 97.85% of tokens are non-entities (O tags)
- **Most Frequent Entities**: Legal violations and sanctions dominate
- **Personnel Information**: Names of legal actors are well-represented
- **Complete Coverage**: All 993 documents have consistent annotations

## 🏗️ Model Architecture

### Base Model
- **Architecture**: XLM-RoBERTa Large
- **Parameters**: ~560M parameters
- **Pre-training**: Multilingual corpus including Indonesian
- **Task Adaptation**: Token Classification for NER

### Model Configuration
```python
Model: xlm-roberta-large
Number of Labels: 45 (22 entity types × 2 for B-/I- tags + 1 for O tag)
Max Sequence Length: 512 tokens
Classification Head: Linear layer for token classification
Loss Function: CrossEntropyLoss with label smoothing
Entity Types: 22 specific Indonesian legal entities
Annotation Scheme: IOB (Inside-Outside-Begin) tagging
```

### Training Configuration
- **Optimizer**: AdamW
- **Learning Rate**: 2e-5
- **Batch Size**: 4 (training), 6 (evaluation)
- **Epochs**: 2 (with early stopping)
- **Weight Decay**: 0.01
- **Warmup Ratio**: 0.1
- **Gradient Accumulation**: Enabled for memory optimization

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
PyTorch 2.0+
Transformers 4.0+
CUDA (optional, for GPU acceleration)
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/indoler.git
cd indoler
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For GPU support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Basic Usage

#### 1. Data Preprocessing
```python
import pandas as pd
import json
from transformers import AutoTokenizer

# Load dataset
with open('Datasets/PUBLIC/data.json', 'r') as f:
    data = json.load(f)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

# Run preprocessing (see indoler.ipynb for full implementation)
```

#### 2. Model Training
```python
from transformers import AutoModelForTokenClassification, Trainer

# Load model
model = AutoModelForTokenClassification.from_pretrained(
    "xlm-roberta-large",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# Train model (see notebook for complete training loop)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
```

#### 3. Inference
```python
# Load trained model
model = AutoModelForTokenClassification.from_pretrained("./models/xlm_roberta_ner_results")
tokenizer = AutoTokenizer.from_pretrained("./models/xlm_roberta_ner_results")

# Example text
text = "Majelis Hakim yang diketuai oleh Budi Santoso telah memutuskan bahwa terdakwa Ahmad Rahman terbukti melakukan tindak pidana korupsi."

# Tokenize and predict
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=2)

# Decode predictions
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
predicted_labels = [id2label[pred.item()] for pred in predictions[0]]
```

## 📋 Preprocessing Pipeline

### 1. Data Loading and Exploration
- Load JSON dataset with court decision documents
- Analyze document structure and metadata
- Perform exploratory data analysis (EDA)
- Visualize entity type distributions

### 2. Data Quality Assurance
- **Consistency Check**: Validate IOB tagging format
- **Entity Extraction**: Extract and analyze entity types
- **Annotation Quality**: Review annotation consistency across documents
- **Statistical Analysis**: Compute entity distribution and dataset statistics

### 3. Dataset Splitting
- Load predefined train/validation/test splits from CSV files
- Ensure no overlap between splits
- Maintain entity distribution balance across splits
- Verify verdict and metadata distribution

### 4. Tokenization and Alignment
- **Subword Tokenization**: Use XLM-RoBERTa tokenizer
- **Label Alignment**: Align IOB labels with subword tokens
- **Special Token Handling**: Handle [CLS], [SEP], and [PAD] tokens
- **Length Management**: Truncate sequences to maximum length (512 tokens)
- **Label Mapping**: Convert string labels to numerical IDs

### 5. Data Serialization
- Save tokenized datasets in pickle format
- Store metadata and label mappings in JSON
- Create preprocessing artifacts for model training
- Generate dataset statistics and summaries

## 🧠 Model Training Process

### 1. Model Initialization
- Load pre-trained XLM-RoBERTa Large model
- Configure token classification head
- Set up label mappings and model parameters
- Initialize training components (optimizer, scheduler)

### 2. Training Configuration
```python
TrainingArguments(
    output_dir='./models/xlm_roberta_ner_results',
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=6,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)
```

### 3. Training Loop
- **Forward Pass**: Compute model predictions
- **Loss Calculation**: CrossEntropyLoss with ignored tokens (-100)
- **Backward Pass**: Gradient computation and optimization
- **Evaluation**: Regular validation on development set
- **Early Stopping**: Based on F1-score improvement
- **Model Checkpointing**: Save best performing models

### 4. Evaluation Metrics
- **Sequence-level Evaluation**: Using seqeval library
- **Entity-level F1-Score**: Primary evaluation metric
- **Precision and Recall**: Per-entity and overall metrics
- **Classification Report**: Detailed per-class performance

## 📊 Model Evaluation

### Performance Metrics
The model is evaluated using standard NER evaluation protocols:

- **Entity-level F1-Score**: Primary metric (strict evaluation)
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **Support**: Number of true instances for each entity type

### Evaluation Process
1. **Token-level Predictions**: Model outputs logits for each token
2. **Label Decoding**: Convert logits to predicted labels
3. **Entity Extraction**: Group B- and I- tags into complete entities
4. **Sequence Evaluation**: Compare predicted vs. true entity sequences
5. **Detailed Analysis**: Per-entity-type performance breakdown

### Validation Strategy
- **Development Set Evaluation**: Regular evaluation during training
- **Test Set Evaluation**: Final model performance assessment
- **Entity-wise Analysis**: Performance breakdown by entity type
- **Error Analysis**: Identification of common prediction errors

## 💡 Key Features

### 1. **Multilingual Foundation**
- Built on XLM-RoBERTa for robust Indonesian language understanding
- Cross-lingual transfer learning capabilities
- Subword tokenization for handling Indonesian morphology

### 2. **Domain Specialization**
- Specifically trained on legal document corpus
- Captures legal terminology and context
- Optimized for Indonesian court decision documents

### 3. **Robust Preprocessing**
- Comprehensive data quality checks
- IOB format validation and correction
- Efficient tokenization and label alignment

### 4. **Production Ready**
- Complete training and inference pipeline
- Model serialization and deployment support
- Comprehensive evaluation and monitoring

### 5. **Extensible Architecture**
- Modular code structure for easy extension
- Support for additional entity types
- Configurable hyperparameters and training settings

## 📂 Project Structure

```
indoler/
├── Code/
│   └── main/
│       └── indoler.ipynb              # Main training notebook
├── Datasets/
│   └── PUBLIC/
│       ├── data.json                  # Main dataset file
│       ├── data.meta.json            # Dataset metadata
│       ├── README.md                 # Dataset documentation
│       ├── train.ids.csv             # Training set IDs
│       ├── val.ids.csv               # Validation set IDs
│       └── test.ids.csv              # Test set IDs
├── models/                           # Trained model outputs
├── results/                          # Preprocessing results
├── requirements.txt                  # Python dependencies
└── README.md                        # Project documentation
```

## 🔧 Technical Requirements

### Dependencies
```txt
torch>=2.0.0
transformers>=4.20.0
datasets>=2.0.0
tokenizers>=0.12.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
seqeval>=1.2.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
```

### Hardware Requirements
- **Minimum**: 8GB RAM, 4GB GPU memory
- **Recommended**: 16GB RAM, 8GB+ GPU memory
- **Training Time**: ~30-60 minutes (depending on hardware)

### Software Requirements
- **Python**: 3.8 or higher
- **CUDA**: 11.0+ (for GPU acceleration)
- **Operating System**: Linux, macOS, or Windows

## 📈 Performance Optimization

### Memory Optimization
- **Gradient Accumulation**: Handle large effective batch sizes
- **Mixed Precision**: FP16 training for memory efficiency
- **Dynamic Padding**: Reduce memory usage with variable sequence lengths
- **Checkpoint Management**: Automatic cleanup of old checkpoints

### Training Optimization
- **Learning Rate Scheduling**: Warmup and decay strategies
- **Early Stopping**: Prevent overfitting with validation monitoring
- **Model Ensemble**: Combine multiple model predictions
- **Hyperparameter Tuning**: Systematic optimization of training parameters

## 🚀 Usage Examples

### Example 1: Entity Recognition in Indonesian Legal Text
```python
text = """
Majelis Hakim Pengadilan Negeri Jakarta Pusat yang diketuai oleh 
Dr. Siti Nurhaliza, S.H., M.H. telah menjatuhkan putusan nomor 
123/Pid.Sus/2023/PN.Jkt.Pst terhadap terdakwa Bambang Wijaya dalam 
perkara pidana korupsi berdasarkan Pasal 2 ayat (1) UU No. 31 Tahun 1999 
dengan putusan pidana penjara 5 tahun pada tanggal 15 Juni 2023.
"""

entities = predict_entities(text)
# Expected Output:
# Nama Pengadilan: Pengadilan Negeri Jakarta Pusat
# Nama Hakim Ketua: Dr. Siti Nurhaliza, S.H., M.H.
# Nomor Putusan: 123/Pid.Sus/2023/PN.Jkt.Pst
# Nama Terdakwa: Bambang Wijaya
# Jenis Perkara: pidana korupsi
# Melanggar UU (Pertimbangan Hukum): Pasal 2 ayat (1) UU No. 31 Tahun 1999
# Putusan Hukuman: pidana penjara 5 tahun
# Tanggal Putusan: 15 Juni 2023
```

### Example 2: Batch Processing of Legal Documents
```python
documents = [
    "Jaksa Penuntut Umum menuntut terdakwa Ahmad Subandi dengan pidana penjara 3 tahun...",
    "Berdasarkan Kitab Undang-Undang Hukum Pidana Pasal 362 tentang pencurian...",
    "Pengadilan Tinggi DKI Jakarta memutuskan menolak banding terdakwa..."
]

batch_results = model.predict_batch(documents)
# Returns entity predictions for each document
```

### Example 3: Real Court Decision Processing
```python
# Example from actual Indonesian court decision
court_text = """
Pengadilan Negeri Tangerang yang diperiksa dan diadili dalam persidangan 
Majelis Hakim yang diketuai oleh Ahmad Fauzi, S.H., M.H., dengan Hakim Anggota 
Drs. Bambang Sutrisno, S.H. dan Dra. Siti Maryam, S.H., M.H. serta didampingi 
Panitera Pengganti Rina Sari, S.H. telah menjatuhkan putusan terhadap 
terdakwa Joko Widodo dalam perkara Tindak Pidana Korupsi.
"""

entities = extract_legal_entities(court_text)
# Nama Pengadilan: Pengadilan Negeri Tangerang
# Nama Hakim Ketua: Ahmad Fauzi, S.H., M.H.
# Nama Hakim Anggota: Drs. Bambang Sutrisno, S.H., Dra. Siti Maryam, S.H., M.H.
# Nama Panitera: Rina Sari, S.H.
# Nama Terdakwa: Joko Widodo
# Jenis Perkara: Tindak Pidana Korupsi
```

## 🤝 Contributing

We welcome contributions to improve the IndoLER project:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Make your changes** and add tests
4. **Commit your changes**: `git commit -m 'Add new feature'`
5. **Push to the branch**: `git push origin feature/new-feature`
6. **Submit a Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation for API changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## 🙏 Acknowledgments

- **Indonesian Supreme Court** for providing the legal document corpus
- **Hugging Face** for the Transformers library and model hub
- **XLM-RoBERTa Team** for the multilingual language model
- **Dataset Annotators**: agree, dhipa, fariz, jafar for annotation work



## 🔗 Related Work

- [Indonesian NLP Resources](https://github.com/indonesian-nlp/indonesian-nlp)
- [Legal NLP Papers](https://github.com/legal-nlp/legal-nlp-papers)
- [XLM-RoBERTa Paper](https://arxiv.org/abs/1911.02116)
- [NER Evaluation Metrics](https://github.com/chakki-works/seqeval)

---
