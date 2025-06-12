# Penjelasan Tahapan Named Entity Recognition (NER) dalam Notebook

## Daftar Isi
1. [Overview Proyek](#overview-proyek)
2. [Setup dan Library](#setup-dan-library)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Preprocessing Data](#preprocessing-data)
5. [Model Development](#model-development)
6. [Training Model](#training-model)
7. [Evaluasi Model](#evaluasi-model)
8. [Deployment Preparation](#deployment-preparation)

---

## Overview Proyek

Proyek ini mengembangkan sistem **Named Entity Recognition (NER)** untuk dokumen hukum Indonesia menggunakan model **XLM-RoBERTa**. NER adalah tugas untuk mengidentifikasi dan mengklasifikasikan entitas bernama dalam teks seperti nama orang, tempat, organisasi, dll.

**Tujuan Utama:**
- Mengekstrak entitas penting dari dokumen hukum Indonesia
- Mengotomatisasi proses analisis dokumen legal
- Menyediakan model NER yang dapat di-deploy untuk production

---

## Setup dan Library

### Langkah 1: Import Library
```python
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from sklearn.model_selection import train_test_split
```

**Tujuan:**
- Mempersiapkan environment untuk data analysis dan machine learning
- Mengimport library untuk transformers (Hugging Face)
- Setup untuk visualisasi data dan preprocessing

### Langkah 2: Dependency Check
**Tujuan:**
- Memverifikasi instalasi library yang diperlukan
- Mengecek ketersediaan CUDA untuk GPU acceleration
- Memastikan compatibility antar library

---

## Exploratory Data Analysis (EDA)

### Langkah 3: Load Data
```python
with open('../../Datasets/PUBLIC/data.json', 'r') as f:
    data = json.load(f)
df = pd.json_normalize(data)
```

**Tujuan:**
- Memuat dataset dokumen hukum Indonesia dalam format JSON
- Mengkonversi ke DataFrame untuk analisis yang lebih mudah
- Memahami struktur data yang akan diproses

### Langkah 4: Analisis Struktur Data
**Yang Dilakukan:**
- Mengecek shape dan kolom DataFrame
- Memahami format anotasi BIO (Begin-Inside-Outside)
- Mengidentifikasi kolom penting: `text`, `text-tags`, `id`, `verdict`, dll.

**Tujuan:**
- Memahami struktur data anotasi NER
- Mengidentifikasi format input dan target labels
- Merencanakan strategi preprocessing

### Langkah 5: Analisis Distribusi Tags
```python
def get_all_tags(df):
    all_tags = []
    for tags in df['text-tags']:
        all_tags.extend(tags)
    return Counter(all_tags)
```

**Yang Dilakukan:**
- Menghitung distribusi semua tags NER dalam dataset
- Menganalisis persentase tag 'O' vs non-'O' tags
- Mengidentifikasi class imbalance

**Tujuan:**
- Memahami distribusi kelas dalam dataset
- Mengidentifikasi potensi masalah class imbalance
- Merencanakan strategi handling untuk kelas minoritas

### Langkah 6: Analisis Jenis Entity
```python
def analyze_entity_types(df):
    entity_types = set()
    entity_examples = {}
    # Extract entities dari format BIO
```

**Yang Dilakukan:**
- Mengekstrak semua jenis entitas dari tag B- dan I-
- Mengumpulkan contoh untuk setiap jenis entitas
- Menganalisis distribusi entity types

**Tujuan:**
- Memahami jenis-jenis entitas yang ada dalam dataset
- Melihat contoh konkret untuk setiap jenis entitas
- Memvalidasi kualitas anotasi manual

### Langkah 7: Konsistensi Anotasi BIO
```python
def check_annotation_consistency(df):
    # Mengecek konsistensi format BIO
    # I-tag harus didahului B-tag atau I-tag yang sama
```

**Yang Dilakukan:**
- Memvalidasi format anotasi BIO
- Mengidentifikasi error konsistensi seperti I-tag tanpa B-tag
- Melaporkan error untuk perbaikan data

**Tujuan:**
- Memastikan kualitas data anotasi
- Mengidentifikasi dan memperbaiki inconsistency
- Menjamin model dapat belajar dengan baik

### Langkah 8: Visualisasi Manual Review
```python
def display_annotations(df, idx, max_tokens=50):
    # Menampilkan teks dengan highlight anotasi
```

**Yang Dilakukan:**
- Menampilkan beberapa dokumen dengan anotasi visual
- Menunjukkan mapping token-to-label
- Mengekstrak entitas yang terdeteksi

**Tujuan:**
- Manual validation terhadap kualitas anotasi
- Memahami konteks penggunaan entitas
- Mengidentifikasi pattern anotasi

### Langkah 9: Visualisasi Distribusi Entity
**Yang Dilakukan:**
- Membuat bar chart dan pie chart distribusi entity types
- Menampilkan statistik jumlah entities per type
- Menganalisis top 10 entity types

**Tujuan:**
- Visual understanding distribusi data
- Mengidentifikasi entity types yang dominan
- Membantu dalam interpretasi hasil model nanti

---

## Preprocessing Data

### Langkah 10: Load Dataset Splits
```python
train_ids = pd.read_csv('../../Datasets/PUBLIC/train.ids.csv')
val_ids = pd.read_csv('../../Datasets/PUBLIC/val.ids.csv')
test_ids = pd.read_csv('../../Datasets/PUBLIC/test.ids.csv')
```

**Yang Dilakukan:**
- Memuat predefined split berdasarkan ID dokumen
- Memverifikasi tidak ada overlap antar splits
- Membagi dataset menjadi train/validation/test

**Tujuan:**
- Menggunakan split yang konsisten dan reproducible
- Memastikan evaluasi yang fair dan tidak bias
- Mengikuti best practice dalam ML experiment

### Langkah 11: Analisis Distribusi Split
**Yang Dilakukan:**
- Mengecek distribusi verdict di setiap split
- Memverifikasi balance antar splits
- Menganalisis representativeness setiap split

**Tujuan:**
- Memastikan setiap split representatif terhadap keseluruhan data
- Menghindari bias dalam training dan evaluation
- Memvalidasi strategi pembagian data

### Langkah 12: Setup XLM-RoBERTa Tokenizer
```python
model_name = "xlm-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

**Yang Dilakukan:**
- Load tokenizer XLM-RoBERTa untuk bahasa Indonesia
- Mengecek special tokens dan vocabulary size
- Testing tokenisasi pada sample data

**Tujuan:**
- Mempersiapkan tokenizer yang cocok untuk bahasa Indonesia
- Memahami cara kerja subword tokenization
- Memvalidasi compatibility dengan format data

### Langkah 13: Label Mapping
```python
def create_label_mapping(df):
    all_labels = set()
    for tags in df['text-tags']:
        all_labels.update(tags)
    # Buat mapping label ke ID numerik
```

**Yang Dilakukan:**
- Membuat mapping dari string labels ke numerical IDs
- Memastikan label 'O' di index 0
- Menganalisis jumlah unique labels dan entity types

**Tujuan:**
- Mengkonversi labels menjadi format yang compatible dengan model
- Mempersiapkan input/output format untuk training
- Mendokumentasikan mapping untuk inference nanti

### Langkah 14: Tokenisasi dan Alignment
```python
def tokenize_and_align_labels(examples_df, tokenizer, label2id, max_length=512):
    # Tokenisasi dengan preserving word boundaries
    # Alignment labels dengan subword tokens
```

**Yang Dilakukan:**
- Tokenisasi teks dengan `is_split_into_words=True`
- Alignment labels dengan subword tokens
- Handling special tokens dengan label -100 (ignore index)
- Truncation dan padding ke max_length

**Tujuan:**
- Mengkonversi format data menjadi input yang sesuai untuk BERT
- Menangani masalah subword tokenization dalam NER
- Mempersiapkan data untuk training dengan format yang benar

### Langkah 15: Analisis Panjang Dokumen
**Yang Dilakukan:**
- Menganalisis distribusi panjang dokumen dalam tokens
- Menentukan max_length optimal (512 tokens)
- Mengecek berapa dokumen yang akan kena truncation

**Tujuan:**
- Menentukan max_length yang optimal untuk model
- Meminimalkan information loss dari truncation
- Memahami implikasi dari setting max_length

### Langkah 16: Full Dataset Tokenization
**Yang Dilakukan:**
- Tokenisasi semua splits (train/val/test)
- Progress monitoring untuk dataset besar
- Error handling untuk dokumen bermasalah

**Tujuan:**
- Mempersiapkan dataset final untuk training
- Menghasilkan format input yang consistent
- Memvalidasi proses tokenisasi pada keseluruhan data

### Langkah 17: Analisis Hasil Tokenisasi
**Yang Dilakukan:**
- Menganalisis statistik hasil tokenisasi
- Mengecek distribusi label setelah alignment
- Menghitung persentase tokens yang di-ignore

**Tujuan:**
- Memvalidasi kualitas hasil preprocessing
- Memahami karakteristik data yang akan di-training
- Mengidentifikasi potensi masalah sebelum training

### Langkah 18: Save Preprocessing Results
**Yang Dilakukan:**
- Menyimpan tokenized datasets dalam format pickle
- Menyimpan metadata dan label mappings dalam JSON
- Dokumentasi lengkap preprocessing configuration

**Tujuan:**
- Mempersiapkan data untuk training phase
- Reproducibility dan traceability
- Efficiency dalam loading data untuk training

---

## Model Development

### Langkah 19: Load Preprocessed Data
**Yang Dilakukan:**
- Load tokenized datasets dari pickle files
- Load metadata dan label mappings
- Setup PyTorch Dataset class untuk NER

**Tujuan:**
- Memulai training phase dengan data yang sudah diproses
- Memastikan consistency dengan preprocessing phase
- Setup data pipeline untuk training

### Langkah 20: Model Setup
```python
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)
```

**Yang Dilakukan:**
- Load pre-trained XLM-RoBERTa model
- Configure untuk token classification task
- Setup label mappings dalam model config

**Tujuan:**
- Leveraging pre-trained knowledge untuk bahasa Indonesia
- Mengadaptasi model untuk NER task spesifik
- Setup architecture yang sesuai dengan jumlah entity types

### Langkah 21: Data Collator dan Metrics
```python
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    # Menggunakan seqeval metrics untuk NER
    # Precision, Recall, F1-score entity-level
```

**Yang Dilakukan:**
- Setup data collator untuk batch processing
- Implementasi metrics computation dengan seqeval
- Entity-level evaluation (bukan token-level)

**Tujuan:**
- Efficient batch processing selama training
- Evaluasi yang sesuai dengan task NER
- Metrics yang meaningful untuk entity extraction

---

## Training Model

### Langkah 22: Training Arguments Configuration
```python
training_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    warmup_ratio=0.1
)
```

**Yang Dilakukan:**
- Konfigurasi hyperparameters untuk training
- Setup logging dan evaluation strategy
- Configure model saving dan best model selection

**Tujuan:**
- Optimasi training process untuk NER task
- Balance antara performance dan computational cost
- Setup monitoring dan checkpointing

### Langkah 23: Trainer Setup dan Training
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_torch_dataset,
    eval_dataset=val_torch_dataset,
    compute_metrics=compute_metrics
)
```

**Yang Dilakukan:**
- Setup Hugging Face Trainer
- Execute training process dengan monitoring
- Auto save best model berdasarkan validation F1

**Tujuan:**
- Fine-tuning model untuk Indonesian legal NER
- Optimasi model parameters untuk domain-specific task
- Menghasilkan model yang siap untuk evaluation

---

## Evaluasi Model

### Langkah 24: Validation Evaluation
**Yang Dilakukan:**
- Evaluasi model pada validation set
- Menghitung precision, recall, dan F1-score
- Monitoring validation loss

**Tujuan:**
- Memvalidasi performance model selama training
- Early stopping berdasarkan validation metrics
- Memastikan model tidak overfitting

### Langkah 25: Test Set Evaluation
```python
test_predictions = trainer.predict(test_torch_dataset)
# Detailed classification report dengan seqeval
```

**Yang Dilakukan:**
- Evaluasi final pada test set yang belum pernah dilihat model
- Detailed classification report per entity type
- Entity-level precision, recall, F1 untuk setiap class

**Tujuan:**
- Unbiased evaluation terhadap model performance
- Understanding performance per entity type
- Validasi generalisasi model ke data baru

### Langkah 26: Entity-wise Performance Analysis
**Yang Dilakukan:**
- Analisis performance per jenis entitas
- Membandingkan predicted vs true entity counts
- Identifikasi entity types yang sulit diprediksi

**Tujuan:**
- Deep understanding tentang strengths dan weaknesses model
- Identifikasi area improvement untuk iterasi selanjutnya
- Validasi balance prediction across entity types

### Langkah 27: Model Inference Testing
```python
test_text = "Majelis Hakim yang diketuai oleh Budi Santoso..."
# Testing inference pada contoh teks baru
```

**Yang Dilakukan:**
- Testing model inference pada contoh real
- Demonstrasi entity extraction pipeline
- Validasi end-to-end functionality

**Tujuan:**
- Memastikan model dapat digunakan untuk inference real
- Demonstrasi capability model untuk stakeholders
- Validasi deployment readiness

---

## Deployment Preparation

### Langkah 28: Model Artifacts Preparation
**Yang Dilakukan:**
- Menyimpan trained model dan tokenizer
- Membuat comprehensive metadata dan configuration
- Setup label mappings untuk inference

**Tujuan:**
- Mempersiapkan model untuk production deployment
- Dokumentasi lengkap untuk reproducibility
- Easy loading dan configuration untuk inference

### Langkah 29: Preprocessing Utilities
**Yang Dilakukan:**
- Membuat class `IndonesianLegalNERPreprocessor`
- Implementasi text cleaning dan preprocessing functions
- End-to-end inference pipeline dari raw text ke entities

**Tujuan:**
- Abstraksi complexity preprocessing untuk end users
- Reusable components untuk different applications
- Production-ready inference pipeline

### Langkah 30: Final Model Summary
**Yang Dilakukan:**
- Comprehensive summary hasil training dan evaluation
- Dokumentasi model specifications dan performance
- Guide untuk model usage dan deployment

**Tujuan:**
- Dokumentasi lengkap project untuk stakeholders
- Clear performance benchmarks dan capabilities
- Instructions untuk production deployment

---

## Kesimpulan

Notebook ini mengimplementasikan complete pipeline untuk Indonesian Legal NER menggunakan XLM-RoBERTa, meliputi:

1. **EDA Comprehensive**: Analisis mendalam terhadap data dan anotasi
2. **Preprocessing Robust**: Tokenisasi dan alignment yang proper untuk BERT
3. **Model Training**: Fine-tuning dengan hyperparameter optimization
4. **Evaluation Thorough**: Entity-level metrics dan analysis per class
5. **Deployment Ready**: Complete artifacts dan utilities untuk production

**Key Achievements:**
- Model NER yang dapat mengidentifikasi entitas dalam dokumen hukum Indonesia
- Pipeline yang reproducible dan well-documented
- Ready-to-deploy solution dengan preprocessing utilities
- Comprehensive evaluation dan performance analysis

**Next Steps:**
- Production deployment dengan monitoring
- Continuous improvement berdasarkan user feedback
- Extension ke domain hukum lainnya atau bahasa regional Indonesia
