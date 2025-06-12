"""
Indonesian Legal NER - Hugging Face Gradio App
PDF Entity Extraction Interface

This app provides a user-friendly interface for extracting named entities
from Indonesian legal documents using a fine-tuned XLM-RoBERTa model.
"""

import gradio as gr
import torch
import json
import os
import tempfile
from typing import List, Dict, Any, Tuple
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
import PyPDF2
import re
from io import StringIO

# Configuration
MODEL_PATH = "../Code/main/models/model_final/model_artifacts"
CONFIG_PATH = "../Code/main/models/model_final"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model path does not exist: {MODEL_PATH}")

class IndonesianLegalNER:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.label_mappings = None
        self.max_length = 512
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        """Load the trained model and configurations"""
        try:
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            self.model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
            self.model.to(self.device)
            self.model.eval()
            
            # Load label mappings
            with open(os.path.join(CONFIG_PATH, "label_mappings.json"), 'r', encoding='utf-8') as f:
                self.label_mappings = json.load(f)
            
            # Load model config
            with open(os.path.join(CONFIG_PATH, "model_config.json"), 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.max_length = config.get('max_length', 512)
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text with better error handling"""
        try:
            if not isinstance(text, str):
                text = str(text)
            
            # Remove null bytes and other problematic characters
            text = text.replace('\x00', '')
            
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Handle quotes safely
            text = text.replace('"', '"').replace('"', '"').replace('„', '"')
            text = text.replace(''', "'").replace(''', "'")
            
            # Remove excessive punctuation
            text = re.sub(r'\.{3,}', '...', text)
            
            # Remove control characters safely
            printable_chars = []
            for char in text:
                # Keep normal printable ASCII and extended characters
                if ord(char) >= 32 or char in ['\n', '\t']:
                    printable_chars.append(char)
            
            text = ''.join(printable_chars)
            
            return text.strip()
            
        except Exception as e:
            print(f"Error in clean_text: {e}")
            # Ultra-safe fallback
            try:
                return ' '.join(str(text).split())
            except:
                return ""
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            if pdf_file is None:
                return ""
            
            # Read PDF file
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return self.clean_text(text)
            
        except Exception as e:
            return f"Error extracting PDF: {str(e)}"
    def predict_entities(self, text: str) -> Tuple[List[Dict], str, Dict]:
        """Predict named entities in text with chunking for long texts"""
        try:
            if not text.strip():
                return [], "No text provided", {}
            words = text.split()
            original_length = len(words)
            
            # Choose strategy based on text length
            if len(words) > 2000:  # Very long text - use sliding window
                return self._predict_with_sliding_window(text, words, window_size=400, stride=200)
            elif len(words) > 400:  # Long text - use chunking
                return self._predict_with_chunking(text, words)
            else:  # Short text - process normally
                return self._predict_single_chunk(text, words)
            
        except Exception as e:
            return [], f"Error during prediction: {str(e)}", {}
    
    def _predict_single_chunk(self, text: str, words: List[str]) -> Tuple[List[Dict], str, Dict]:
        """Predict entities for single chunk (short text)"""
        try:
            # Tokenize with return_offsets_mapping
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
                max_length=self.max_length,
                padding="max_length"
            )
            
            # Get word IDs and offset mapping
            word_ids = encoding.word_ids(batch_index=0)
            offset_mapping = encoding.pop('offset_mapping')
            
            # Move to device
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**encoding)
                predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
            
            # Convert predictions to labels
            predicted_labels = [self.label_mappings['id2label'][str(pred)] for pred in predictions]
            
            # Extract entities
            entities = self._extract_entities_from_predictions(words, predicted_labels, word_ids)
            
            # Create statistics
            stats = self._create_statistics(entities)
            
            return entities, "", stats
            
        except Exception as e:
            return [], f"Error during single chunk prediction: {str(e)}", {}
    
    def _predict_with_chunking(self, text: str, words: List[str]) -> Tuple[List[Dict], str, Dict]:
        """Predict entities using chunking strategy for long texts"""
        try:
            chunk_size = 400
            overlap = 50
            all_entities = []
            processed_words = 0
            chunk_count = 0
            
            status_msg = f"📄 Processing long text ({len(words)} words) using chunking strategy..."
            
            while processed_words < len(words):
                # Calculate chunk boundaries
                start_idx = max(0, processed_words - overlap if processed_words > 0 else 0)
                end_idx = min(len(words), processed_words + chunk_size)
                
                # Get chunk words and text
                chunk_words = words[start_idx:end_idx]
                chunk_text = ' '.join(chunk_words)
                
                chunk_count += 1
                
                # Predict entities in chunk
                try:
                    chunk_entities, _, _ = self._predict_single_chunk(chunk_text, chunk_words)
                    
                    # Adjust entity positions for full text and avoid duplicates
                    for entity in chunk_entities:
                        # Adjust positions to global text
                        entity['start'] += start_idx
                        entity['end'] += start_idx
                        
                        # Check for duplicates (entities from overlap regions)
                        is_duplicate = False
                        for existing_entity in all_entities:
                            # Consider as duplicate if same text, label, and overlapping positions
                            if (entity['text'] == existing_entity['text'] and 
                                entity['label'] == existing_entity['label'] and
                                abs(entity['start'] - existing_entity['start']) <= overlap):
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            all_entities.append(entity)
                    
                except Exception as chunk_error:
                    print(f"Error processing chunk {chunk_count}: {chunk_error}")
                    continue
                
                processed_words += chunk_size
            
            # Sort entities by position
            all_entities.sort(key=lambda x: x['start'])
            
            # Create statistics
            stats = self._create_statistics(all_entities)
            
            # Update status message
            final_msg = f"✅ Processed {len(words)} words in {chunk_count} chunks. Found {len(all_entities)} entities."
            return all_entities, final_msg, stats
            
        except Exception as e:
            return [], f"Error during chunking prediction: {str(e)}", {}
    
    def _predict_with_sliding_window(self, text: str, words: List[str], 
                                   window_size: int = 400, stride: int = 200) -> Tuple[List[Dict], str, Dict]:
        """Alternative method using sliding window for very long documents"""
        try:
            all_entities = []
            entity_tracker = set()  # Track unique entities to avoid duplicates
            window_count = 0
            
            for start in range(0, len(words), stride):
                end = min(start + window_size, len(words))
                window_words = words[start:end]
                window_text = ' '.join(window_words)
                
                window_count += 1
                
                try:
                    entities, _, _ = self._predict_single_chunk(window_text, window_words)
                    
                    for entity in entities:
                        # Adjust positions to global text
                        entity['start'] += start
                        entity['end'] += start
                        
                        # Create unique identifier for deduplication
                        entity_key = (entity['start'], entity['text'], entity['label'])
                        
                        if entity_key not in entity_tracker:
                            entity_tracker.add(entity_key)
                            all_entities.append(entity)
                
                except Exception as window_error:
                    print(f"Error processing window {window_count}: {window_error}")
                    continue
            
            # Sort entities by position
            all_entities.sort(key=lambda x: x['start'])
            
            # Create statistics
            stats = self._create_statistics(all_entities)
            
            # Status message
            status_msg = f"✅ Processed {len(words)} words using sliding window ({window_count} windows). Found {len(all_entities)} entities."
            
            return all_entities, status_msg, stats
            
        except Exception as e:
            return [], f"Error during sliding window prediction: {str(e)}", {}

    def _extract_entities_from_predictions(self, words: List[str],
                                     labels: List[str], 
                                     word_ids: List[int]) -> List[Dict[str, Any]]:
        """Extract entities from predictions with proper word alignment"""
        entities = []
        current_entity = None
        
        for i, (word_id, label) in enumerate(zip(word_ids, labels)):
            if word_id is None:  # Skip special tokens
                continue
            
            # Safety check for word_id bounds
            if word_id >= len(words):
                continue
                
            word = words[word_id]
            
            # Handle beginning of entity
            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "text": word,
                    "label": label[2:],
                    "start": word_id,
                    "end": word_id,
                    "confidence": 0.95  # Default value
                }
            # Handle continuation of entity
            elif (label.startswith("I-") and 
                current_entity and 
                label[2:] == current_entity["label"]):
                # Check if this is the next consecutive word
                if word_id <= current_entity["end"] + 1:
                    if word_id == current_entity["end"] + 1:
                        current_entity["text"] += " " + word
                    current_entity["end"] = word_id
            # Handle non-entity tokens or broken sequences
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Add the last entity if exists
        if current_entity:
            entities.append(current_entity)
        
        return entities

    def _create_statistics(self, entities: List[Dict]) -> Dict:
        """Create entity statistics"""
        if not entities:
            return {}
        
        stats = {}
        entity_counts = {}
        
        for entity in entities:
            label = entity['label']
            if label not in entity_counts:
                entity_counts[label] = 0
            entity_counts[label] += 1
        
        stats['total_entities'] = len(entities)
        stats['entity_types'] = len(entity_counts)
        stats['entity_counts'] = entity_counts
        
        return stats

# Initialize the NER model
ner_model = IndonesianLegalNER()

def initialize_model():
    """Initialize model on app startup"""
    success = ner_model.load_model()
    if success:
        return "✅ Model loaded successfully!"
    else:
        return "❌ Failed to load model. Please check the model files."

def process_pdf(pdf_file):
    """Process uploaded PDF file"""
    if pdf_file is None:
        return "Please upload a PDF file", "", ""
    
    # Extract text from PDF
    text = ner_model.extract_text_from_pdf(pdf_file)
    
    if text.startswith("Error"):
        return text, "", ""
    
    if not text.strip():
        return "No text could be extracted from the PDF", "", ""
    
    # Show first 1000 characters of extracted text
    preview = text[:1000] + "..." if len(text) > 1000 else text
    
    return f"✅ Text extracted successfully ({len(text)} characters)", preview, text

def process_text(text):
    """Process text input for NER with chunking support"""    # Buat DataFrame kosong dengan struktur yang benar
    empty_df = pd.DataFrame(columns=['text', 'label', 'confidence', 'position'])
    
    if not text.strip():
        return "Please provide some text to analyze", empty_df, ""
    
    # Get word count for information
    word_count = len(text.split())
    
    entities, message, stats = ner_model.predict_entities(text)
    
    # Format results with processing info
    if not entities:
        result_text = f"📊 **Text Analysis Summary**\n"
        result_text += f"- **Words processed**: {word_count:,}\n"
        result_text += f"- **Processing method**: {'Sliding Window' if word_count > 2000 else 'Chunking' if word_count > 400 else 'Single Pass'}\n\n"
        result_text += "❌ **No entities found** in the text\n\n"
        result_text += "**Possible reasons:**\n"
        result_text += "- Text may not contain legal entities\n"
        result_text += "- Text format may not match training data\n"
        result_text += "- Try with more formal legal language\n"
        
        return result_text, empty_df, message
    
    # Create formatted output with processing info
    result_text = f"📊 **Text Analysis Summary**\n"
    result_text += f"- **Words processed**: {word_count:,}\n"
    result_text += f"- **Processing method**: {'Sliding Window' if word_count > 2000 else 'Chunking' if word_count > 400 else 'Single Pass'}\n"
    result_text += f"- **Entities found**: {len(entities)}\n"
    result_text += f"- **Entity types**: {stats.get('entity_types', 0)}\n\n"
    
    result_text += f"🏷️ **Named Entities Detected:**\n\n"
    
    # Group entities by type
    entity_groups = {}
    for entity in entities:
        label = entity['label']
        if label not in entity_groups:
            entity_groups[label] = []
        entity_groups[label].append(entity['text'])
    
    # Sort entity types by count (descending)
    sorted_groups = sorted(entity_groups.items(), key=lambda x: len(x[1]), reverse=True)
    
    for label, texts in sorted_groups:
        result_text += f"**{label}** ({len(texts)} entities):\n"
        # Show unique entities only
        unique_texts = list(dict.fromkeys(texts))  # Preserve order while removing duplicates
        for i, text in enumerate(unique_texts[:10]):  # Show first 10 unique entities
            result_text += f"  {i+1}. {text}\n"
        if len(unique_texts) > 10:
            result_text += f"  ... and {len(unique_texts)-10} more unique entities\n"
        if len(texts) > len(unique_texts):
            result_text += f"  (Total mentions: {len(texts)})\n"
        result_text += "\n"
    
    # Create detailed table - pastikan formatnya konsisten
    entity_data = []
    for entity in entities:
        entity_data.append({
            'text': entity.get('text', ''),
            'label': entity.get('label', ''),
            'confidence': entity.get('confidence', 0.0),
            'position': f"{entity.get('start', 0)}-{entity.get('end', 0)}"
        })
    
    entity_df = pd.DataFrame(entity_data)
    
    # Add processing information to message
    processing_info = ""
    if word_count > 2000:
        processing_info = f"🔄 Used sliding window processing for {word_count:,} words"
    elif word_count > 400:
        processing_info = f"🧩 Used chunking processing for {word_count:,} words"
    
    final_message = f"{processing_info}\n{message}".strip()
    
    return result_text, entity_df, final_message

def create_gradio_app():
    """Create the Gradio interface"""
    
    # Custom CSS for better appearance
    css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .entity-highlight {
        background-color: #ffeb3b;
        padding: 2px 4px;
        border-radius: 3px;
        margin: 0 1px;
    }
    """
    
    with gr.Blocks(css=css, title="Indonesian Legal NER", theme=gr.themes.Soft()) as app:
        
        gr.Markdown("""
        # 🏛️ Indonesian Legal Named Entity Recognition
        
        Extract named entities from Indonesian legal documents using a fine-tuned XLM-RoBERTa model.
        Upload a PDF document or paste text directly to identify legal entities such as names, places, organizations, and legal terms.
        """)
        
        # Model status
        with gr.Row():
            model_status = gr.Textbox(
                value=initialize_model(),
                label="Model Status",
                interactive=False
            )
        
        with gr.Tabs():
            # PDF Upload Tab
            with gr.TabItem("📄 PDF Upload"):
                with gr.Row():
                    with gr.Column(scale=1):
                        pdf_input = gr.File(
                            file_types=[".pdf"],
                            label="Upload PDF Document",
                            type="filepath"
                        )
                        pdf_process_btn = gr.Button("Extract Text from PDF", variant="primary")
                    
                    with gr.Column(scale=2):
                        pdf_status = gr.Textbox(label="Extraction Status", interactive=False)
                        pdf_preview = gr.Textbox(
                            label="Extracted Text Preview",
                            lines=10,
                            interactive=False
                        )
                
                pdf_extracted_text = gr.Textbox(visible=False)  # Hidden full text
                
                with gr.Row():
                    pdf_analyze_btn = gr.Button("Analyze Entities", variant="secondary", size="lg")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        pdf_results = gr.Textbox(
                            label="Entity Extraction Results",
                            lines=15,
                            interactive=False
                        )
                    with gr.Column(scale=1):
                        pdf_table = gr.Dataframe(
                            label="Detailed Entity Table",
                            headers=["text", "label", "confidence", "position"],
                            interactive=False
                        )
                
                pdf_message = gr.Textbox(label="Messages", interactive=False)
            
            # Text Input Tab  
            with gr.TabItem("📝 Text Input"):
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="Indonesian Legal Text",
                            placeholder="Paste your Indonesian legal text here...",
                            lines=10
                        )
                        
                        text_analyze_btn = gr.Button("Analyze Text", variant="primary", size="lg")
                        
                        gr.Examples(
                            examples=[
                                ["Majelis Hakim yang diketuai oleh Budi Santoso telah memutuskan bahwa terdakwa Ahmad Rahman terbukti melakukan tindak pidana korupsi dengan putusan nomor 123/Pid.Sus-TPK/2023/PN.Jkt.Pst."],
                                ["Pengadilan Negeri Jakarta Pusat melalui putusannya No. 456/Pid.B/2023/PN.Jkt.Pst memutuskan bahwa terdakwa Siti Aminah terbukti bersalah melakukan tindak pidana penipuan."],
                                ["Jaksa Penuntut Umum pada Kejaksaan Negeri Surabaya menuntut terdakwa Bambang Widodo dengan hukuman 5 tahun penjara atas dakwaan korupsi dana desa."]
                            ],
                            inputs=text_input
                        )

                    with gr.Column(scale=1):
                        text_results = gr.Textbox(
                            label="Entity Analysis Results",
                            lines=15,
                            interactive=False
                        )
                    with gr.Row():
                        text_table = gr.Dataframe(
                        label="Detailed Entity Table",
                        headers=["text", "label", "confidence", "position"],
                        interactive=False
                    )
                
                text_message = gr.Textbox(label="Analysis Messages", interactive=False)
        
        # Information section
        with gr.Accordion("ℹ️ Model Information", open=False):
            gr.Markdown(f"""
            ### Model Details
            - **Base Model**: XLM-RoBERTa Large
            - **Task**: Named Entity Recognition (Token Classification)
            - **Language**: Indonesian
            - **Domain**: Legal Documents
            - **Entity Types**: Judge names, defendant names, case numbers, court names, legal terms, etc.
            
            ### Supported Entity Types
            The model can identify various types of legal entities including:
            - **PERSON**: Names of judges, defendants, lawyers, witnesses
            - **ORG**: Court names, law firms, government institutions
            - **LOC**: Geographical locations mentioned in legal documents
            - **MISC**: Case numbers, legal references, dates
            
            ### Usage Tips
            - For best results, use formal Indonesian legal text
            - PDF files should contain searchable text (not scanned images)
            - Text longer than 5000 characters will be truncated
            - The model works best with complete sentences and proper formatting
            """)
        
        # Event handlers
        pdf_process_btn.click(
            fn=process_pdf,
            inputs=pdf_input,
            outputs=[pdf_status, pdf_preview, pdf_extracted_text]
        )
        
        pdf_analyze_btn.click(
            fn=process_text,
            inputs=pdf_extracted_text,
            outputs=[pdf_results, pdf_table, pdf_message]
        )
        
        text_analyze_btn.click(
            fn=process_text,
            inputs=text_input,
            outputs=[text_results, text_table, text_message]
        )
    
    return app

if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7800,
        show_error=True
    )
