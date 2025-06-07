#!/usr/bin/env python3
"""
Enhanced Indonesian Legal NER - Gradio Application
Complete deployment package with advanced features
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
from datetime import datetime
import logging

# Import local modules
from config import MODEL_CONFIG, UI_CONFIG, FILE_CONFIG, ENTITY_CONFIG, LOGGING_CONFIG
from utils import (
    setup_logging, clean_indonesian_text, format_entities_for_display,
    create_entity_dataframe, highlight_entities_in_text, export_entities_to_json,
    validate_pdf_file, extract_legal_context, generate_summary_report
)

# Setup logging
logger = setup_logging(LOGGING_CONFIG)

class IndonesianLegalNERAdvanced:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.label_mappings = None
        self.max_length = MODEL_CONFIG["max_length"]
        self.device = self._get_device()
        self.model_loaded = False
        
    def _get_device(self):
        """Determine the best device to use"""
        device_config = MODEL_CONFIG["device"]
        if device_config == "auto":
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_config == "cuda" and torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
        
    def load_model(self):
        """Load the trained model and configurations"""
        try:
            model_path = MODEL_CONFIG["model_path"]
            config_path = MODEL_CONFIG["config_path"]
            
            logger.info(f"Loading model from {model_path}")
            
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Load label mappings
            with open(os.path.join(config_path, "label_mappings.json"), 'r', encoding='utf-8') as f:
                self.label_mappings = json.load(f)
            
            # Load model config
            with open(os.path.join(config_path, "model_config.json"), 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.max_length = config.get('max_length', 512)
            
            self.model_loaded = True
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def extract_text_from_pdf(self, pdf_file) -> Tuple[str, str]:
        """Extract text from uploaded PDF file with validation"""
        try:
            if pdf_file is None:
                return "", "Please upload a PDF file"
            
            # Validate file
            is_valid, message = validate_pdf_file(pdf_file, FILE_CONFIG["max_pdf_size"])
            if not is_valid:
                return "", f"Validation error: {message}"
            
            # Extract text
            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                if len(pdf_reader.pages) > FILE_CONFIG["max_pages"]:
                    return "", f"PDF has too many pages ({len(pdf_reader.pages)}). Maximum allowed: {FILE_CONFIG['max_pages']}"
                
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += f"\\n--- Page {page_num + 1} ---\\n{page_text}\\n"
                
                if not text.strip():
                    return "", "No text could be extracted from the PDF. The file might contain only images."
            
            # Clean text
            cleaned_text = clean_indonesian_text(text)
            
            logger.info(f"Successfully extracted {len(cleaned_text)} characters from PDF")
            return cleaned_text, f"✅ Successfully extracted {len(cleaned_text)} characters from {len(pdf_reader.pages)} pages"
            
        except Exception as e:
            error_msg = f"Error extracting PDF: {str(e)}"
            logger.error(error_msg)
            return "", error_msg
    
    def predict_entities(self, text: str) -> Tuple[List[Dict], str, Dict]:
        """Predict named entities in text with enhanced features"""
        try:
            if not self.model_loaded:
                return [], "Model not loaded", {}
            
            if not text.strip():
                return [], "No text provided", {}
            
            # Clean and prepare text
            cleaned_text = clean_indonesian_text(text)
            original_length = len(cleaned_text)
            
            # Limit text length
            max_length = MODEL_CONFIG["max_text_length"]
            if len(cleaned_text) > max_length:
                cleaned_text = cleaned_text[:max_length]
                truncated_msg = f"⚠️ Text truncated from {original_length} to {len(cleaned_text)} characters for processing"
            else:
                truncated_msg = ""
            
            # Split into words
            words = cleaned_text.split()
            
            if len(words) == 0:
                return [], "No words found in text", {}
            
            logger.info(f"Processing {len(words)} words")
            
            # Tokenize
            encoding = self.tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Move to device
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**encoding)
                predictions = torch.argmax(outputs.logits, dim=2)
                confidence_scores = torch.softmax(outputs.logits, dim=2)
            
            # Get word IDs for alignment
            word_ids = encoding.word_ids()
            
            # Convert predictions to labels
            predicted_labels = []
            confidences = []
            
            for i, pred in enumerate(predictions[0]):
                label_id = pred.item()
                label = self.label_mappings['id2label'][str(label_id)]
                predicted_labels.append(label)
                
                # Get confidence score
                conf_score = confidence_scores[0][i][label_id].item()
                confidences.append(conf_score)
            
            # Extract entities
            entities = self._extract_entities_from_predictions(
                words, predicted_labels, word_ids, confidences
            )
            
            # Extract legal context
            context = extract_legal_context(cleaned_text, entities)
            
            # Create statistics
            stats = self._create_enhanced_statistics(entities, context, original_length)
            
            logger.info(f"Found {len(entities)} entities")
            return entities, truncated_msg, stats
            
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            logger.error(error_msg)
            return [], error_msg, {}
    
    def _extract_entities_from_predictions(self, words: List[str], 
                                         labels: List[str], 
                                         word_ids: List[int],
                                         confidences: List[float]) -> List[Dict[str, Any]]:
        """Extract entities from predictions with confidence scores"""
        entities = []
        current_entity = []
        current_label = None
        current_start = None
        current_confidences = []
        
        for i, (word_id, label, conf) in enumerate(zip(word_ids, labels, confidences)):
            if word_id is None:  # Special tokens
                continue
                
            if word_id >= len(words):  # Safety check
                continue
                
            word = words[word_id]
            
            if label.startswith('B-'):
                # Save previous entity
                if current_entity:
                    avg_confidence = sum(current_confidences) / len(current_confidences)
                    entities.append({
                        'text': ' '.join(current_entity),
                        'label': current_label,
                        'start_word': current_start,
                        'end_word': current_start + len(current_entity) - 1,
                        'confidence': avg_confidence,
                        'word_count': len(current_entity)
                    })
                
                # Start new entity
                current_entity = [word]
                current_label = label[2:]
                current_start = word_id
                current_confidences = [conf]
                
            elif label.startswith('I-') and current_label == label[2:]:
                # Continue current entity
                if len(current_entity) == 0 or word_id <= current_start + len(current_entity):
                    current_entity.append(word)
                    current_confidences.append(conf)
                    
            else:  # O label or different entity type
                # Save previous entity
                if current_entity:
                    avg_confidence = sum(current_confidences) / len(current_confidences)
                    entities.append({
                        'text': ' '.join(current_entity),
                        'label': current_label,
                        'start_word': current_start,
                        'end_word': current_start + len(current_entity) - 1,
                        'confidence': avg_confidence,
                        'word_count': len(current_entity)
                    })
                
                current_entity = []
                current_label = None
                current_start = None
                current_confidences = []
        
        # Save final entity
        if current_entity:
            avg_confidence = sum(current_confidences) / len(current_confidences)
            entities.append({
                'text': ' '.join(current_entity),
                'label': current_label,
                'start_word': current_start,
                'end_word': current_start + len(current_entity) - 1,
                'confidence': avg_confidence,
                'word_count': len(current_entity)
            })
        
        # Filter by confidence threshold
        threshold = ENTITY_CONFIG["confidence_threshold"]
        filtered_entities = [e for e in entities if e['confidence'] >= threshold]
        
        logger.info(f"Filtered {len(entities)} -> {len(filtered_entities)} entities (threshold: {threshold})")
        
        return filtered_entities
    
    def _create_enhanced_statistics(self, entities: List[Dict], context: Dict, text_length: int) -> Dict:
        """Create enhanced entity statistics"""
        if not entities:
            return {}
        
        stats = {}
        entity_counts = {}
        confidence_sum = 0
        
        for entity in entities:
            label = entity['label']
            if label not in entity_counts:
                entity_counts[label] = 0
            entity_counts[label] += 1
            confidence_sum += entity['confidence']
        
        stats['total_entities'] = len(entities)
        stats['entity_types'] = len(entity_counts)
        stats['entity_counts'] = entity_counts
        stats['average_confidence'] = confidence_sum / len(entities)
        stats['text_length'] = text_length
        stats['context'] = context
        stats['analysis_time'] = datetime.now().isoformat()
        
        return stats

# Initialize the enhanced NER model
ner_model = IndonesianLegalNERAdvanced()

def initialize_model():
    """Initialize model on app startup"""
    logger.info("Initializing model...")
    success = ner_model.load_model()
    if success:
        device_info = f" (Device: {ner_model.device})"
        return f"✅ Model loaded successfully!{device_info}"
    else:
        return "❌ Failed to load model. Please check the model files."

def process_pdf_enhanced(pdf_file):
    """Enhanced PDF processing with validation"""
    if pdf_file is None:
        return "Please upload a PDF file", "", ""
    
    logger.info(f"Processing PDF: {pdf_file}")
    
    # Extract text from PDF
    text, status = ner_model.extract_text_from_pdf(pdf_file)
    
    if not text:
        return status, "", ""
    
    # Show preview (first 1000 characters)
    preview = text[:1000] + "..." if len(text) > 1000 else text
    
    return status, preview, text

def process_text_enhanced(text, include_context=True):
    """Enhanced text processing with context analysis"""
    if not text.strip():
        return "Please provide some text to analyze", "", "", ""
    
    logger.info(f"Processing text: {len(text)} characters")
    
    entities, message, stats = ner_model.predict_entities(text)
    
    # Format results
    if not entities:
        return "No entities found in the text", pd.DataFrame(), "", ""
    
    # Create formatted output
    formatted_output = format_entities_for_display(entities)
    
    # Create entity table
    entity_df = create_entity_dataframe(entities)
    
    # Generate summary report
    context = stats.get('context', {})
    summary = generate_summary_report(entities, context, len(text)) if include_context else ""
    
    # Create highlighted text (limited for display)
    display_text = text[:2000] + "..." if len(text) > 2000 else text
    highlighted_text = highlight_entities_in_text(display_text, entities, ENTITY_CONFIG["entity_colors"])
    
    return formatted_output, entity_df, summary, highlighted_text

def export_results(entities_json, filename_prefix="legal_ner_results"):
    """Export results to JSON file"""
    if not entities_json:
        return "No results to export"
    
    try:
        # Parse entities if they're in string format
        if isinstance(entities_json, str):
            entities = json.loads(entities_json)
        else:
            entities = entities_json
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.json"
        
        # Export to temporary file
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)
        
        export_entities_to_json(entities, filepath)
        
        return f"Results exported to: {filepath}"
        
    except Exception as e:
        logger.error(f"Export error: {e}")
        return f"Export failed: {str(e)}"

def create_enhanced_gradio_app():
    """Create the enhanced Gradio interface"""
    
    # Custom CSS for better appearance
    css = """
    .gradio-container {
        max-width: 1400px !important;
        margin: auto;
    }
    .entity-highlight {
        padding: 2px 6px;
        border-radius: 4px;
        margin: 0 2px;
        font-weight: 500;
    }
    .stats-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .header-text {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(css=css, title=UI_CONFIG["title"], theme=getattr(gr.themes, UI_CONFIG["theme"].title())()) as app:
        
        # Header
        gr.Markdown("""
        <div class="header-text">
        
        # 🏛️ Indonesian Legal Named Entity Recognition
        ### Advanced AI-powered entity extraction for Indonesian legal documents
        
        Extract and analyze named entities from Indonesian court decisions, legal contracts, and other legal documents using state-of-the-art XLM-RoBERTa model.
        
        </div>
        """)
        
        # Model status
        with gr.Row():
            model_status = gr.Textbox(
                value=initialize_model(),
                label="🔧 Model Status",
                interactive=False,
                show_copy_button=True
            )
        
        with gr.Tabs():
            # PDF Upload Tab
            with gr.TabItem("📄 PDF Document Analysis"):
                gr.Markdown("Upload PDF legal documents for automatic entity extraction and analysis.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        pdf_input = gr.File(
                            file_types=[".pdf"],
                            label="📤 Upload PDF Document",
                            type="filepath"
                        )
                        
                        with gr.Row():
                            pdf_process_btn = gr.Button("🔍 Extract Text", variant="primary")
                            pdf_analyze_btn = gr.Button("🧠 Analyze Entities", variant="secondary")
                    
                    with gr.Column(scale=2):
                        pdf_status = gr.Textbox(label="📋 Processing Status", interactive=False)
                        pdf_preview = gr.Textbox(
                            label="👀 Text Preview",
                            lines=8,
                            interactive=False
                        )
                
                pdf_extracted_text = gr.Textbox(visible=False)  # Hidden full text
                
                with gr.Row():
                    with gr.Column(scale=2):
                        pdf_results = gr.Textbox(
                            label="📊 Entity Analysis Results",
                            lines=12,
                            interactive=False
                        )
                    
                    with gr.Column(scale=1):
                        pdf_summary = gr.Textbox(
                            label="📈 Document Summary",
                            lines=12,
                            interactive=False
                        )
                
                with gr.Row():
                    pdf_table = gr.Dataframe(
                        label="📋 Detailed Entity Table",
                        headers=["Entity", "Type", "Confidence", "Position"],
                        interactive=False
                    )
                
                pdf_highlighted = gr.HTML(label="🎨 Highlighted Text")
                pdf_message = gr.Textbox(label="💬 Analysis Messages", interactive=False)
            
            # Text Input Tab  
            with gr.TabItem("📝 Direct Text Analysis"):
                gr.Markdown("Paste or type Indonesian legal text directly for immediate entity analysis.")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="📝 Indonesian Legal Text",
                            placeholder="Masukkan teks hukum Indonesia di sini...",
                            lines=12
                        )
                        
                        with gr.Row():
                            text_analyze_btn = gr.Button("🧠 Analyze Text", variant="primary", size="lg")
                            text_clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                        
                        # Enhanced examples
                        gr.Examples(
                            examples=[
                                ["Majelis Hakim Pengadilan Negeri Jakarta Pusat yang diketuai oleh Dr. Budi Santoso, S.H., M.H. dengan anggota Hakim Siti Nurhaliza, S.H. dan Ahmad Rahman, S.H., M.H. telah menjatuhkan putusan Nomor 123/Pid.Sus-TPK/2023/PN.Jkt.Pst. bahwa terdakwa Ahmad Wijaya terbukti secara sah dan meyakinkan bersalah melakukan tindak pidana korupsi."],
                                ["Berdasarkan dakwaan Jaksa Penuntut Umum pada Kejaksaan Negeri Surabaya, terdakwa Siti Aminah, S.E. yang beralamat di Jalan Merdeka No. 45, Surabaya, terbukti melakukan tindak pidana penipuan sebagaimana diatur dalam Pasal 378 KUHP dengan kerugian negara sebesar Rp 2.500.000.000,- (dua miliar lima ratus juta rupiah)."],
                                ["Pengadilan Tinggi DKI Jakarta melalui putusannya Nomor 456/Pid.B/2023/PT.DKI memutuskan menolak permohonan banding yang diajukan oleh Penasehat Hukum terdakwa, advokat Bambang Widodo, S.H., M.H. dari Law Firm Widodo & Partners, sehingga putusan Pengadilan Negeri Jakarta Selatan tetap berkekuatan hukum tetap."]
                            ],
                            inputs=text_input,
                            label="💡 Example Legal Texts"
                        )
                    
                    with gr.Column(scale=1):
                        text_summary = gr.Textbox(
                            label="📈 Quick Summary",
                            lines=8,
                            interactive=False
                        )
                        
                        text_stats = gr.JSON(
                            label="📊 Analysis Statistics",
                            visible=True
                        )
                
                with gr.Row():
                    with gr.Column(scale=2):
                        text_results = gr.Textbox(
                            label="📊 Entity Analysis Results",
                            lines=15,
                            interactive=False
                        )
                    
                    with gr.Column(scale=1):
                        text_highlighted = gr.HTML(label="🎨 Highlighted Text")
                
                with gr.Row():
                    text_table = gr.Dataframe(
                        label="📋 Detailed Entity Table",
                        headers=["Entity", "Type", "Confidence", "Position"],
                        interactive=False
                    )
        
        # Information and Export Section
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("ℹ️ Model Information", open=False):
                    gr.Markdown(f"""
                    ### 🤖 Model Details
                    - **Base Model**: XLM-RoBERTa Large
                    - **Task**: Named Entity Recognition (Token Classification)
                    - **Language**: Indonesian 🇮🇩
                    - **Domain**: Legal Documents ⚖️
                    - **Max Input Length**: {MODEL_CONFIG['max_length']} tokens
                    - **Device**: {ner_model.device if ner_model.model_loaded else 'Not loaded'}
                    
                    ### 🏷️ Supported Entity Types
                    The model can identify various types of legal entities:
                    - **👤 PERSON**: Judge names, defendant names, lawyers, witnesses
                    - **🏢 ORG**: Court names, law firms, government institutions
                    - **📍 LOC**: Geographical locations mentioned in documents
                    - **📄 MISC**: Case numbers, legal references, dates, amounts
                    - **⚖️ LEGAL**: Legal terms, article references, law citations
                    
                    ### 💡 Usage Tips
                    - For best results, use formal Indonesian legal text
                    - PDF files should contain searchable text (not scanned images)
                    - Text longer than {MODEL_CONFIG['max_text_length']:,} characters will be truncated
                    - The model works best with complete sentences and proper formatting
                    - Confidence threshold is set to {ENTITY_CONFIG['confidence_threshold']}
                    """)
            
            with gr.Column(scale=1):
                with gr.Accordion("📤 Export & Download", open=False):
                    export_filename = gr.Textbox(
                        label="📄 Filename Prefix",
                        value="legal_ner_results",
                        placeholder="Enter filename prefix"
                    )
                    
                    export_btn = gr.Button("💾 Export Results", variant="secondary")
                    export_status = gr.Textbox(label="Export Status", interactive=False)
        
        # Event Handlers
        
        # PDF processing
        pdf_process_btn.click(
            fn=process_pdf_enhanced,
            inputs=pdf_input,
            outputs=[pdf_status, pdf_preview, pdf_extracted_text]
        )
        
        pdf_analyze_btn.click(
            fn=lambda text: process_text_enhanced(text, include_context=True),
            inputs=pdf_extracted_text,
            outputs=[pdf_results, pdf_table, pdf_summary, pdf_highlighted]
        )
        
        # Text processing
        text_analyze_btn.click(
            fn=lambda text: process_text_enhanced(text, include_context=True),
            inputs=text_input,
            outputs=[text_results, text_table, text_summary, text_highlighted]
        )
        
        text_clear_btn.click(
            lambda: ("", "", pd.DataFrame(), "", ""),
            outputs=[text_input, text_results, text_table, text_summary, text_highlighted]
        )
        
        # Export functionality (placeholder - would need additional implementation)
        export_btn.click(
            lambda filename: f"Export functionality would save results as {filename}_timestamp.json",
            inputs=export_filename,
            outputs=export_status
        )
    
    return app

if __name__ == "__main__":
    logger.info("Starting Indonesian Legal NER Application")
    
    app = create_enhanced_gradio_app()
    
    app.launch(
        share=UI_CONFIG["share"],
        server_name=UI_CONFIG["server_name"],
        server_port=UI_CONFIG["server_port"],
        show_error=True,
        show_tips=True,
        enable_monitoring=True
    )
