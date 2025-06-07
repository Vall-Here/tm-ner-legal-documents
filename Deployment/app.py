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
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        # Clean up quotes
        text = re.sub(r'[""]', '"', text)
        text = re.sub(r'['']', "'", text)
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        return text.strip()
    
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
        """Predict named entities in text"""
        try:
            if not text.strip():
                return [], "No text provided", {}
            
            # Limit text length for processing
            if len(text) > 5000:
                text = text[:5000] + "..."
                truncated_msg = "⚠️ Text truncated to 5000 characters for processing"
            else:
                truncated_msg = ""
            
            # Split into words
            words = text.split()
            
            if len(words) == 0:
                return [], "No words found in text", {}
            
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
            
            # Get word IDs for alignment
            word_ids = encoding.word_ids()
            
            # Convert predictions to labels
            predicted_labels = []
            for pred in predictions[0]:
                label_id = pred.item()
                label = self.label_mappings['id2label'][str(label_id)]
                predicted_labels.append(label)
            
            # Extract entities
            entities = self._extract_entities_from_predictions(words, predicted_labels, word_ids)
            
            # Create statistics
            stats = self._create_statistics(entities)
            
            return entities, truncated_msg, stats
            
        except Exception as e:
            return [], f"Error during prediction: {str(e)}", {}
    
    def _extract_entities_from_predictions(self, words: List[str], 
                                        labels: List[str], 
                                        word_ids: List[int]) -> List[Dict[str, Any]]:
        """Extract entities from predictions"""

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
    """Process text input for NER"""
    if not text.strip():
        return "Please provide some text to analyze", "", ""
    
    entities, message, stats = ner_model.predict_entities(text)
    
    # Format results
    if not entities:
        return "No entities found in the text", "", ""
    
    # Create formatted output
    result_text = f"Found {len(entities)} entities:\n\n"
    
    # Group entities by type
    entity_groups = {}
    for entity in entities:
        label = entity['label']
        if label not in entity_groups:
            entity_groups[label] = []
        entity_groups[label].append(entity['text'])
    
    for label, texts in entity_groups.items():
        result_text += f"**{label}** ({len(texts)}):\n"
        for text in texts[:10]:  # Show first 10
            result_text += f"• {text}\n"
        if len(texts) > 10:
            result_text += f"... and {len(texts)-10} more\n"
        result_text += "\n"
    
    # Create detailed table
    entity_df = pd.DataFrame(entities)
    
    return result_text, entity_df, message

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
                            headers=["text", "label", "confidence"],
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
                        headers=["text", "label", "confidence"],
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
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
