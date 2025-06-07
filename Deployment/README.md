# Indonesian Legal NER - Deployment Guide

## Overview

This directory contains the deployment setup for the Indonesian Legal Named Entity Recognition (NER) model using Gradio for a user-friendly web interface.

## Features

- **PDF Upload**: Upload PDF legal documents for entity extraction
- **Text Input**: Direct text input for entity analysis  
- **Interactive UI**: Clean, responsive web interface
- **Entity Visualization**: Color-coded entity highlighting
- **Export Results**: Download extracted entities as JSON/CSV

## Files Structure

```
Deployment/
├── app.py                 # Main Gradio application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── config.py             # Configuration settings (optional)
```

## Setup Instructions

### 1. Install Dependencies

```bash
cd Deployment
pip install -r requirements.txt
```

### 2. Prepare Model Files

Ensure your trained model is available at:
```
../Code/main/models/model_final/
├── model_artifacts/       # Trained model files
├── label_mappings.json   # Label mappings
└── model_config.json     # Model configuration
```

### 3. Run the Application

```bash
python app.py
```

The app will start on `http://localhost:7860`

## Usage

### PDF Upload Method
1. Go to the "📄 PDF Upload" tab
2. Upload your Indonesian legal document (PDF format)
3. Click "Extract Text from PDF" to extract text
4. Click "Analyze Entities" to identify named entities

### Text Input Method
1. Go to the "📝 Text Input" tab
2. Paste or type Indonesian legal text
3. Click "Analyze Text" to extract entities
4. View results in the table and text output

## Model Information

- **Base Model**: XLM-RoBERTa Large
- **Language**: Indonesian
- **Domain**: Legal Documents
- **Task**: Named Entity Recognition
- **Max Input Length**: 512 tokens

## Supported Entity Types

The model recognizes various legal entities including:
- **PERSON**: Judge names, defendant names, lawyers, witnesses
- **ORG**: Court names, law firms, government institutions  
- **LOC**: Geographical locations
- **MISC**: Case numbers, legal references, dates
- **And more...**

## Deployment Options

### Local Deployment
```bash
python app.py
```

### Hugging Face Spaces
1. Create a new Space on Hugging Face
2. Upload all files to the Space
3. The app will automatically deploy

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 7860

CMD ["python", "app.py"]
```

## Performance Considerations

- **GPU Support**: The app automatically detects and uses GPU if available
- **Memory Usage**: Large model (~1.2GB) requires adequate RAM
- **Text Length**: Input text is truncated to 5000 characters for processing
- **PDF Size**: Large PDF files may take longer to process

## Troubleshooting

### Common Issues

1. **Model Loading Error**
   - Check if model files exist in the correct directory
   - Verify all required files are present

2. **PDF Extraction Error**
   - Ensure PDF contains searchable text (not scanned images)
   - Try with different PDF files

3. **Memory Error**
   - Reduce batch size or use smaller model
   - Ensure adequate system memory

4. **Slow Performance**
   - Use GPU if available
   - Reduce input text length

## Customization

### Modify Entity Types
Edit the label mappings in `../Code/main/models/model_final/label_mappings.json`

### Change UI Theme
Modify the Gradio theme in `app.py`:
```python
with gr.Blocks(theme=gr.themes.Monochrome()) as app:
```

### Add Custom Processing
Extend the `IndonesianLegalNER` class in `app.py` to add custom preprocessing or post-processing steps.

## API Endpoints

When running, the Gradio app provides API endpoints:
- `POST /api/predict` - Main prediction endpoint
- `GET /api/` - API documentation

## Support

For issues or questions:
1. Check model files and dependencies
2. Review error messages in console
3. Verify input format and content
4. Test with provided examples

## License

This deployment setup follows the same license as the main project.
