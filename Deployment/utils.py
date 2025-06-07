"""
Utility functions for Indonesian Legal NER Deployment
"""

import re
import logging
from typing import List, Dict, Any, Tuple
import pandas as pd
from datetime import datetime
import json
import os

def setup_logging(config):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, config["level"]),
        format=config["format"],
        handlers=[
            logging.FileHandler(config["file"]),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def clean_indonesian_text(text: str) -> str:
    """
    Advanced text cleaning for Indonesian legal documents
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Clean up common OCR errors in Indonesian
    text = re.sub(r'[''""„""]', '"', text)
    text = re.sub(r'[''`´]', "'", text)
    
    # Fix common Indonesian legal terms
    replacements = {
        'Mahkamah Agung': 'Mahkamah Agung',
        'Pengadilan Negeri': 'Pengadilan Negeri',
        'Pengadilan Tinggi': 'Pengadilan Tinggi',
        'Kejaksaan Negeri': 'Kejaksaan Negeri',
        'Kejaksaan Tinggi': 'Kejaksaan Tinggi',
    }
    
    for old, new in replacements.items():
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
    
    # Remove excessive punctuation
    text = re.sub(r'[.]{3,}', '...', text)
    text = re.sub(r'[-]{2,}', '--', text)
    
    # Clean up numbers and dates
    text = re.sub(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b', r'\1/\2/\3', text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    return text.strip()

def format_entities_for_display(entities: List[Dict]) -> str:
    """
    Format entities for display in the UI
    """
    if not entities:
        return "No entities found"
    
    # Group entities by type
    entity_groups = {}
    for entity in entities:
        label = entity['label']
        if label not in entity_groups:
            entity_groups[label] = []
        entity_groups[label].append(entity)
    
    # Format output
    output = []
    output.append(f"**Total Entities Found: {len(entities)}**\n")
    
    for label, group in sorted(entity_groups.items()):
        output.append(f"### {label} ({len(group)} entities)")
        
        # Show unique entities only
        unique_texts = list(set([e['text'] for e in group]))
        unique_texts.sort()
        
        for i, text in enumerate(unique_texts[:20], 1):  # Show max 20
            output.append(f"{i:2d}. {text}")
        
        if len(unique_texts) > 20:
            output.append(f"    ... and {len(unique_texts)-20} more")
        
        output.append("")  # Empty line
    
    return "\n".join(output)

def create_entity_dataframe(entities: List[Dict]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from entities for table display
    """
    if not entities:
        return pd.DataFrame(columns=['Entity', 'Type', 'Confidence', 'Position'])
    
    data = []
    for entity in entities:
        data.append({
            'Entity': entity['text'],
            'Type': entity['label'],
            'Confidence': f"{entity.get('confidence', 0.95):.2f}",
            'Position': f"{entity.get('start_word', 0)}-{entity.get('end_word', 0)}"
        })
    
    df = pd.DataFrame(data)
    return df

def highlight_entities_in_text(text: str, entities: List[Dict], colors: Dict[str, str]) -> str:
    """
    Create HTML with highlighted entities for display
    """
    if not entities:
        return text
    
    # Sort entities by start position (descending) to avoid position shifts
    sorted_entities = sorted(entities, key=lambda x: x.get('start_word', 0), reverse=True)
    
    words = text.split()
    
    for entity in sorted_entities:
        start = entity.get('start_word', 0)
        end = entity.get('end_word', 0)
        label = entity['label']
        
        if start < len(words) and end < len(words):
            color = colors.get(label, '#FFFACD')
            
            # Wrap entity words in span with background color
            highlighted_words = []
            for i in range(start, min(end + 1, len(words))):
                highlighted_words.append(
                    f'<span style="background-color: {color}; padding: 2px; border-radius: 3px; margin: 1px;">{words[i]}</span>'
                )
            
            # Replace original words with highlighted version
            words[start:end+1] = [' '.join(highlighted_words)]
    
    return ' '.join(words)

def export_entities_to_json(entities: List[Dict], filename: str) -> str:
    """
    Export entities to JSON format
    """
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'total_entities': len(entities),
        'entities': entities,
        'entity_counts': {}
    }
    
    # Count entities by type
    for entity in entities:
        label = entity['label']
        if label not in export_data['entity_counts']:
            export_data['entity_counts'][label] = 0
        export_data['entity_counts'][label] += 1
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    return filename

def validate_pdf_file(file_path: str, max_size: int) -> Tuple[bool, str]:
    """
    Validate uploaded PDF file
    """
    if not file_path:
        return False, "No file provided"
    
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    # Check file size
    file_size = os.path.getsize(file_path)
    if file_size > max_size:
        return False, f"File too large: {file_size/1024/1024:.1f}MB (max: {max_size/1024/1024:.1f}MB)"
    
    # Check file extension
    if not file_path.lower().endswith('.pdf'):
        return False, "File must be a PDF"
    
    return True, "File validation passed"

def extract_legal_context(text: str, entities: List[Dict]) -> Dict[str, Any]:
    """
    Extract additional legal context from the document
    """
    context = {
        'document_type': 'unknown',
        'court_level': 'unknown',
        'case_category': 'unknown',
        'judgment_type': 'unknown',
        'key_dates': [],
        'case_numbers': []
    }
    
    text_lower = text.lower()
    
    # Determine document type
    if 'putusan' in text_lower:
        context['document_type'] = 'court_decision'
    elif 'dakwaan' in text_lower:
        context['document_type'] = 'indictment'
    elif 'tuntutan' in text_lower:
        context['document_type'] = 'prosecution_demand'
    elif 'pledoi' in text_lower:
        context['document_type'] = 'defense_plea'
    
    # Determine court level
    if 'mahkamah agung' in text_lower:
        context['court_level'] = 'supreme_court'
    elif 'pengadilan tinggi' in text_lower:
        context['court_level'] = 'high_court'
    elif 'pengadilan negeri' in text_lower:
        context['court_level'] = 'district_court'
    
    # Extract case numbers using regex
    case_patterns = [
        r'\b\d+/[A-Za-z.]+/\d{4}/[A-Za-z.]+\b',
        r'\bNo\.\s*\d+/[A-Za-z.]+/\d{4}/[A-Za-z.]+\b',
        r'\bNomor\s*\d+/[A-Za-z.]+/\d{4}/[A-Za-z.]+\b'
    ]
    
    for pattern in case_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        context['case_numbers'].extend(matches)
    
    # Extract dates
    date_patterns = [
        r'\b\d{1,2}\s+\w+\s+\d{4}\b',
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b'
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        context['key_dates'].extend(matches)
    
    # Remove duplicates
    context['case_numbers'] = list(set(context['case_numbers']))
    context['key_dates'] = list(set(context['key_dates']))
    
    return context

def generate_summary_report(entities: List[Dict], context: Dict, text_length: int) -> str:
    """
    Generate a summary report of the analysis
    """
    entity_counts = {}
    for entity in entities:
        label = entity['label']
        entity_counts[label] = entity_counts.get(label, 0) + 1
    
    report = f"""
# Document Analysis Summary

## Document Information
- **Document Type**: {context.get('document_type', 'Unknown').replace('_', ' ').title()}
- **Court Level**: {context.get('court_level', 'Unknown').replace('_', ' ').title()}
- **Text Length**: {text_length:,} characters
- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Entity Statistics
- **Total Entities**: {len(entities)}
- **Entity Types**: {len(entity_counts)}

### Entity Breakdown
"""
    
    for label, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(entities) * 100) if entities else 0
        report += f"- **{label}**: {count} entities ({percentage:.1f}%)\n"
    
    if context.get('case_numbers'):
        report += f"\n## Case Numbers Found\n"
        for case_num in context['case_numbers'][:5]:  # Show first 5
            report += f"- {case_num}\n"
    
    if context.get('key_dates'):
        report += f"\n## Key Dates Found\n"
        for date in context['key_dates'][:5]:  # Show first 5
            report += f"- {date}\n"
    
    return report
