"""
Biomedical Named Entity Recognition (NER) Module.
Extracts genes/proteins, diseases, and drugs/chemicals.
"""

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

try:
    import spacy
    try:
        nlp = spacy.load("en_core_sci_sm")
        SPACY_AVAILABLE = True
    except OSError:
        logger.warning("en_core_sci_sm model not found. Falling back to regex.")
        nlp = None
        SPACY_AVAILABLE = False
except ImportError:
    logger.warning("spacy or scispacy not installed. Falling back to regex.")
    nlp = None
    SPACY_AVAILABLE = False

def regex_fallback_extract(text: str) -> Dict[str, List[str]]:
    """Simple regex-based fallback for biomedical entity extraction."""
    genes = re.findall(r'\b[A-Z][A-Z0-9-]{2,}\b', text)
    diseases = re.findall(r'\b(?:cancer|tumor|carcinoma|syndrome|disease|disorder)\b', text, re.IGNORECASE)
    drugs = re.findall(r'\b(?:mab|nib|zole|cillin|mycin|acid|sodium)\b', text, re.IGNORECASE)
    
    return {
        "genes": list(set(genes)),
        "diseases": list(set(diseases)),
        "drugs": list(set(drugs))
    }

def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract biomedical entities from text.
    Returns a dict with keys 'genes', 'diseases', 'drugs'.
    """
    if not SPACY_AVAILABLE or nlp is None:
        return regex_fallback_extract(text)

    doc = nlp(text)
    entities = {
        "genes": [],
        "diseases": [],
        "drugs": []
    }
    
    for ent in doc.ents:
        ent_text = ent.text
        lower_text = ent_text.lower()
        if any(word in lower_text for word in ['cancer', 'tumor', 'carcinoma', 'syndrome', 'disease', 'disorder', 'infection']):
            entities["diseases"].append(ent_text)
        elif any(word in lower_text for word in ['mab', 'nib', 'zole', 'cillin', 'mycin', 'acid', 'sodium']):
            entities["drugs"].append(ent_text)
        elif re.match(r'^[A-Z][A-Z0-9-]{1,}$', ent_text):
            entities["genes"].append(ent_text)
        else:
            # Default fallback for unclassified generic entities
            entities["genes"].append(ent_text)

    # Deduplicate
    return {k: list(set(v)) for k, v in entities.items()}
