"""
Biomedical Named Entity Recognition (NER) Module.
Extracts genes/proteins, diseases, and drugs/chemicals.
"""

import importlib
import logging
import re
from typing import Dict, List

from app.config import settings

logger = logging.getLogger(__name__)


def _load_model(name: str):
    """
    Load the biomedical spaCy pipeline, or return None to use the regex path.

    ``spacy.load(name)`` resolves models through installed *distribution*
    metadata. scripts/install_ner_model.py places the package directly (the
    upstream wheel can't be pip-installed on 3.12+), so there is no
    .dist-info and that lookup misses. Every spaCy model package also exposes
    its own ``load()``, which reads the data dir directly — so fall back to it.
    """
    try:
        import spacy
    except ImportError:
        logger.warning("spacy not installed — using regex entity extraction.")
        return None

    try:
        return spacy.load(name)
    except OSError:
        pass

    try:
        return importlib.import_module(name).load()
    except Exception as exc:
        logger.warning(
            "Biomedical NER model '%s' unavailable (%s) — using regex "
            "extraction. Run: python scripts/install_ner_model.py",
            name,
            exc.__class__.__name__,
        )
        return None


nlp = _load_model(settings.NER_MODEL)
SPACY_AVAILABLE = nlp is not None

if SPACY_AVAILABLE:
    logger.info("Biomedical NER active: %s", settings.NER_MODEL)

#: Drug names are matched by their stem suffix (e.g. trastuzu·mab, imati·nib),
#: so the pattern has to allow a prefix — an anchored \b before the suffix can
#: only ever match the bare suffix as a standalone word.
DRUG_SUFFIX_PATTERN = r'\b\w*(?:mab|nib|zole|cillin|mycin|acid|sodium)\b'

#: Gene/protein symbols are short all-caps tokens, optionally with digits or a
#: hyphen: BRCA1, TP53, ACE2, IL-6, SARS-CoV-2.
GENE_PATTERN = r'^[A-Z][A-Z0-9]*(?:-[A-Za-z0-9]+)*\d*$'

DISEASE_KEYWORDS = (
    'cancer', 'tumor', 'tumour', 'carcinoma', 'syndrome', 'disease',
    'disorder', 'infection', 'itis', 'emia', 'pathy', 'oma',
)


def _classify(term: str) -> str | None:
    """
    Bucket a single entity string, or return None if it isn't confidently
    biomedical.

    ``en_core_sci_sm`` tags every span with the single generic label
    ``ENTITY`` — it marks *where* entities are, not what kind. Typing is
    therefore ours to do, and it must be able to say "no": the previous
    version funnelled every unmatched span into ``genes``, so ordinary words
    like "increase", "mutations" and "risk" were reported as genes. Beyond
    being wrong on its face, those high-frequency terms match almost every
    chunk and so wash out the entity-aware re-ranking signal.
    """
    lowered = term.lower()

    if any(k in lowered for k in DISEASE_KEYWORDS):
        return 'diseases'
    if re.fullmatch(DRUG_SUFFIX_PATTERN.replace(r'\b', ''), lowered):
        return 'drugs'
    if _gene_symbols(term):
        return 'genes'
    return None


def _gene_symbols(span: str) -> List[str]:
    """
    Pull gene/protein symbols out of a span.

    The model returns whole noun phrases ("ACE2 receptor", "SARS-CoV-2
    entry"), so testing the span as a unit misses the symbol inside it — and
    the symbol is precisely the token that has to match for entity-aware
    re-ranking to beat plain vector search.
    """
    return [
        tok for tok in re.split(r'[\s/,()]+', span)
        if tok
        and re.fullmatch(GENE_PATTERN, tok)
        and any(c.isupper() for c in tok)
        and any(c.isalpha() for c in tok)
        and len(tok) > 1
    ]


def regex_fallback_extract(text: str) -> Dict[str, List[str]]:
    """Regex-based extraction, used when the scispaCy model is unavailable."""
    genes = re.findall(r'\b[A-Z][A-Z0-9-]{2,}\b', text)
    diseases = re.findall(
        r'\b(?:cancer|tumor|carcinoma|syndrome|disease|disorder)\b', text, re.IGNORECASE
    )
    drugs = re.findall(DRUG_SUFFIX_PATTERN, text, re.IGNORECASE)

    return {
        "genes": sorted(set(genes)),
        "diseases": sorted(set(diseases)),
        "drugs": sorted(set(drugs)),
    }


def extract_key_terms(text: str) -> List[str]:
    """
    Every biomedical span the model finds, lowercased and untyped.

    Deliberately separate from :func:`extract_entities`. Typing exists for the
    UI, which needs gene/disease/drug buckets; re-ranking does not care what
    *kind* of thing a term is, only whether it discriminates between chunks.
    Filtering to typed entities threw away exactly the discriminative terms —
    "metformin" matches no drug suffix, "RNA polymerase II" is not a gene
    symbol — leaving three of eight eval questions with no terms at all and
    silently degrading entity-aware retrieval to plain vector search.
    """
    if not SPACY_AVAILABLE or nlp is None:
        buckets = regex_fallback_extract(text)
        return sorted({t.lower() for v in buckets.values() for t in v})

    terms = {ent.text.lower().strip() for ent in nlp(text).ents}
    # Keep bare gene symbols too: "ACE2 receptor" should also match "ACE2".
    for ent in nlp(text).ents:
        terms.update(s.lower() for s in _gene_symbols(ent.text))
    return sorted(t for t in terms if len(t) > 2)


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract biomedical entities from text.

    Returns a dict with keys 'genes', 'diseases', 'drugs'. Spans that can't be
    confidently typed are dropped rather than guessed.
    """
    if not SPACY_AVAILABLE or nlp is None:
        return regex_fallback_extract(text)

    entities: Dict[str, List[str]] = {"genes": [], "diseases": [], "drugs": []}

    for ent in nlp(text).ents:
        bucket = _classify(ent.text)
        if bucket == 'genes':
            # Record the bare symbol(s), not the surrounding noun phrase.
            entities['genes'].extend(_gene_symbols(ent.text))
        elif bucket:
            entities[bucket].append(ent.text)

    return {k: sorted(set(v)) for k, v in entities.items()}
