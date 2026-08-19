"""
PubMed E-utilities client.

Two levels of detail are available:

* :func:`fetch_summaries` — titles + links only (esearch → esummary). Cheap;
  used to decorate an answer with clickable citations.
* :func:`fetch_abstracts` — full abstract text (esearch → efetch). Used when
  the retrieved literature should actually be fed to the model as context,
  and to build a corpus from a topic query.

NCBI asks that unauthenticated clients stay under ~3 requests/second, so the
helpers here batch aggressively rather than fetching one record at a time.
"""

import logging
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from typing import List

import httpx

from app import cache

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_TIMEOUT = 20.0


@dataclass
class PubMedArticle:
    """A single PubMed record with its abstract text."""

    pmid: str
    title: str
    abstract: str
    journal: str = ""
    year: str = ""
    authors: List[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

    @property
    def citation(self) -> str:
        """Short human-readable citation, e.g. 'Nature (2024)'."""
        bits = [b for b in (self.journal, f"({self.year})" if self.year else "") if b]
        return " ".join(bits) or "PubMed"

    def as_context(self) -> str:
        """Render the article the way it should appear in an LLM prompt."""
        header = f"[PubMed {self.pmid}] {self.title}"
        if self.citation != "PubMed":
            header += f" — {self.citation}"
        return f"{header}\n{self.abstract}"


async def search_pmids(
    query: str, max_results: int = 10, client: httpx.AsyncClient | None = None
) -> List[str]:
    """Run an esearch query and return the matching PMIDs."""
    cache_key = f"{query}|{max_results}"
    cached = await cache.get_json("pubmed:search", cache_key)
    if cached is not None:
        logger.info("PubMed search '%s' → %d PMIDs (cached)", query, len(cached))
        return cached

    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
    try:
        resp = await client.get(
            f"{EUTILS_BASE}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": max_results,
            },
        )
        resp.raise_for_status()
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        logger.info("PubMed search '%s' → %d PMIDs", query, len(ids))
        await cache.set_json("pubmed:search", cache_key, ids)
        return ids
    finally:
        if owns_client:
            await client.aclose()


def _text_of(node: ET.Element | None) -> str:
    """Flatten an element's text, including any nested markup tails."""
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def _parse_article(article: ET.Element) -> PubMedArticle | None:
    """Convert one <PubmedArticle> element into a :class:`PubMedArticle`."""
    pmid = _text_of(article.find(".//PMID"))
    if not pmid:
        return None

    title = _text_of(article.find(".//ArticleTitle"))

    # Structured abstracts split across several <AbstractText Label="..."> nodes.
    parts: List[str] = []
    for node in article.findall(".//Abstract/AbstractText"):
        text = _text_of(node)
        if not text:
            continue
        label = node.get("Label")
        parts.append(f"{label}: {text}" if label else text)
    abstract = " ".join(parts)

    journal = _text_of(article.find(".//Journal/ISOAbbreviation")) or _text_of(
        article.find(".//Journal/Title")
    )
    year = _text_of(article.find(".//Journal/JournalIssue/PubDate/Year"))

    authors: List[str] = []
    for author in article.findall(".//AuthorList/Author")[:3]:
        last, initials = _text_of(author.find("LastName")), _text_of(author.find("Initials"))
        if last:
            authors.append(f"{last} {initials}".strip())

    return PubMedArticle(
        pmid=pmid,
        title=title,
        abstract=abstract,
        journal=journal,
        year=year,
        authors=authors,
    )


async def fetch_abstracts(
    pmids: List[str], client: httpx.AsyncClient | None = None
) -> List[PubMedArticle]:
    """
    Fetch full abstract records for the given PMIDs via efetch.

    Records without an abstract (editorials, errata) are dropped — they carry
    no retrievable content.
    """
    if not pmids:
        return []

    # Cached per PMID rather than per request: topic queries overlap heavily
    # (a BRCA1 search and a BRCA2 search return many of the same papers), so
    # only the genuinely new PMIDs need fetching.
    articles: List[PubMedArticle] = []
    to_fetch: List[str] = []
    for pmid in pmids:
        entry = await cache.get_json("pubmed:abstract", pmid)
        if entry is None:
            to_fetch.append(pmid)
        elif entry.get("abstract"):
            articles.append(PubMedArticle(**entry))

    if articles:
        logger.info(
            "Abstract cache: %d/%d hit (%d to fetch).",
            len(articles), len(pmids), len(to_fetch),
        )

    if not to_fetch:
        return articles

    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
    try:
        resp = await client.get(
            f"{EUTILS_BASE}/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(to_fetch),
                "retmode": "xml",
                "rettype": "abstract",
            },
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
    except ET.ParseError as exc:
        logger.error("Could not parse PubMed efetch XML: %s", exc)
        return articles
    finally:
        if owns_client:
            await client.aclose()

    fetched: List[PubMedArticle] = []
    for element in root.findall(".//PubmedArticle"):
        parsed = _parse_article(element)
        if parsed:
            await cache.set_json("pubmed:abstract", parsed.pmid, asdict(parsed))
            if parsed.abstract:
                fetched.append(parsed)

    logger.info("Fetched %d abstracts for %d PMIDs", len(fetched), len(to_fetch))
    return articles + fetched


async def search_abstracts(query: str, max_results: int = 10) -> List[PubMedArticle]:
    """Convenience wrapper: esearch followed by efetch, on one connection."""
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        pmids = await search_pmids(query, max_results, client=client)
        return await fetch_abstracts(pmids, client=client)
