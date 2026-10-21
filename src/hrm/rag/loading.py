"""Read the policy documents listed in catalog.yaml into SourceDocument objects."""

import hashlib
import re
from datetime import date, datetime
from pathlib import Path

import yaml
from pydantic import ValidationError

from hrm.core.errors import DocumentLoadError
from hrm.core.logging import get_logger
from hrm.rag.models import CatalogEntry, SourceDocument

log = get_logger(__name__)

CATALOG_FILE = "catalog.yaml"

_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_LAST_UPDATED = re.compile(r"^Last updated:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)


def load_catalog(docs_dir: Path) -> list[CatalogEntry]:
    """Read and validate the catalog; it must list every .md file in docs_dir exactly once."""
    path = docs_dir / CATALOG_FILE
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        entries = [CatalogEntry.model_validate(item) for item in raw["documents"]]
    except FileNotFoundError as exc:
        raise DocumentLoadError(f"Catalog not found: {path}") from exc
    except (yaml.YAMLError, KeyError, TypeError, ValidationError) as exc:
        raise DocumentLoadError(f"Invalid catalog {path}: {exc}") from exc

    doc_ids = [entry.doc_id for entry in entries]
    if len(doc_ids) != len(set(doc_ids)):
        raise DocumentLoadError(f"Duplicate doc_id in {path}")

    listed = sorted(entry.file for entry in entries)
    on_disk = sorted(p.name for p in docs_dir.glob("*.md"))
    if listed != on_disk:
        missing = sorted(set(on_disk) - set(listed))
        unknown = sorted(set(listed) - set(on_disk))
        raise DocumentLoadError(
            f"Catalog out of sync with {docs_dir}: not in catalog={missing}, no such file={unknown}"
        )
    return entries


def load_documents(docs_dir: Path) -> list[SourceDocument]:
    """Load every catalogued document, in catalog order."""
    documents = [_load_one(docs_dir, entry) for entry in load_catalog(docs_dir)]
    log.info("documents.loaded", count=len(documents), docs_dir=str(docs_dir))
    return documents


def _load_one(docs_dir: Path, entry: CatalogEntry) -> SourceDocument:
    data = (docs_dir / entry.file).read_bytes()
    text = data.decode("utf-8")
    return SourceDocument(
        doc_id=entry.doc_id,
        title=_extract_title(text, entry.file),
        category=entry.category,
        audience=entry.audience,
        source_path=entry.file,
        content_hash=hashlib.sha256(data).hexdigest(),
        last_updated=_extract_last_updated(text, entry.file),
        text=text,
    )


def _extract_title(text: str, file: str) -> str:
    match = _H1.search(text)
    if match is None:
        raise DocumentLoadError(f"{file}: no '# Title' heading found")
    return match.group(1).strip("* ")  # tolerate a bold title: # **Title**


def _extract_last_updated(text: str, file: str) -> date | None:
    match = _LAST_UPDATED.search(text)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%B %d, %Y").date()  # "April 29, 2022"
    except ValueError:
        # A date we can't read shouldn't block ingestion; it is only informational.
        log.warning("documents.bad_last_updated", file=file, value=match.group(1))
        return None
