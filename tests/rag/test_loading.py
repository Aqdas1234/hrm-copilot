import hashlib
from datetime import date
from pathlib import Path

import pytest

from hrm.core.config import PROJECT_ROOT
from hrm.core.errors import DocumentLoadError
from hrm.rag.loading import load_catalog, load_documents

DOCS_DIR = PROJECT_ROOT / "data" / "docs"

# --- the real documents -------------------------------------------------------------------


def test_real_catalog_is_valid_and_in_sync() -> None:
    entries = load_catalog(DOCS_DIR)

    assert len(entries) == 5


def test_loads_all_five_documents_with_titles_and_hashes() -> None:
    docs = {doc.doc_id: doc for doc in load_documents(DOCS_DIR)}

    assert {doc_id: doc.title for doc_id, doc in docs.items()} == {
        "benefits-overview": "Health benefits at Sourcegraph",
        "code-of-conduct": "Code of conduct",
        "home-office-and-stipends": "Spending company money",
        "parental-leave": "Sourcegraph Paid Parental Leave Policy",
        "time-off-policy": "Paid time off (PTO) policy",
    }
    for doc in docs.values():
        expected = hashlib.sha256((DOCS_DIR / doc.source_path).read_bytes()).hexdigest()
        assert doc.content_hash == expected
        assert doc.text


def test_last_updated_is_parsed_only_where_present() -> None:
    docs = {doc.doc_id: doc for doc in load_documents(DOCS_DIR)}

    assert docs["code-of-conduct"].last_updated == date(2022, 4, 29)
    assert docs["parental-leave"].last_updated is None


# --- error cases on a temporary docs folder -----------------------------------------------


def write_docs(tmp_path: Path, catalog: str, files: dict[str, str]) -> Path:
    (tmp_path / "catalog.yaml").write_text(catalog, encoding="utf-8")
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


ONE_ENTRY = """
documents:
  - {file: a.md, doc_id: doc-a, category: conduct, audience: all}
"""


def test_file_missing_from_catalog_is_rejected(tmp_path: Path) -> None:
    docs_dir = write_docs(tmp_path, ONE_ENTRY, {"a.md": "# A\n", "b.md": "# B\n"})

    with pytest.raises(DocumentLoadError, match=r"not in catalog=\['b.md'\]"):
        load_catalog(docs_dir)


def test_catalog_entry_without_file_is_rejected(tmp_path: Path) -> None:
    docs_dir = write_docs(tmp_path, ONE_ENTRY, {})

    with pytest.raises(DocumentLoadError, match=r"no such file=\['a.md'\]"):
        load_catalog(docs_dir)


@pytest.mark.parametrize(
    "bad_field",
    [
        "category: payroll",  # unknown category
        "audience: eu",  # unknown audience
        "doc_id: Doc A",  # not a slug
        "owner: hr",  # unexpected extra field
    ],
)
def test_invalid_entry_is_rejected(tmp_path: Path, bad_field: str) -> None:
    key = bad_field.split(":")[0]
    fields = {"file": "a.md", "doc_id": "doc-a", "category": "conduct", "audience": "all"}
    fields[key] = bad_field.split(": ")[1]
    entry = ", ".join(f"{k}: {v}" for k, v in fields.items())
    docs_dir = write_docs(tmp_path, f"documents:\n  - {{{entry}}}\n", {"a.md": "# A\n"})

    with pytest.raises(DocumentLoadError, match="Invalid catalog"):
        load_catalog(docs_dir)


def test_duplicate_doc_id_is_rejected(tmp_path: Path) -> None:
    catalog = """
documents:
  - {file: a.md, doc_id: same, category: conduct, audience: all}
  - {file: b.md, doc_id: same, category: conduct, audience: all}
"""
    docs_dir = write_docs(tmp_path, catalog, {"a.md": "# A\n", "b.md": "# B\n"})

    with pytest.raises(DocumentLoadError, match="Duplicate doc_id"):
        load_catalog(docs_dir)


def test_document_without_h1_title_is_rejected(tmp_path: Path) -> None:
    docs_dir = write_docs(tmp_path, ONE_ENTRY, {"a.md": "## Only a subheading\n"})

    with pytest.raises(DocumentLoadError, match="no '# Title'"):
        load_documents(docs_dir)


def test_bold_title_and_unreadable_date_are_handled(tmp_path: Path) -> None:
    text = "# **Bold title**\n\nLast updated: sometime in 2023\n"
    docs_dir = write_docs(tmp_path, ONE_ENTRY, {"a.md": text})

    [doc] = load_documents(docs_dir)

    assert doc.title == "Bold title"
    assert doc.last_updated is None
