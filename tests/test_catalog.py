"""The document catalog must stay in sync with the files in data/docs."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from hrm.core.config import PROJECT_ROOT

DOCS_DIR = PROJECT_ROOT / "data" / "docs"
CATEGORIES = {"benefits", "conduct", "expenses", "parental_leave", "time_off"}
AUDIENCES = {"all", "us"}


@pytest.fixture(scope="module")
def entries() -> list[dict[str, Any]]:
    catalog = yaml.safe_load((DOCS_DIR / "catalog.yaml").read_text(encoding="utf-8"))
    documents: list[dict[str, Any]] = catalog["documents"]
    return documents


def test_every_markdown_file_is_listed_exactly_once(entries: list[dict[str, Any]]) -> None:
    listed = [entry["file"] for entry in entries]
    on_disk = sorted(path.name for path in DOCS_DIR.glob("*.md"))

    assert sorted(listed) == on_disk


def test_entries_have_exactly_the_expected_fields(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        assert set(entry) == {"file", "doc_id", "category", "audience"}, entry


def test_doc_ids_are_unique_slugs(entries: list[dict[str, Any]]) -> None:
    doc_ids = [entry["doc_id"] for entry in entries]

    assert len(doc_ids) == len(set(doc_ids))
    for doc_id in doc_ids:
        assert doc_id == doc_id.lower() and " " not in doc_id, doc_id


def test_categories_and_audiences_are_known_values(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        assert entry["category"] in CATEGORIES, entry
        assert entry["audience"] in AUDIENCES, entry


def test_listed_files_exist(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        assert Path(DOCS_DIR / entry["file"]).is_file(), entry["file"]
