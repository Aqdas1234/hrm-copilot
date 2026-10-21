"""The Streamlit page, run headless with Streamlit's AppTest; the API client is faked."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from hrm.api.schemas import AskResponse
from hrm.ui.client import ApiError
from tests.test_ui_client import ASK_OK

PAGE = Path(__file__).parents[1] / "src" / "hrm" / "ui" / "streamlit_app.py"


def run_page(monkeypatch: pytest.MonkeyPatch, reply: AskResponse | ApiError) -> AppTest:
    def fake_ask(self: object, question: str, category: str | None = None) -> AskResponse:
        if isinstance(reply, ApiError):
            raise reply
        return reply

    monkeypatch.setattr("hrm.ui.client.ApiClient.ask", fake_ask)
    page = AppTest.from_file(str(PAGE)).run()
    page.text_input[0].input("Within how many days do I file an expense report?")
    page.button[0].click()
    return page.run()


def test_answer_and_sources_are_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    page = run_page(monkeypatch, AskResponse.model_validate(ASK_OK))

    assert not page.exception
    assert any("File within 60 days [1]." in m.value for m in page.markdown)
    assert any("home-office-and-stipends" in m.value for m in page.markdown)
    assert any("within 60 days" in c.value for c in page.caption)


def test_not_covered_is_shown_as_info(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = AskResponse.model_validate(
        {**ASK_OK, "answer": "Not covered.", "found": False, "citations": []}
    )

    page = run_page(monkeypatch, reply)

    assert [i.value for i in page.info] == ["Not covered."]


def test_api_errors_show_the_message_and_request_id(monkeypatch: pytest.MonkeyPatch) -> None:
    page = run_page(monkeypatch, ApiError("Service temporarily unavailable.", "req-42"))

    assert [e.value for e in page.error] == ["Service temporarily unavailable."]
    assert any("req-42" in c.value for c in page.caption)
