"""Where conversations are kept, and who may open them.

Thread ids are always built from the logged-in user plus a conversation id the client
chooses. A client never sends a raw thread id, so it cannot even name another user's
conversation: ownership holds by construction, not by a check someone might forget.
"""

import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from hrm.auth.context import UserContext
from hrm.core.errors import HRMError

CONVERSATION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
_CONVERSATION_ID = re.compile(CONVERSATION_ID_PATTERN)


class InvalidConversationError(HRMError):
    """The conversation id has characters or a length we do not accept."""


def thread_id(user: UserContext, conversation_id: str) -> str:
    if not _CONVERSATION_ID.fullmatch(conversation_id):
        raise InvalidConversationError("Conversation ids are 1-64 letters, digits, '-' or '_'.")
    return f"user-{user.user_id}:{conversation_id}"


@contextmanager
def open_memory(path: Path) -> Iterator[SqliteSaver]:
    """A checkpointer backed by one SQLite file, separate from the HR database.

    check_same_thread=False because API requests run in worker threads; SqliteSaver
    serialises access to the connection with its own lock.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    try:
        yield SqliteSaver(conn)
    finally:
        conn.close()
