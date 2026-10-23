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
    return f"{thread_id_prefix(user)}{conversation_id}"


class ConversationStore:
    """Which conversations a user has, newest first, and deleting one. Conversations stay
    in the checkpointer until the user deletes them; logging out does not touch them."""

    def __init__(self, saver: SqliteSaver) -> None:
        self._saver = saver

    def conversation_ids(self, user: UserContext) -> list[str]:
        prefix = thread_id_prefix(user)
        # Checkpoint ids are time-ordered (uuid6), so the largest one is the latest activity.
        with self._saver.cursor(transaction=False) as cur:
            rows = cur.execute(
                "SELECT thread_id FROM checkpoints"
                " WHERE checkpoint_ns = '' AND substr(thread_id, 1, ?) = ?"
                " GROUP BY thread_id ORDER BY MAX(checkpoint_id) DESC",
                (len(prefix), prefix),
            ).fetchall()
        return [row[0][len(prefix) :] for row in rows]

    def delete(self, user: UserContext, conversation_id: str) -> None:
        self._saver.delete_thread(thread_id(user, conversation_id))


def thread_id_prefix(user: UserContext) -> str:
    return f"user-{user.user_id}:"


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
