"""Streamlit app: the HR assistant (chat with confirmations), approvals, and policy Q&A.

Run with `hrm ui` (and `hrm serve` in another terminal). It only talks to the HTTP API
and keeps nothing but the login token and the open conversation id in the browser session.

Pages:
- Assistant: chat with the agent; actions that change data show Confirm / Decline.
- Approvals: leave requests waiting for the logged-in manager. Approve/Reject goes
  through the assistant, so it gets the same confirmation step and audit trail.
- Tickets: your HR tickets; HR admins also see the open queue. Opening and updating
  tickets goes through the assistant too. Ticket text is shown as plain text: employees
  write it, and as markdown it could load outside images or links.
- Policy Q&A: the public, cited policy search (no login).
"""

import uuid
from typing import get_args

import streamlit as st

from hrm.agents.runner import AgentReply
from hrm.api.chat_routes import UNTITLED
from hrm.core.config import get_settings
from hrm.db.models import Role
from hrm.rag.models import Category
from hrm.ui.client import ApiClient, ApiError
from hrm.workflows.tickets import TicketView

ASSISTANT, APPROVALS, TICKETS, POLICIES = "Assistant", "Approvals", "Tickets", "Policy Q&A"
STATUS_ICON = {"open": "🟡", "in_progress": "🔵", "closed": "✅"}
DEMO_USERS = "alice.park, hira.khan (employees) · bob.chen (manager) · carol.reyes (HR admin)"
CATEGORY_LABELS: dict[Category | None, str] = {
    None: "All policies",
    "time_off": "Time off",
    "parental_leave": "Parental leave",
    "benefits": "Benefits (US)",
    "expenses": "Expenses & equipment",
    "conduct": "Code of conduct",
}
assert set(CATEGORY_LABELS) - {None} == set(get_args(Category))


def main() -> None:
    st.set_page_config(page_title="HRM Copilot", page_icon="💬")
    page = sidebar()
    if page == POLICIES:
        policy_page()
    elif "token" not in st.session_state:
        login_page()
    elif page == ASSISTANT:
        assistant_page()
    elif page == TICKETS:
        tickets_page()
    else:
        approvals_page()


# --- layout --------------------------------------------------------------------------------


def sidebar() -> str:
    with st.sidebar:
        st.title("HRM Copilot")
        page = st.radio("Page", [ASSISTANT, APPROVALS, TICKETS, POLICIES], key="page")
        if "token" in st.session_state:
            me = st.session_state.me
            st.markdown(f"**{me.name}**  \n{me.job_title}")
            conversation_picker()
            st.button("Log out", on_click=logout)
    return str(page)


def conversation_picker() -> None:
    """Saved conversations come from the API on every run, so the list is the same after
    logging out and in, after a restart, and on another device. A new conversation is a
    local draft until its first message saves it."""
    try:
        titles = {c.conversation_id: c.title for c in api().conversations()}
    except ApiError as exc:
        show_error(exc)
        return
    draft = st.session_state.get("draft")
    if draft in titles:
        del st.session_state["draft"]
        draft = None
    if draft is None and not titles:
        draft = st.session_state.draft = _new_id()
    options = ([draft] if draft else []) + list(titles)
    current = st.session_state.get("conversation")
    if current not in options:
        current = options[0]
    st.session_state.conversation = st.selectbox(
        "Conversation",
        options,
        index=options.index(current),
        format_func=lambda c: titles.get(c, UNTITLED),
    )
    st.button("New conversation", on_click=new_conversation)
    with st.popover("Delete conversation"):
        st.caption("Deletes this conversation for good.")
        st.button("Delete", type="primary", on_click=delete_conversation)


def login_page() -> None:
    st.subheader("Log in")
    with st.form("login"):
        email = st.text_input("Email", placeholder="alice.park@example.com")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
    st.caption(f"Demo users (@example.com): {DEMO_USERS}")
    if submitted:
        try:
            login = api().login(email.strip(), password)
            st.session_state.me = api(login.token).me()
        except ApiError as exc:
            show_error(exc)
            return
        st.session_state.token = login.token
        st.session_state.role = login.role
        st.rerun()


# --- assistant -----------------------------------------------------------------------------


def assistant_page() -> None:
    conversation_id = st.session_state.get("conversation")
    if conversation_id is None:  # the conversation list could not be loaded
        return
    try:
        conversation = api().conversation(conversation_id)
    except ApiError as exc:
        show_error(exc)
        return

    if not conversation.messages:
        st.caption("Ask about policies, your leave, pay or expenses.")
    for turn in conversation.messages:
        with st.chat_message(turn.role):
            st.markdown(_md(turn.text))
    if conversation.pending:
        confirmation(conversation_id, conversation.pending)

    typed = st.chat_input("Your message")
    message = typed or st.session_state.pop("queued", None)
    if message:
        with st.spinner("Working on it..."):
            try:
                api().send(conversation_id, message)
            except ApiError as exc:
                show_error(exc)
                return
        st.rerun()


def confirmation(conversation_id: str, pending: AgentReply) -> None:
    with st.container(border=True):
        st.markdown("**Please confirm**")
        for action in pending.actions:
            st.markdown(f"- {_md(action.summary)}")
        confirm, decline = st.columns(2)
        decision = None
        if confirm.button("Confirm", type="primary", use_container_width=True):
            decision = True
        if decline.button("Decline", use_container_width=True):
            decision = False
        st.caption("Sending a new message instead cancels this action.")
    if decision is not None:
        with st.spinner("Working on it..."):
            try:
                api().confirm(conversation_id, decision)
            except ApiError as exc:
                show_error(exc)
                return
        st.rerun()


# --- approvals -----------------------------------------------------------------------------


def approvals_page() -> None:
    st.subheader("Waiting for your approval")
    try:
        requests = api().approvals()
        names = {e.id: e.name for e in api().team()}
    except ApiError as exc:
        show_error(exc)
        return
    if not requests:
        st.info("Nothing is waiting for your approval.")
        return

    st.caption("Approve or reject through the assistant; you will confirm before it is saved.")
    for r in requests:
        with st.container(border=True):
            st.markdown(
                f"**{names.get(r.employee_id, f'Employee #{r.employee_id}')}** · "
                f"leave request #{r.id} · {r.leave_type}  \n"
                f"{r.start_date} → {r.end_date} · {r.working_days:g} working days"
            )
            if r.reason:
                st.caption(r.reason)
            approve, reject = st.columns(2)
            approve.button(
                "Approve",
                key=f"approve-{r.id}",
                on_click=ask_assistant,
                args=(f"Approve leave request #{r.id}.",),
                use_container_width=True,
            )
            reject.button(
                "Reject",
                key=f"reject-{r.id}",
                on_click=ask_assistant,
                args=(f"Reject leave request #{r.id}.",),
                use_container_width=True,
            )


def ask_assistant(message: str) -> None:
    """Button callback: runs before the next rerun, so it may switch the page widget."""
    st.session_state.queued = message
    st.session_state.page = ASSISTANT


def new_conversation() -> None:
    st.session_state.draft = st.session_state.conversation = _new_id()


def delete_conversation() -> None:
    current = st.session_state.pop("conversation", None)
    if current == st.session_state.get("draft"):
        del st.session_state["draft"]
    elif current:
        try:
            api().delete_conversation(current)
        except ApiError as exc:
            show_error(exc)


# --- tickets -------------------------------------------------------------------------------


def tickets_page() -> None:
    hr_admin = st.session_state.get("role") == Role.HR_ADMIN
    try:
        mine = api().my_tickets()
        if hr_admin:
            queue = api().ticket_queue()
            names = {e.id: e.name for e in api().all_employees()}
        else:
            queue, names = [], {}
    except ApiError as exc:
        show_error(exc)
        return

    st.subheader("My tickets")
    st.button(
        "Open a ticket with the assistant",
        on_click=ask_assistant,
        args=("I'd like to open an HR ticket.",),
    )
    if not mine:
        st.info("You have no HR tickets.")
    for t in mine:
        ticket_card(t)

    if hr_admin:
        st.subheader("Open tickets (HR)")
        if not queue:
            st.info("The queue is empty.")
        for t in queue:
            ticket_card(t, employee_name=names.get(t.employee_id))
            if t.status == "open":
                st.button(
                    "Start working on it",
                    key=f"start-{t.id}",
                    on_click=ask_assistant,
                    args=(f"Set HR ticket #{t.id} to in progress.",),
                )
        st.caption("Close tickets through the assistant: it asks for the resolution.")


def ticket_card(t: TicketView, *, employee_name: str | None = None) -> None:
    with st.container(border=True):
        header = f"{STATUS_ICON[t.status]} **#{t.id}** · {t.category} · {t.status}"
        if employee_name:
            header += f" · {employee_name}"
        st.markdown(header)
        st.text(t.subject)  # plain text: written by an employee
        st.text(t.body)
        if t.resolution:
            st.markdown("**Resolution**")
            st.text(t.resolution)


# --- policy Q&A (public) -------------------------------------------------------------------


def policy_page() -> None:
    st.subheader("Policy Q&A")
    st.caption("Answers from Sourcegraph's HR policies, with sources. Not legal advice.")
    with st.form("ask"):
        question = st.text_input(
            "Your question", placeholder="How much notice do I give before PTO?"
        )
        category = st.selectbox(
            "Policy area", list(CATEGORY_LABELS), format_func=lambda c: CATEGORY_LABELS[c]
        )
        submitted = st.form_submit_button("Ask")
    if not (submitted and question.strip()):
        return

    with st.spinner("Searching the policies..."):
        try:
            result = api().ask(question.strip(), category)
        except ApiError as exc:
            show_error(exc)
            return
    if result.found:
        st.markdown(_md(result.answer))
        with st.expander(f"Sources ({len(result.citations)})", expanded=True):
            for c in result.citations:
                st.markdown(f"**[{c.number}] {c.doc_id}** · {c.section}")
                st.caption(c.quote)
    else:
        st.info(result.answer)
    st.caption(
        f"Retrieval {result.timings.retrieve_ms} ms · answer {result.timings.generate_ms} ms "
        f"· total {result.timings.total_ms} ms"
    )


# --- helpers -------------------------------------------------------------------------------


def api(token: str | None = None) -> ApiClient:
    return ApiClient(get_settings().api_url, token=token or st.session_state.get("token"))


def show_error(exc: ApiError) -> None:
    if exc.login_expired and "token" in st.session_state:
        logout()
        st.warning("Your login has expired. Please log in again.")
        return
    st.error(str(exc))
    if exc.request_id:
        st.caption(f"Request id: {exc.request_id}")


def logout() -> None:
    # Only this browser's session state: the conversations themselves stay on the server.
    for key in ("token", "role", "me", "conversation", "draft", "queued"):
        st.session_state.pop(key, None)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _md(text: str) -> str:
    """Streamlit renders $...$ as LaTeX; "$234 and $1,000" must stay money."""
    return text.replace("$", r"\$")


main()
