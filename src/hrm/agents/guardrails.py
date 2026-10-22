"""Input and output guardrails around the agent: plain functions, no LLM calls.

These are defence in depth, not the security boundary. Identity, role filtering and
permissions are enforced in code below the LLM (see tools.py), so even an injection that
slips past these checks cannot read another person's data or skip a confirmation.
What these checks add:
- input: refuse empty or oversized messages and obvious prompt-injection attempts before
  any tokens are spent, with a clear message instead of a confused agent;
- output: never show secrets (API keys, tokens, password hashes) or the hidden system
  prompt, whatever the model was tricked into writing.

Deliberately NOT checked here:
- "off-topic" questions: a keyword filter refuses real HR questions; the system prompt
  scopes the agent and the policy tool answers "not covered" instead;
- "I'm Carol" style impersonation: identity never comes from the chat, so it is harmless,
  and blocking "I am a manager" would refuse legitimate questions.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

MAX_INPUT_CHARS = 2_000
MAX_OUTPUT_CHARS = 6_000
# Placed in the system prompt; if it ever appears in an answer, the prompt is leaking.
PROMPT_CANARY = "hrm-canary-5f1c9e"

InputBlock = Literal["empty", "too_long", "injection"]

_INVISIBLE = re.compile(r"[​-‏⁠﻿]")  # zero-width chars used to split words
_INJECTION = re.compile(
    r"""
      \b(ignore|disregard|forget|override)\b .{0,30}
        \b(previous|prior|above|earlier|all|your|the)\b .{0,20}
        \b(instructions?|rules|prompts?|guidelines|directions)\b
    | \b(system|developer|hidden)\s+(prompt|message|instructions?)\b
    | \b(reveal|show|print|repeat|leak)\b .{0,20} \byour\s+(instructions|prompt|rules)\b
    | \byou\s+are\s+now\b
    | \b(developer|dan|god|jailbreak)\s+mode\b
    | \bjailbreak
    | \bpretend\s+(to\s+be|you\s+are)\b
    | \bnew\s+instructions?\s*:
    | </?\s*(system|assistant|tool)\s*>
    """,
    re.IGNORECASE | re.VERBOSE,
)
_SECRETS = [
    re.compile(r"\bgsk_[A-Za-z0-9]{20,}"),  # Groq API key
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),  # OpenAI-style API key
    re.compile(r"\beyJ[\w-]{10,}\.[\w-]{10,}\.[\w-]{10,}"),  # JWT
    re.compile(r"scrypt\$\d+\$\d+\$\d+\$\S+"),  # our password hash format
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(-----END [A-Z ]*PRIVATE KEY-----|$)"),
]
_REDACTED = "[redacted]"

INPUT_REFUSALS: dict[InputBlock, str] = {
    "empty": "Please type a question or request.",
    "too_long": f"That message is too long; please keep it under {MAX_INPUT_CHARS} characters.",
    "injection": (
        "I can't follow instructions that try to change how I work. "
        "Ask me about HR policies, your leave, pay or expenses."
    ),
}
PROMPT_LEAK_REPLY = "Sorry, I can't share that. How can I help with an HR question?"


@dataclass(frozen=True)
class InputCheck:
    text: str  # normalised message to send to the agent
    blocked: InputBlock | None = None

    @property
    def ok(self) -> bool:
        return self.blocked is None

    @property
    def refusal(self) -> str | None:
        return INPUT_REFUSALS[self.blocked] if self.blocked else None


@dataclass(frozen=True)
class OutputCheck:
    text: str  # what the user may see
    findings: list[str] = field(default_factory=list)  # what was removed, for the audit log

    @property
    def changed(self) -> bool:
        return bool(self.findings)


def check_input(message: str) -> InputCheck:
    """Normalise a user message and decide whether it may reach the agent."""
    text = normalise(message)
    if not text:
        return InputCheck(text, "empty")
    if len(text) > MAX_INPUT_CHARS:
        return InputCheck(text, "too_long")
    if looks_like_injection(text):
        return InputCheck(text, "injection")
    return InputCheck(text)


def check_output(answer: str) -> OutputCheck:
    """Make an agent answer safe to show: no prompt leak, no secrets, bounded length."""
    if PROMPT_CANARY in answer:
        return OutputCheck(PROMPT_LEAK_REPLY, ["prompt_leak"])
    findings: list[str] = []
    for pattern in _SECRETS:
        answer, count = pattern.subn(_REDACTED, answer)
        if count:
            findings.append("secret")
    if len(answer) > MAX_OUTPUT_CHARS:
        answer = answer[:MAX_OUTPUT_CHARS].rstrip() + " …"
        findings.append("truncated")
    return OutputCheck(answer.strip(), findings)


def looks_like_injection(text: str) -> bool:
    """Also used on text the agent reads from tools, which other people may have written."""
    return bool(_INJECTION.search(normalise(text)))


def normalise(text: str) -> str:
    """NFKC folds look-alike characters (full-width letters, ligatures) into plain ones and
    zero-width characters are dropped, so "ign​ore" cannot dodge the patterns."""
    return _INVISIBLE.sub("", unicodedata.normalize("NFKC", text)).strip()
