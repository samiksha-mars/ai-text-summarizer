"""Summarize text with Claude."""

import argparse
import sys

MODEL = "claude-opus-5-5"
MAX_TOKENS = 16000
SYSTEM_PROMPT = (
    "You summarize documents accurately. Only use information that appears in the "
    "document, keep names and numbers exact, and reply with the summary only."
)
STYLES = {
    "bullets": "Summarize the document as 3-7 concise bullet points, each starting with '- '.",
    "paragraph": "Summarize the document in one clear, well-structured paragraph.",
    "tldr": "Summarize the document in a single sentence starting with 'TL;DR:'.",
}


class SummaryRefused(Exception):
    """Raised when Claude declines to summarize the text."""


def build_prompt(text, style="bullets", max_words=None):
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}, choose from {', '.join(STYLES)}")
    if not text.strip():
        raise ValueError("there is no text to summarize")
    instruction = STYLES[style]
    if max_words:
        instruction += f" Use at most {max_words} words."
    return f"<document>\n{text.strip()}\n</document>\n\n{instruction}"


def summarize(client, text, style="bullets", max_words=None):
    """Return Claude's summary of `text`."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(text, style, max_words)}],
        output_config={"effort": "low"},
    )
    if response.stop_reason == "refusal":
        raise SummaryRefused("Claude declined to summarize this text.")
    return "".join(block.text for block in response.content if block.type == "text").strip()


def read_input(path, stdin=sys.stdin):
    if path in (None, "-"):
        return stdin.read()
    with open(path, encoding="utf-8") as f:
        return f.read()
