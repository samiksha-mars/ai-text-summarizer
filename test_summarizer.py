"""Tests use a fake client, so they run without an API key or the anthropic package."""

import io
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from summarizer import (MODEL, SummaryRefused, build_prompt, has_credentials, main,
                        read_input, summarize)

SAMPLE = str(Path(__file__).with_name("sample_article.txt"))


class FakeClient:
    def __init__(self, text="- point one", stop_reason="end_turn"):
        self.calls = []
        self.response = types.SimpleNamespace(
            stop_reason=stop_reason,
            content=[types.SimpleNamespace(type="thinking", thinking=""),
                     types.SimpleNamespace(type="text", text=text)],
        )
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class BuildPromptTest(unittest.TestCase):

    def test_wraps_document_and_adds_instruction(self):
        prompt = build_prompt("Hello world", "tldr")
        self.assertIn("<document>\nHello world\n</document>", prompt)
        self.assertIn("TL;DR", prompt)

    def test_word_limit(self):
        self.assertIn("at most 50 words", build_prompt("text", "paragraph", 50))

    def test_unknown_style(self):
        with self.assertRaises(ValueError):
            build_prompt("text", "haiku")

    def test_empty_text(self):
        with self.assertRaises(ValueError):
            build_prompt("   \n", "bullets")

    def test_non_positive_word_limit(self):
        for limit in (0, -5):
            with self.assertRaises(ValueError):
                build_prompt("text", "paragraph", limit)

    def test_cli_rejects_bad_word_limit_before_calling_api(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            self.assertEqual(main([SAMPLE, "--max-words", "0"]), 1)
        self.assertIn("--max-words", stderr.getvalue())


class SummarizeTest(unittest.TestCase):

    def test_returns_text_blocks_only(self):
        self.assertEqual(summarize(FakeClient("- a\n- b\n"), "doc"), "- a\n- b")

    def test_request_parameters(self):
        client = FakeClient()
        summarize(client, "doc", style="paragraph")
        call = client.calls[0]
        self.assertEqual(call["model"], MODEL)
        self.assertEqual(call["messages"][0]["role"], "user")
        self.assertIn("one clear", call["messages"][0]["content"])
        self.assertNotIn("temperature", call)
        self.assertNotIn("thinking", call)

    def test_refusal_raises(self):
        with self.assertRaises(SummaryRefused):
            summarize(FakeClient("", stop_reason="refusal"), "doc")


class ReadInputTest(unittest.TestCase):

    def test_reads_stdin(self):
        self.assertEqual(read_input(None, stdin=io.StringIO("piped")), "piped")

    def test_reads_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("from file")
        self.assertEqual(read_input(f.name), "from file")


class CredentialsTest(unittest.TestCase):

    def test_has_credentials(self):
        self.assertTrue(has_credentials(types.SimpleNamespace(api_key="sk-test")))
        self.assertTrue(has_credentials(types.SimpleNamespace(api_key=None, credentials=object())))
        self.assertFalse(has_credentials(types.SimpleNamespace(api_key=None, auth_token=None)))

    def test_missing_credentials_exit_without_calling_api(self):
        client = FakeClient()
        client.api_key = client.auth_token = client.credentials = None
        fake_anthropic = types.SimpleNamespace(Anthropic=lambda: client)
        stderr = io.StringIO()
        with mock.patch.dict(sys.modules, anthropic=fake_anthropic), redirect_stderr(stderr):
            self.assertEqual(main([SAMPLE]), 1)
        self.assertIn("ANTHROPIC_API_KEY", stderr.getvalue())
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
