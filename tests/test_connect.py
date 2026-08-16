"""Regression for the Aug 16 preflight failure: the first Day 2 run reported
every query shape as unmeasured because the client was built without a token
and the node answered 401. The shapes were never exercised at all. Any tool
that talks to this node must build its client the same way."""
from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_TOKEN = "local-development-token-32-bytes"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Connect(unittest.TestCase):
    def test_every_tool_defaults_to_the_same_token(self):
        blank = argparse.Namespace(base="", token="")
        for name in ("preflight2", "answer_run"):
            client = load(name).connect(blank)
            self.assertEqual(client.token, DEFAULT_TOKEN, name)
            self.assertEqual(client.base_url, "http://127.0.0.1:8443", name)
            self.assertEqual(client.consistency, "causal", name)

    def test_an_explicit_token_wins_over_the_default(self):
        args = argparse.Namespace(base="http://elsewhere:9999", token="real")
        for name in ("preflight2", "answer_run"):
            client = load(name).connect(args)
            self.assertEqual(client.token, "real", name)
            self.assertEqual(client.base_url, "http://elsewhere:9999", name)

    def test_the_day_one_tools_use_that_same_default(self):
        # If ingest_run ever changes its fallback, this catches the drift
        # rather than letting two tools disagree about who they are.
        text = (ROOT / "tools" / "ingest_run.py").read_text()
        self.assertIn(DEFAULT_TOKEN, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class RunShape(unittest.TestCase):
    """Regression for the Aug 16 preflight bug: the probed value must travel
    back through the return, not be captured by the caller. A walrus inside a
    lambda binds in the lambda's scope, so the outer name kept its initial
    empty value and a full graph was reported as empty."""

    def test_the_probed_value_comes_back_in_the_result(self):
        pf = load("preflight2")
        out = pf.run_shape("x", lambda: {"a": 1, "b": 2})
        self.assertEqual(out["verdict"], "supported")
        self.assertEqual(out["sample"], {"a": 1, "b": 2})

    def test_preflight_does_not_capture_by_walrus_in_a_lambda(self):
        text = (ROOT / "tools" / "preflight2.py").read_text()
        for line in text.splitlines():
            code = line.split("#", 1)[0]      # the comment explaining the bug
            if "lambda" in code:              # is allowed to contain one
                self.assertNotIn(":=", code, f"walrus in a lambda: {code.strip()}")
