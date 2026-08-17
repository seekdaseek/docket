"""The README is the front door, and it was hiding the best things in here.

Nine tools were undocumented, including the generated capability map, the
browsable register and the as-of demo — the three most worth finding. A reader
of the README had no way to learn they existed. This test fails if a tool is
added without a mention, so it cannot happen again by accident.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReadmeDocumentsTheRepo(unittest.TestCase):
    def setUp(self):
        self.readme = (ROOT / "README.md").read_text()

    def test_every_tool_is_mentioned(self):
        undocumented = [n for n in sorted(os.listdir(ROOT / "tools"))
                        if n.endswith(".py") and n not in self.readme]
        self.assertEqual([], undocumented,
                         "tools missing from README.md: %s" % undocumented)

    def test_the_generated_artifacts_are_linked(self):
        for name in ("HYDRADB-CAPABILITIES.md", "RESULTS.md", "inspector.html"):
            self.assertIn(name, self.readme, "%s not in README" % name)

    def test_the_regeneration_commands_are_present(self):
        # A generated artifact a reader cannot regenerate is just a claim.
        for cmd in ("tools/capabilities.py", "tools/report.py",
                    "tools/inspector.py"):
            self.assertIn(cmd, self.readme)

    def test_no_stale_embeddings_claim(self):
        # The gate is BM25 and EmbeddingGate raises. This exact sentence was
        # in the opening section for days, contradicting the same file ninety
        # lines further down.
        self.assertNotIn("Embeddings are computed\noutside the database",
                         self.readme)
