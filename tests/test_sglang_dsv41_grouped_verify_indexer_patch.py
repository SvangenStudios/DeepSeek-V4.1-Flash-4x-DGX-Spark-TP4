#!/usr/bin/env python3
"""Packaging checks for the SGLang #45251 pinned-tree adaptation."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/sglang-45251-grouped-verify-indexer.patch"


class Dsv41GroupedVerifyIndexerPackagingTests(unittest.TestCase):
    def test_patch_is_content_pinned_and_zero_fuzz(self):
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        digest = hashlib.sha256(PATCH.read_bytes()).hexdigest()
        self.assertIn("476fe86a12535173ea0eee1e849f80e350cc6ffd", fetch)
        self.assertIn(digest, fetch)
        self.assertIn("sglang-45251-grouped-verify-indexer.patch", fetch)
        self.assertIn("--fuzz=0", fetch)

    def test_images_verify_grouped_layout(self):
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            with self.subTest(name=name):
                source = (ROOT / name).read_text()
                self.assertIn(
                    "verify_sglang_dsv41_grouped_verify_indexer.py", source
                )
                self.assertIn("test_sglang_dsv41_grouped_verify_indexer.py", source)
                self.assertIn(
                    "test_sglang_dsv41_grouped_verify_indexer_patch.py", source
                )


if __name__ == "__main__":
    unittest.main()
