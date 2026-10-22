#!/usr/bin/env python3
"""Fail-closed source checks for the SGLang #45251 pinned-tree adaptation."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise SystemExit(f"missing {label}: {needle!r}")


def main() -> None:
    root = Path(sys.argv[1])
    metadata = (root / "sglang/srt/layers/attention/dsv4/metadata.py").read_text()
    indexer = (root / "sglang/srt/layers/attention/dsv4/indexer.py").read_text()
    candidate = (
        root / "sglang/srt/layers/attention/dsv4/candidate_deep_gemm.py"
    ).read_text()
    backend = (root / "sglang/srt/layers/attention/deepseek_v4_backend.py").read_text()

    for path in (
        root / "sglang/srt/layers/attention/dsv4/metadata.py",
        root / "sglang/srt/layers/attention/dsv4/candidate_deep_gemm.py",
        root / "sglang/srt/layers/attention/deepseek_v4_backend.py",
    ):
        ast.parse(path.read_text(), filename=str(path))

    require(metadata, "rows_per_request: int = 1", "group default")
    require(metadata, "and group == 1", "JIT-planner safety gate")
    require(metadata, ".view(-1, group)", "grouped planner lengths")
    require(
        metadata,
        "chunk = _SM120_INDEXER_M_CHUNK // group * group",
        "group-aligned SM120 chunks",
    )
    require(metadata, "def paged_logits_args(", "grouping helper")
    require(metadata, "q_data.view(-1, n, heads, width)", "grouped query layout")
    require(metadata, "q_sf.view(-1, n, heads)", "grouped scale layout")
    require(metadata, "table = table[::n]", "one page-table row per request")
    require(metadata, '"rows_per_request",', "graph-copy invariant")

    require(
        backend,
        "rows_per_request = 1 if is_ragged else num_draft_tokens",
        "static/ragged verify gate",
    )
    if backend.count("rows_per_request=rows_per_request") < 3:
        raise SystemExit("both low-ratio indexers must receive the verify row group")
    if backend.count("metadata.paged_logits_args(") != 1:
        raise SystemExit("the direct low-ratio full scan must group exactly once")
    require(candidate, "metadata.paged_logits_args(", "candidate-source full scan grouping")
    if "paged_logits_args(" in indexer:
        raise SystemExit("C4 indexer must retain its one-row layout")

    print("SGLang #45251 grouped static-verify indexer scans: verified")


if __name__ == "__main__":
    main()
