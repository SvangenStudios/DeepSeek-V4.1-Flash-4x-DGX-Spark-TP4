#!/usr/bin/env python3
"""CPU behavior test for the production #45251 query-layout helper."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace


class FakeTensor:
    def __init__(self, shape, rows=None):
        self.shape = tuple(shape)
        self.rows = list(range(self.shape[0])) if rows is None else list(rows)

    def __getitem__(self, item):
        if not isinstance(item, slice):
            raise TypeError(item)
        rows = self.rows[item]
        return FakeTensor((len(rows), *self.shape[1:]), rows)

    def view(self, *shape):
        known = 1
        infer = None
        for index, size in enumerate(shape):
            if size == -1:
                assert infer is None
                infer = index
            else:
                known *= size
        total = 1
        for size in self.shape:
            total *= size
        shape = list(shape)
        if infer is not None:
            assert total % known == 0
            shape[infer] = total // known
        product = 1
        for size in shape:
            product *= size
        assert product == total
        return FakeTensor(shape, self.rows)


def load_helper(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PagedIndexerMetadata")
    fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "paged_logits_args")
    ast.fix_missing_locations(fn)
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["paged_logits_args"]


def main() -> None:
    root = Path(sys.argv[1])
    helper = load_helper(root / "sglang/srt/layers/attention/dsv4/metadata.py")
    payload = FakeTensor((18, 1, 2, 64))
    scales = FakeTensor((18, 1, 2))
    lens = FakeTensor((18, 1))
    table = FakeTensor((18, 4))

    grouped = SimpleNamespace(
        rows_per_request=6, compressed_seq_lens=lens, page_table=table
    )
    (q, sf), out_lens, out_table = helper(
        grouped, (payload, scales), slice(6, 18)
    )
    assert q.shape == (2, 6, 2, 64)
    assert sf.shape == (2, 6, 2)
    assert out_lens.shape == (2, 6)
    assert out_table.shape == (2, 4)
    assert out_table.rows == [6, 12]

    single = SimpleNamespace(
        rows_per_request=1, compressed_seq_lens=lens, page_table=table
    )
    q_single, lens_single, table_single = helper(single, payload)
    assert q_single.shape == payload.shape
    assert lens_single.shape == lens.shape
    assert table_single.shape == table.shape
    assert table_single.rows == list(range(18))

    sm120_chunk = 4096 // 6 * 6
    assert sm120_chunk == 4092 and sm120_chunk % 6 == 0
    print("SGLang #45251 grouped-indexer CPU layout test: PASS")


if __name__ == "__main__":
    main()
