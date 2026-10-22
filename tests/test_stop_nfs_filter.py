#!/usr/bin/env python3
"""stop.sh must remove every dsv41-* container except the shared NFS exporter, on the head
and on every worker.

The worker filter lives inside a double-quoted remote script, so its awk program is only
right after bash expands that string: bare quotes around "dsv41-nfs" there closed the outer
string and turned the test into `$2 != dsv41-nfs` (numeric, always true), which removed the
exporter on workers. This expands each filter the way bash does and runs it on sample input.
"""
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = "id-nfs dsv41-nfs\nid-w ds41-worker\nid-x dsv41-old\n"


def awk_lines():
    return [l for l in (ROOT / "stop.sh").read_text().splitlines() if "awk '" in l and "dsv41-nfs" in l]


def run_filter(line, remote):
    prog = line.strip().lstrip("|").strip()
    prog = prog[: prog.rindex("||")].strip() if "||" in prog else prog
    if remote:
        # inside the remote script's double quotes: let bash expand it once
        prog = subprocess.run(["bash", "-c", f's="{prog}"; printf %s "$s"'],
                              capture_output=True, text=True, check=True).stdout
    return subprocess.run(["bash", "-c", prog], input=SAMPLE, capture_output=True, text=True).stdout.split()


class StopKeepsNfs(unittest.TestCase):
    def test_head_and_worker_filters_keep_the_exporter(self):
        lines = awk_lines()
        self.assertEqual(len(lines), 2, lines)
        head, worker = lines
        for line, remote in ((head, False), (worker, True)):
            kept_out = run_filter(line, remote)
            self.assertNotIn("id-nfs", kept_out, line)
            self.assertIn("id-x", kept_out, line)


if __name__ == "__main__":
    unittest.main()
