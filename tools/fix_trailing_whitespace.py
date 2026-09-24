from __future__ import annotations

import re
import sys
from pathlib import Path

TRAILING_WHITESPACE = re.compile(rb"[ \t]+(?=(?:\r?\n)?$)")


def fix_file(path: str) -> None:
    file_path = Path(path)
    original = file_path.read_bytes()
    lines = original.splitlines(keepends=True)
    fixed = b"".join(TRAILING_WHITESPACE.sub(b"", line) for line in lines)
    if fixed != original:
        file_path.write_bytes(fixed)


for filename in sys.argv[1:]:
    fix_file(filename)
