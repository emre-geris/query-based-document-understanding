"""Gzipped text files that are byte-identical when rewritten from the same content."""

import gzip
import io
from pathlib import Path


def open_gzip_text(path: Path) -> io.TextIOWrapper:
    """A text writer for ``path``; the gzip header carries no timestamp (mtime 0), so packing
    the same pages again gives the same bytes and git sees no change."""
    raw = gzip.GzipFile(filename="", mode="wb", fileobj=path.open("wb"), mtime=0)
    return io.TextIOWrapper(raw, encoding="utf-8", newline="\n")
