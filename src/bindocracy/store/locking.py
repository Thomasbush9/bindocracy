"""Coordinate campaign readers and writers across independent controllers.

DuckDB's native locks refuse concurrent cross-process writers. This advisory
lock makes supported harness operations wait instead, including readers while
a writer owns the file. The sidecar is permanent: deleting it breaks ownership.
"""
from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb


@contextmanager
def locked_connection(
    database: str | Path, *, read_only: bool = False,
) -> Iterator[duckdb.DuckDBPyConnection]:
    path = Path(database).resolve()
    lock_path = path.with_name(path.name + ".lock")
    # Existing read-only archives can retain a read-only lock file. Working
    # campaign directories need write access to create their first sidecar.
    if read_only and lock_path.exists():
        handle = lock_path.open("rb")
    else:
        handle = lock_path.open("a+b")
    with handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH if read_only else fcntl.LOCK_EX)
        try:
            connection = duckdb.connect(str(path), read_only=read_only)
            try:
                yield connection
            finally:
                connection.close()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
