import os
import sys
import pathlib


def pytest_sessionstart(session):
    # Ensure `src` is on sys.path so `import logit_graph` works in tests
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    src_path = repo_root / 'src'
    if src_path.exists():
        sys.path.insert(0, str(src_path))


