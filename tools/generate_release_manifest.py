"""Write a transparent file inventory for a local SCOPED release package."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


IGNORED_PARTS = {".git", "__pycache__"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--output", default="RELEASE_MANIFEST.json")
    args = parser.parse_args()

    root = args.package.resolve()
    output = root / args.output
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == output or any(part in IGNORED_PARTS for part in path.parts):
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        )

    payload = {
        "package_root": root.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "total_bytes": sum(entry["bytes"] for entry in files),
        "files": files,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
