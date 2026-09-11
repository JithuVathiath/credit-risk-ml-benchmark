"""Download and verify the pinned official SBA data snapshot."""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "sources.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "credit-risk-ml-benchmark/1.0"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)
    temporary.replace(destination)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for resource in manifest["resources"]:
        destination = ROOT / resource["destination"]
        expected = resource["sha256"]
        expected_size = resource["bytes"]
        if (
            destination.exists()
            and destination.stat().st_size == expected_size
            and sha256(destination) == expected
        ):
            print(f"verified  {destination.relative_to(ROOT)}")
            continue
        print(f"download  {resource['name']}")
        download(resource["url"], destination)
        actual = sha256(destination)
        actual_size = destination.stat().st_size
        if actual != expected or actual_size != expected_size:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"integrity mismatch for {resource['name']}: expected "
                f"{expected_size} bytes/{expected}, got {actual_size} bytes/{actual}"
            )
        print(f"verified  {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as error:
        print(f"data download failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
