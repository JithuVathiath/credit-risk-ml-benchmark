import json
from pathlib import Path


def test_pinned_sources_are_official_complete_and_unique() -> None:
    manifest_path = Path(__file__).parents[1] / "data" / "sources.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resources = manifest["resources"]
    loan_files = [resource for resource in resources if "records" in resource]

    assert manifest["snapshot_date"] == "2026-06-30"
    assert len(loan_files) == 4
    assert sum(resource["records"] for resource in loan_files) == 1_961_455
    assert len({resource["destination"] for resource in resources}) == len(resources)
    assert all(resource["url"].startswith("https://data.sba.gov/") for resource in resources)
    assert all(resource["bytes"] > 0 for resource in resources)
    assert all(len(resource["sha256"]) == 64 for resource in resources)
