# Data Access and Integrity

The raw data is downloaded directly from the U.S. Small Business Administration and is
not committed to Git. The pinned manifest in `sources.json` records the official URLs,
snapshot date, byte sizes, and SHA-256 checksums used for the published benchmark.

Run:

```bash
python scripts/download_data.py
```

The command skips valid existing files, rejects checksum mismatches, and never uses a
third-party mirror. Approximately 900 MB of free disk space is required.

## Privacy minimisation

Although borrower names and addresses appear in the public FOIA files, this project never
loads those columns. No direct borrower or street-level identifiers are written to processed
data, metrics, figures, or logs.

## Licence

The SBA catalogue labels the source as U.S. Government Works/Public Domain. The repository's
MIT licence applies only to original code and documentation.
