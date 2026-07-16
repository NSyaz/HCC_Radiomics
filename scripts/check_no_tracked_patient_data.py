from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys


BLOCKED_SUFFIXES = {".xlsx", ".xls", ".xlsm"}
BLOCKED_DATA_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".xlsm"}
APPROVED_SYNTHETIC_ROOT = Path("tests/fixtures/synthetic")
SENSITIVE_COLUMN_TERMS = {
    "patient name",
    "mrn",
    "accession",
    "date of birth",
    "medical record number",
    "identity card",
    "national id",
    "hospital identifier",
}


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line.strip()]


def is_under(path: Path, root: Path) -> bool:
    try:
        path.as_posix()
        path.relative_to(root)
    except ValueError:
        return False
    return True


def is_approved_synthetic_fixture(path: Path) -> bool:
    return is_under(path, APPROVED_SYNTHETIC_ROOT) and "synthetic" in path.as_posix().lower()


def blocked_data_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in BLOCKED_SUFFIXES:
        return not is_approved_synthetic_fixture(path)
    if path.parts and path.parts[0] == "data" and suffix in BLOCKED_DATA_SUFFIXES:
        return True
    return False


def check_sensitive_headers(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".tsv", ".txt"} or not is_approved_synthetic_fixture(path):
        return []

    delimiter = "\t" if suffix in {".tsv", ".txt"} else ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader, [])

    normalised = {" ".join(column.lower().replace("_", " ").split()) for column in header}
    return sorted(term for term in SENSITIVE_COLUMN_TERMS if term in normalised)


def main() -> int:
    failures: list[str] = []
    for path in tracked_files():
        if blocked_data_file(path):
            failures.append(f"Blocked tracked research-data file: {path.as_posix()}")
        sensitive_headers = check_sensitive_headers(path)
        if sensitive_headers:
            failures.append(
                f"Sensitive-looking header(s) in approved fixture {path.as_posix()}: "
                + ", ".join(sensitive_headers)
            )

    if failures:
        print("Tracked patient-data guard failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("No prohibited tracked patient-data files found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
