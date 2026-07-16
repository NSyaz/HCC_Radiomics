from __future__ import annotations

import subprocess
import sys


def test_tracked_patient_data_guard_passes_for_repository() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_no_tracked_patient_data.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "No prohibited tracked patient-data files found." in result.stdout
