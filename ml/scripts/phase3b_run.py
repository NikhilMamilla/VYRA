"""Launch the Phase 3B orchestrator.

    python scripts/phase3b_run.py

Long-running (feature recomputation + a partial VizWiz-train download). Runs
every step, writes runs/phase3b-calibration-v1/status.json after each, and is
resumable -- re-running skips completed steps. Intended to be started in the
background; inspect status.json / reports/phase3b-calibration-v1/ when done.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vyra_ml.experiment.phase3b import run_all  # noqa: E402

if __name__ == "__main__":
    run_all()
