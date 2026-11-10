"""
ML System Resource Monitor v3
=============================
Watches this laptop in the background, learns what "normal" looks like for
*you* (idle, gaming, on battery, …), and records explained events when
something is genuinely off — naming the process responsible.

    pythonw monitor.py               run silently (the Startup shortcut does this)
    python  monitor.py               run with console output
    python  monitor.py status        running? model state, live readings
    python  monitor.py report        what happened in the last 24 h
    python  monitor.py explain       AI & forensic root-cause analysis
    python  monitor.py ask "..."     ask questions about freezes or high usage
    python  monitor.py diagnose      executive system health audit
    python  monitor.py reset         wipe collected data

See README.md for how it works and config.example.toml for tuning.
"""

import os
import sys

# A background tool should not spin up one numeric thread per core (32 here).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from resmon.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
