"""Sequential driver for the v4 rankact chunk runner.

Runs `pickle` once, then `sweep <i>` for every selected activity state in its
own fresh process (memory-reaper safety, same contract as the v3 chunked
runner), then `merge`. Logs every stage to
`reports/algorithm_diagnostics/v4_rankact_driver.log` and prints
RANKACT_V4_DONE on success.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "reports" / "algorithm_diagnostics" / "v4_rankact_driver.log"
META = ROOT / "reports" / "algorithm_diagnostics" / "rankact_chunks_v4" / "meta.pkl"


def run_stage(args: list[str]) -> None:
    t0 = time.time()
    with LOG.open("ab") as fh:
        fh.write(f"\n===== stage {' '.join(args)} {time.strftime('%H:%M:%S')} =====\n".encode())
        fh.flush()
        proc = subprocess.run([sys.executable, "src/diagnose_p2_rankact_v4.py", *args],
                              cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise SystemExit(f"stage {args} failed rc={proc.returncode}; see {LOG}")
    with LOG.open("ab") as fh:
        fh.write(f"----- stage {' '.join(args)} done in {time.time()-t0:.0f}s -----\n".encode())


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    run_stage(["pickle"])
    with META.open("rb") as fh:
        meta = pickle.load(fh)
    n = int(meta["n_states"])
    print(f"pickle done; {n} activity states to sweep", flush=True)
    for i in range(n):
        run_stage(["sweep", str(i)])
        print(f"sweep {i+1}/{n} done", flush=True)
    run_stage(["merge"])
    print("RANKACT_V4_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
