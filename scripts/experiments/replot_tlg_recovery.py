"""Re-render the parameter-recovery figure from the cached CSV (no rerun): loads
runs/tlg_recovery/recovery_all.csv and calls the existing plot fn with the updated suptitle.
Output path is the first CLI argument."""
import os
import sys
from pathlib import Path

os.environ.setdefault("LG_TLG_QUICK", "0")  # full DS=[0,1,2], full NS
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import run_tlg_recovery as R

combined = pd.read_csv(R.OUT_DIR / "recovery_all.csv")
scenarios = list(zip(R.SIGMAS, R.ALPHAS))
out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
R.plot_combined_recovery(combined, scenarios, out)
print(f"Saved {out}")
