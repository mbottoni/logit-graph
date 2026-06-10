"""Re-render the AIC d-selection confusion figure from the cached CSV (no rerun).

Usage: replot_tlg_aic_d.py <out_png>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import run_tlg_aic_d_recovery as R

df = pd.read_csv(R.OUT_DIR / "aic_d_long.csv")
out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
R._plot_confusion(df, out)
print(f"Saved {out}")
