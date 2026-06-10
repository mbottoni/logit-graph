"""Re-render the ROC figures from the cached roc_long.csv (no experiment rerun).

Usage: replot_tlg_roc.py <out_dir>
Writes <out_dir>/tlg_roc_effect.png and <out_dir>/tlg_roc_sample.png.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import run_tlg_roc_experiments as R

df = pd.read_csv(R.OUT_DIR / "roc_long.csv")
outdir = Path(sys.argv[1])
outdir.mkdir(parents=True, exist_ok=True)
R._plot(df, "effect", outdir / "tlg_roc_effect.png")
R._plot(df, "sample", outdir / "tlg_roc_sample.png")
print(f"Saved tlg_roc_effect.png and tlg_roc_sample.png to {outdir}")
