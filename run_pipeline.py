"""Run the whole pipeline once the three collectors' JSON files are in data/raw/.

    python run_pipeline.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import analysis  # noqa: E402
import features  # noqa: E402
import integrate  # noqa: E402
import parse_raw  # noqa: E402
import train  # noqa: E402
import visualize  # noqa: E402

STEPS = [
    ("1. parse raw collector output", parse_raw.main),
    ("2. preprocess + integrate sources", integrate.integrate),
    ("3. feature engineering", features.main),
    ("4. encoding experiment, then train and evaluate models", train.train_all),
    ("5. depreciation analysis + feature ranking", analysis.main),
    ("6. figures", visualize.main),
]

if __name__ == "__main__":
    for name, step in STEPS:
        t = time.time()
        print(f"\n=== {name} ===", flush=True)
        step()
        print(f"--- done in {time.time() - t:.1f}s", flush=True)
