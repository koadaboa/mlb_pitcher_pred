#!/usr/bin/env python
import argparse
import subprocess
import sys
from datetime import datetime

def parse_args():
    p = argparse.ArgumentParser(
        description="Full prediction pipeline: engineer_features → create_advanced_features"
    )
    p.add_argument(
        "--prediction-date", type=str, default=datetime.today().strftime("%Y-%m-%d"),
        help="Date for prediction (YYYY-MM-DD). Defaults to today's date."
    )
    return p.parse_args()

def main():
    args = parse_args()
    date = args.prediction_date

    # 1) feature engineering
    cmd1 = [sys.executable, "-m", "src.scripts.engineer_features", "--prediction-date", date]
    try:
        subprocess.check_call(cmd1)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"✗ engineer_features failed ({e.returncode})\n")
        sys.exit(e.returncode)

    # 2) advanced features
    cmd2 = [sys.executable, "-m", "src.scripts.create_advanced_features", "--prediction-date", date]
    try:
        subprocess.check_call(cmd2)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"✗ create_advanced_features failed ({e.returncode})\n")
        sys.exit(e.returncode)

    print("✔ Full pipeline completed successfully.")

if __name__ == "__main__":
    main()