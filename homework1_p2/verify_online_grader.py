#!/usr/bin/env python3
"""
Colab verification script for ADL Homework 1.

Run this in Google Colab (with GPU) BEFORE submitting to simulate the online grader.
The online grader uses identity-layer replacement for LoRA/QLoRA backward tests,
which the local "Val grader" does not do. This script helps verify you'll get full credit.

Usage in Colab:
  1. Upload your homework folder to Colab
  2. Run: !python verify_online_grader.py
  3. Or:  !python -m grader.homework  (runs the safe_grader simulation)

For full simulation, ensure safe_grader.py exists in grader/ folder.
"""

import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).parent
    sys.path.insert(0, str(root))

    # Check for safe_grader
    safe_grader = root / "grader" / "safe_grader.py"
    if not safe_grader.exists():
        print("NOTE: safe_grader.py not found. Running standard grader.")
        print("To simulate online grader, ensure grader/safe_grader.py exists.")
        result = subprocess.run(
            [sys.executable, "-m", "grader", "homework", "-v"],
            cwd=root,
            capture_output=False,
        )
    else:
        # Use safe_grader (simulates online grader with identity replacement)
        print("=" * 60)
        print("Running ONLINE GRADER SIMULATION (safe_grader)")
        print("This replaces base with identity in LoRA/QLoRA layers")
        print("=" * 60)
        from grader.safe_grader import run

        score = run()
        result = type("Result", (), {"returncode": 0 if score >= 100 else 1})()

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
