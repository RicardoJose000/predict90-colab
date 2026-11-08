"""Daily run: fresh data -> predictions -> Excel. Used by RUN_ME.bat.

python run_daily.py            normal run, opens the Excel file at the end
python run_daily.py --retrain  train the models again now
python run_daily.py --quiet    no Excel window (for the scheduled task)
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from p90 import pipeline  # noqa: E402
except ImportError as e:
    if "Application Control" not in str(e):
        raise
    print(f"""
Windows blocked a library file: {e}

This is Windows Smart App Control. It is not a folder problem or a project error.

1. Try an older version of the library. Open cmd and run:
       python -m pip install "pandas==2.2.3"
   Then run this file again.
2. If it stays: use Google Colab, nothing runs on your PC (GUIDE.html, part B).
3. Or turn it off: Windows Security > App & browser control > Smart App Control settings > Off.
   That is your PC's security, so your choice. On older Windows 11 turning it back on can need a reset.
""")
    sys.exit(1)

if __name__ == "__main__":
    out = pipeline.daily(force_retrain="--retrain" in sys.argv)
    path = out.get("excel")
    if path and "--quiet" not in sys.argv and os.name == "nt":
        for f in (out.get("page"), path):
            try:
                os.startfile(f)
            except Exception:
                pass
