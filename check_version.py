"""
Talk2Mind - Version Check
==========================
Run this first if you're troubleshooting: python check_version.py
It confirms you're running the patched code (v1.1, pandas/pyarrow fix included).
"""

import inspect
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

def check():
    import facial_emotion_model
    source = inspect.getsource(facial_emotion_model)
    has_fix = "to_numpy" in source and "future.infer_string" in source
    print("=" * 50)
    if has_fix:
        print("✅ You ARE running the patched version (v1.1).")
        print("   The pandas/scikit-learn compatibility fix is present.")
    else:
        print("❌ You are running an OLDER, unpatched version.")
        print("   Please re-download Talk2Mind_Project.zip from the")
        print("   latest message, delete your existing Talk2Mind folder")
        print("   completely, and extract the new zip fresh.")
    print("=" * 50)

if __name__ == "__main__":
    check()
