"""Diagnose the import issue — run this from the same dir/env as uvicorn."""
import sys
print(f"Python: {sys.executable}")
print(f"Prefix: {sys.prefix}")
print()

# Test 1: Can we import kiteconnect?
try:
    import kiteconnect
    print(f"✅ kiteconnect found: {kiteconnect.__file__}")
except ImportError as e:
    print(f"❌ kiteconnect: {e}")

# Test 2: Can we import load_kite_from_access?
try:
    sys.path.append(".")
    import load_kite_from_access
    print(f"✅ load_kite_from_access found: {load_kite_from_access.__file__}")
except ImportError as e:
    print(f"❌ load_kite_from_access: {e}")
except Exception as e:
    print(f"❌ load_kite_from_access (other error): {type(e).__name__}: {e}")

# Test 3: Can we import config?
try:
    import config
    print(f"✅ config found: {config.__file__}")
except ImportError as e:
    print(f"❌ config: {e}")
