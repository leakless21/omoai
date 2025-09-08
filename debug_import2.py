import sys
print("Initial sys.path:")
for i, path in enumerate(sys.path):
    print(f"{i}: {path}")

# Add chunkformer to path like the ASR module does
import os
from pathlib import Path

chunkformer_dir = Path("./src/chunkformer").resolve()
print(f"\nAdding chunkformer_dir to path: {chunkformer_dir}")

if str(chunkformer_dir) not in sys.path:
    sys.path.insert(0, str(chunkformer_dir))
    
print("\nUpdated sys.path:")
for i, path in enumerate(sys.path):
    print(f"{i}: {path}")

# Try to import after path manipulation
try:
    # This is the actual import from the ASR module
    from chunkformer import decode as cfdecode
    print("\nImport successful!")
except ImportError as e:
    print(f"\nImport failed: {e}")
    
# Try the other import
try:
    from chunkformer.model.utils.ctc_utils import get_output_with_timestamps
    print("\nCTC utils import successful!")
except ImportError as e:
    print(f"\nCTC utils import failed: {e}")