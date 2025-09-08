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

# Try to import
try:
    from omoai.chunkformer import decode as cfdecode
    print("\nImport successful!")
except ImportError as e:
    print(f"\nImport failed: {e}")
    
# Check what's in the chunkformer directory
print(f"\nContents of chunkformer directory:")
for item in chunkformer_dir.iterdir():
    print(f"  {item.name}")
    
print(f"\nContents of omoai directory inside chunkformer:")
omoai_dir = chunkformer_dir / "omoai"
if omoai_dir.exists():
    for item in omoai_dir.iterdir():
        print(f"  {item.name}")
else:
    print("  omoai directory does not exist")