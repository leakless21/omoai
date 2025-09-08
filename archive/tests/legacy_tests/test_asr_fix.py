#!/usr/bin/env python3
"""
Test script to verify the ASR module import fix
"""
import sys
import subprocess
from pathlib import Path

def test_asr_script_directly():
    """Test the ASR script directly to see if the module import fix works"""
    project_root = Path(__file__).resolve().parent
    asr_script = project_root / "scripts" / "asr.py"
    
    # Use a real audio file
    audio_file = project_root / "data" / "input" / "checklistpv.mp3"
    output_file = project_root / "test_output.json"
    config_file = project_root / "config.yaml"
    
    if not audio_file.exists():
        print(f"Audio file not found: {audio_file}")
        return False
    
    cmd = [
        sys.executable, "-m", "scripts.asr",
        "--audio", str(audio_file),
        "--out", str(output_file),
        "--config", str(config_file)
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Working directory: {project_root}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        if result.returncode == 0:
            print("✅ ASR script executed successfully!")
            return True
        else:
            print("❌ ASR script failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ ASR script timed out")
        return False
    except Exception as e:
        print(f"❌ Error running ASR script: {e}")
        return False

if __name__ == "__main__":
    success = test_asr_script_directly()
    sys.exit(0 if success else 1)