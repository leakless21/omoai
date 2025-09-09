#!/usr/bin/env python3
"""
OMOAI API Startup Script

This script properly configures multiprocessing for CUDA and starts the API server.
Uses spawn multiprocessing to avoid CUDA re-initialization issues.
"""

import multiprocessing
import sys
import os
from pathlib import Path

def configure_multiprocessing():
    """Configure multiprocessing for CUDA compatibility."""
    try:
        # This MUST be called before importing any modules that use CUDA
        multiprocessing.set_start_method('spawn', force=True)
        print("✅ Set multiprocessing start method to 'spawn' for CUDA compatibility")
        return True
    except RuntimeError as e:
        if "context has already been set" in str(e):
            current_method = multiprocessing.get_start_method()
            if current_method == 'spawn':
                print(f"✅ Multiprocessing already set to 'spawn' method")
                return True
            else:
                print(f"⚠️ Warning: Multiprocessing method already set to '{current_method}', cannot change to 'spawn'")
                return False
        else:
            print(f"❌ Error setting multiprocessing method: {e}")
            return False

def configure_environment():
    """Configure environment variables for optimal CUDA and API performance."""
    # CUDA environment variables
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA operations for better performance
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # For RTX 3060, adjust based on your GPU
    
    # vLLM environment variables for better multiprocessing
    os.environ['VLLM_USE_TRITON_FLASH_ATTN'] = '0'  # Disable triton flash attention for compatibility
    os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'  # Use FlashInfer backend
    
    # HuggingFace cache
    cache_dir = Path.home() / ".cache" / "huggingface"
    os.environ['HF_HOME'] = str(cache_dir)
    
    print("✅ Configured environment variables for CUDA and vLLM")

def api_server_worker():
    """Worker function that runs the API server in spawn process."""
    try:
        # Re-configure environment in the spawned process
        configure_environment()
        
        # Import API modules only after multiprocessing is configured
        from omoai.api.app import main as api_main
        
        print("🚀 Starting OMOAI API server...")
        api_main()
        
    except Exception as e:
        print(f"❌ Error starting API server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    """Main entry point that configures multiprocessing and starts the API."""
    print("🔧 Configuring OMOAI API with spawn multiprocessing...")
    
    # Step 1: Configure multiprocessing BEFORE any imports
    if not configure_multiprocessing():
        print("⚠️ Continuing with current multiprocessing method...")
    
    # Step 2: Configure environment
    configure_environment()
    
    # Step 3: Check if we should run in the current process or spawn a new one
    if multiprocessing.get_start_method() == 'spawn':
        print("✅ Using spawn multiprocessing - starting API server...")
        # We can safely import and run the API now
        try:
            from omoai.api.app import main as api_main
            api_main()
        except Exception as e:
            print(f"❌ Error in main process: {e}")
            # Fallback: try running in a spawned process
            print("🔄 Trying fallback with spawned process...")
            process = multiprocessing.Process(target=api_server_worker)
            process.start()
            process.join()
            if process.exitcode != 0:
                print(f"❌ API server process exited with code {process.exitcode}")
                sys.exit(1)
    else:
        print("🔄 Current multiprocessing method is not 'spawn', using spawned process...")
        process = multiprocessing.Process(target=api_server_worker)
        process.start()
        process.join()
        if process.exitcode != 0:
            print(f"❌ API server process exited with code {process.exitcode}")
            sys.exit(1)

if __name__ == "__main__":
    main()
