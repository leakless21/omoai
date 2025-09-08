#!/usr/bin/env python3
"""
OMOAI API Startup Script

This script properly configures multiprocessing for CUDA and starts the API server.
"""

import multiprocessing
import sys
import os

def main():
    """Start the OMOAI API with proper CUDA multiprocessing configuration."""
    
    # Set multiprocessing to 'spawn' to avoid CUDA re-initialization issues
    # This must be done before any CUDA operations
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("✅ Set multiprocessing start method to 'spawn' for CUDA compatibility")
    except RuntimeError as e:
        # Start method already set - this is okay
        print(f"ℹ️ Multiprocessing method: {e}")
    
    # Set CUDA environment variables for better stability
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Import and start the API
    from omoai.api.app import main as api_main
    
    print("🚀 Starting OMOAI API...")
    api_main()

if __name__ == "__main__":
    main()
