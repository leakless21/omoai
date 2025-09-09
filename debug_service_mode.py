#!/usr/bin/env python3

import sys
import asyncio
from pathlib import Path

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omoai.api.services_enhanced import should_use_in_memory_service, get_service_mode
from omoai.api.services_v2 import health_check_models

async def debug_service_mode():
    print("🔍 DEBUGGING SERVICE MODE")
    print("")
    
    # Check service mode
    mode = get_service_mode()
    print(f"📋 Service mode from config: {mode}")
    
    # Check if should use in-memory
    should_use = await should_use_in_memory_service()
    print(f"🤔 Should use in-memory service: {should_use}")
    
    # Check model health
    try:
        health = await health_check_models()
        print(f"🏥 Model health check: {health}")
    except Exception as e:
        print(f"❌ Model health check failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_service_mode())
