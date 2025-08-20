"""
Singleton model managers for efficient API services.

This module provides cached model instances to eliminate the overhead of
loading models for each request, dramatically improving API performance.
"""
import asyncio
import threading
from typing import Optional, Dict, Any
from pathlib import Path

from ..config import get_config, OmoAIConfig
from ..pipeline import ChunkFormerASR, run_full_pipeline_memory
from ..pipeline.postprocess import VLLMProcessor


class ModelSingletons:
    """Thread-safe singleton manager for ML models."""
    
    _instance: Optional["ModelSingletons"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "ModelSingletons":
        """Create or return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the singleton (only once)."""
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self._initialized = True
        self._config: Optional[OmoAIConfig] = None
        self._asr_model: Optional[ChunkFormerASR] = None
        self._punctuation_processor: Optional[VLLMProcessor] = None
        self._summarization_processor: Optional[VLLMProcessor] = None
        self._init_lock = threading.Lock()
    
    def _ensure_config(self) -> OmoAIConfig:
        """Ensure configuration is loaded."""
        if self._config is None:
            self._config = get_config()
        return self._config
    
    def get_asr_model(self) -> ChunkFormerASR:
        """Get or create the ASR model instance."""
        if self._asr_model is None:
            with self._init_lock:
                if self._asr_model is None:
                    config = self._ensure_config()
                    self._asr_model = ChunkFormerASR(
                        model_checkpoint=config.paths.chunkformer_checkpoint,
                        device=config.asr.device,
                        chunkformer_dir=config.paths.chunkformer_dir,
                    )
                    # Trigger lazy initialization
                    self._asr_model.initialize()
        return self._asr_model
    
    def get_punctuation_processor(self) -> VLLMProcessor:
        """Get or create the punctuation processor instance."""
        if self._punctuation_processor is None:
            with self._init_lock:
                if self._punctuation_processor is None:
                    config = self._ensure_config()
                    llm_config = config.punctuation.llm
                    
                    self._punctuation_processor = VLLMProcessor(
                        model_id=llm_config.model_id,
                        quantization=llm_config.quantization,
                        max_model_len=llm_config.max_model_len,
                        gpu_memory_utilization=llm_config.gpu_memory_utilization,
                        max_num_seqs=llm_config.max_num_seqs,
                        max_num_batched_tokens=llm_config.max_num_batched_tokens,
                        trust_remote_code=llm_config.trust_remote_code,
                    )
                    # Trigger lazy initialization
                    self._punctuation_processor.initialize()
        return self._punctuation_processor
    
    def get_summarization_processor(self) -> VLLMProcessor:
        """Get or create the summarization processor instance."""
        if self._summarization_processor is None:
            with self._init_lock:
                if self._summarization_processor is None:
                    config = self._ensure_config()
                    llm_config = config.summarization.llm
                    
                    # Check if we can reuse punctuation processor
                    punct_config = config.punctuation.llm
                    if (llm_config.model_id == punct_config.model_id and
                        llm_config.quantization == punct_config.quantization and
                        llm_config.max_model_len == punct_config.max_model_len):
                        # Reuse the punctuation processor
                        self._summarization_processor = self.get_punctuation_processor()
                    else:
                        # Create separate processor
                        self._summarization_processor = VLLMProcessor(
                            model_id=llm_config.model_id,
                            quantization=llm_config.quantization,
                            max_model_len=llm_config.max_model_len,
                            gpu_memory_utilization=llm_config.gpu_memory_utilization,
                            max_num_seqs=llm_config.max_num_seqs,
                            max_num_batched_tokens=llm_config.max_num_batched_tokens,
                            trust_remote_code=llm_config.trust_remote_code,
                        )
                        # Trigger lazy initialization
                        self._summarization_processor.initialize()
        return self._summarization_processor
    
    def preload_models(self) -> Dict[str, bool]:
        """
        Preload all models for faster first request.
        
        Returns:
            Dictionary indicating which models were successfully loaded.
        """
        results = {
            "asr": False,
            "punctuation": False,
            "summarization": False,
        }
        
        try:
            self.get_asr_model()
            results["asr"] = True
        except Exception as e:
            print(f"Failed to preload ASR model: {e}")
        
        try:
            self.get_punctuation_processor()
            results["punctuation"] = True
        except Exception as e:
            print(f"Failed to preload punctuation processor: {e}")
        
        try:
            self.get_summarization_processor()
            results["summarization"] = True
        except Exception as e:
            print(f"Failed to preload summarization processor: {e}")
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models.
        
        Returns:
            Dictionary with model loading status and configurations.
        """
        config = self._ensure_config()
        
        return {
            "asr": {
                "loaded": self._asr_model is not None,
                "device": config.asr.device,
                "model_path": str(config.paths.chunkformer_checkpoint),
                "autocast_dtype": config.asr.autocast_dtype,
            },
            "punctuation": {
                "loaded": self._punctuation_processor is not None,
                "model_id": config.punctuation.llm.model_id,
                "trust_remote_code": config.punctuation.llm.trust_remote_code,
                "max_model_len": config.punctuation.llm.max_model_len,
            },
            "summarization": {
                "loaded": self._summarization_processor is not None,
                "model_id": config.summarization.llm.model_id,
                "trust_remote_code": config.summarization.llm.trust_remote_code,
                "reuses_punctuation": (
                    self._summarization_processor is self._punctuation_processor
                    if self._summarization_processor and self._punctuation_processor
                    else False
                ),
            },
            "config": {
                "security": {
                    "trust_remote_code_any": any([
                        config.llm.trust_remote_code,
                        config.punctuation.llm.trust_remote_code,
                        config.summarization.llm.trust_remote_code,
                    ]),
                    "api_host": config.api.host,
                    "progress_output": config.api.enable_progress_output,
                }
            }
        }
    
    def reload_config(self) -> None:
        """
        Reload configuration and reset models.
        
        Warning: This will clear all cached models and they will need to be
        reloaded on next request. Use sparingly in production.
        """
        with self._init_lock:
            self._config = None
            self._asr_model = None
            self._punctuation_processor = None
            self._summarization_processor = None
        
        # Reload config
        from ..config import reload_config
        reload_config()


# Global singleton instance
model_singletons = ModelSingletons()


# Convenience functions for API services
def get_asr_model() -> ChunkFormerASR:
    """Get the cached ASR model instance."""
    return model_singletons.get_asr_model()


def get_punctuation_processor() -> VLLMProcessor:
    """Get the cached punctuation processor instance."""
    return model_singletons.get_punctuation_processor()


def get_summarization_processor() -> VLLMProcessor:
    """Get the cached summarization processor instance."""
    return model_singletons.get_summarization_processor()


def preload_all_models() -> Dict[str, bool]:
    """
    Preload all models for faster API startup.
    
    This can be called during application startup to warm up models.
    """
    return model_singletons.preload_models()


def get_model_status() -> Dict[str, Any]:
    """Get current model loading status and configuration."""
    return model_singletons.get_model_info()


async def initialize_models_async() -> Dict[str, bool]:
    """
    Asynchronously initialize models to avoid blocking the event loop.
    
    This is useful for startup initialization in async applications.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, preload_all_models)
