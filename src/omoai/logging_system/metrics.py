"""
Performance metrics collection and reporting.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import psutil

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .logger import get_logger


@dataclass
class PerformanceMetric:
    """Single performance metric data point."""

    timestamp: datetime
    operation: str
    duration_ms: float
    success: bool = True
    error_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System resource metrics."""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    gpu_memory_used_gb: float | None = None
    gpu_memory_total_gb: float | None = None
    gpu_utilization_percent: float | None = None


class MetricsCollector:
    """Collects and aggregates performance metrics."""

    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.metrics: deque = deque(maxlen=max_metrics)
        self.system_metrics: deque = deque(
            maxlen=1000
        )  # Keep last 1000 system snapshots
        self.operation_stats = defaultdict(list)
        self.lock = threading.Lock()
        self.logger = get_logger("omoai.metrics")

        # Auto-start system monitoring
        self._monitoring = False
        self._monitor_thread: threading.Thread | None = None

    def record_performance(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        error_type: str | None = None,
        **metadata: Any,
    ) -> None:
        """Record a performance metric."""
        metric = PerformanceMetric(
            timestamp=datetime.now(UTC),
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            error_type=error_type,
            metadata=metadata,
        )

        with self.lock:
            self.metrics.append(metric)
            self.operation_stats[operation].append(duration_ms)

            # Keep only recent stats per operation (last 100)
            if len(self.operation_stats[operation]) > 100:
                self.operation_stats[operation] = self.operation_stats[operation][-100:]

    def record_system_metrics(self) -> SystemMetrics:
        """Record current system resource usage."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()

        gpu_memory_used = None
        gpu_memory_total = None
        gpu_utilization = None

        # GPU metrics if available
        if HAS_TORCH and torch.cuda.is_available():
            try:
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_memory_total = (
                    torch.cuda.get_device_properties(0).total_memory / 1024**3
                )  # GB

                # GPU utilization (approximation based on memory usage)
                gpu_utilization = (
                    (gpu_memory_used / gpu_memory_total * 100)
                    if gpu_memory_total > 0
                    else 0
                )
            except Exception as e:
                self.logger.debug(f"Failed to get GPU metrics: {e}")

        metrics = SystemMetrics(
            timestamp=datetime.now(UTC),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / 1024**3,
            memory_available_gb=memory.available / 1024**3,
            gpu_memory_used_gb=gpu_memory_used,
            gpu_memory_total_gb=gpu_memory_total,
            gpu_utilization_percent=gpu_utilization,
        )

        with self.lock:
            self.system_metrics.append(metrics)

        return metrics

    def start_monitoring(self, interval: int = 30) -> None:
        """Start continuous system monitoring."""
        if self._monitoring:
            return

        self._monitoring = True

        def monitor_loop():
            while self._monitoring:
                try:
                    self.record_system_metrics()
                    time.sleep(interval)
                except Exception as e:
                    self.logger.error(f"System monitoring error: {e}")
                    time.sleep(interval)

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()

        self.logger.info(f"Started system monitoring with {interval}s interval")

    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        if not self._monitoring:
            return

        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)

        self.logger.info("Stopped system monitoring")

    def get_operation_stats(self, operation: str) -> dict[str, Any]:
        """Get statistical summary for an operation."""
        with self.lock:
            durations = self.operation_stats.get(operation, [])

        if not durations:
            return {"operation": operation, "count": 0}

        durations_sorted = sorted(durations)
        count = len(durations)

        return {
            "operation": operation,
            "count": count,
            "min_ms": min(durations),
            "max_ms": max(durations),
            "mean_ms": sum(durations) / count,
            "p50_ms": durations_sorted[count // 2],
            "p90_ms": durations_sorted[int(count * 0.9)]
            if count > 10
            else max(durations),
            "p99_ms": durations_sorted[int(count * 0.99)]
            if count > 100
            else max(durations),
        }

    def get_recent_metrics(self, minutes: int = 10) -> list[PerformanceMetric]:
        """Get metrics from the last N minutes."""
        cutoff = datetime.now(UTC).timestamp() - (minutes * 60)

        with self.lock:
            return [
                metric
                for metric in self.metrics
                if metric.timestamp.timestamp() > cutoff
            ]

    def get_system_health(self) -> dict[str, Any]:
        """Get current system health summary."""
        current_metrics = self.record_system_metrics()

        # Get recent performance metrics
        recent_metrics = self.get_recent_metrics(minutes=5)

        # Calculate error rate
        total_recent = len(recent_metrics)
        failed_recent = sum(1 for m in recent_metrics if not m.success)
        error_rate = (failed_recent / total_recent * 100) if total_recent > 0 else 0

        # System status
        system_status = "healthy"
        if current_metrics.cpu_percent > 90:
            system_status = "high_cpu"
        elif current_metrics.memory_percent > 90:
            system_status = "high_memory"
        elif error_rate > 10:
            system_status = "high_error_rate"

        health = {
            "status": system_status,
            "timestamp": current_metrics.timestamp.isoformat(),
            "system": {
                "cpu_percent": current_metrics.cpu_percent,
                "memory_percent": current_metrics.memory_percent,
                "memory_used_gb": current_metrics.memory_used_gb,
                "memory_available_gb": current_metrics.memory_available_gb,
            },
            "performance": {
                "recent_operations": total_recent,
                "error_rate_percent": error_rate,
                "operations_per_minute": total_recent if total_recent > 0 else 0,
            },
        }

        # Add GPU info if available
        if current_metrics.gpu_memory_used_gb is not None:
            health["system"]["gpu"] = {
                "memory_used_gb": current_metrics.gpu_memory_used_gb,
                "memory_total_gb": current_metrics.gpu_memory_total_gb,
                "utilization_percent": current_metrics.gpu_utilization_percent,
            }

        return health

    def get_performance_report(self, hours: int = 1) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        cutoff = datetime.now(UTC).timestamp() - (hours * 3600)

        with self.lock:
            recent_metrics = [
                metric
                for metric in self.metrics
                if metric.timestamp.timestamp() > cutoff
            ]

        if not recent_metrics:
            return {"period_hours": hours, "no_data": True}

        # Overall stats
        total_operations = len(recent_metrics)
        successful_operations = sum(1 for m in recent_metrics if m.success)
        failed_operations = total_operations - successful_operations

        # Performance stats
        all_durations = [m.duration_ms for m in recent_metrics]
        avg_duration = sum(all_durations) / len(all_durations)

        # Per-operation breakdown
        operations_breakdown = {}
        for operation in set(m.operation for m in recent_metrics):
            operations_breakdown[operation] = self.get_operation_stats(operation)

        # Error breakdown
        error_types = defaultdict(int)
        for metric in recent_metrics:
            if not metric.success and metric.error_type:
                error_types[metric.error_type] += 1

        return {
            "period_hours": hours,
            "summary": {
                "total_operations": total_operations,
                "successful_operations": successful_operations,
                "failed_operations": failed_operations,
                "success_rate_percent": (successful_operations / total_operations * 100)
                if total_operations > 0
                else 0,
                "average_duration_ms": avg_duration,
            },
            "operations": operations_breakdown,
            "errors": dict(error_types),
            "system_health": self.get_system_health(),
        }


class PerformanceLogger:
    """High-level performance logging interface."""

    def __init__(self, collector: MetricsCollector | None = None):
        self.collector = collector or MetricsCollector()
        self.logger = get_logger("omoai.performance")

        # Start monitoring by default
        self.collector.start_monitoring()

    def log_operation(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        error_type: str | None = None,
        **metadata: Any,
    ) -> None:
        """Log an operation with metrics collection."""
        # Record in collector
        self.collector.record_performance(
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            error_type=error_type,
            **metadata,
        )

        # Also log using structured logging
        extra_data = {
            "operation": operation,
            "duration_ms": duration_ms,
            "success": success,
            **metadata,
        }

        if error_type:
            extra_data["error_type"] = error_type

        # Choose log level based on performance and success
        if not success:
            level = logging.ERROR
            msg = f"Operation failed: {operation} after {duration_ms:.2f}ms"
        elif duration_ms > 5000:
            level = logging.WARNING
            msg = f"Slow operation: {operation} took {duration_ms:.2f}ms"
        elif duration_ms > 1000:
            level = logging.INFO
            msg = f"Operation: {operation} took {duration_ms:.2f}ms"
        else:
            level = logging.DEBUG
            msg = f"Fast operation: {operation} completed in {duration_ms:.2f}ms"

        self.logger.log(level, msg, extra=extra_data)

    def get_stats(self, operation: str) -> dict[str, Any]:
        """Get statistics for an operation."""
        return self.collector.get_operation_stats(operation)

    def get_health(self) -> dict[str, Any]:
        """Get current system health."""
        return self.collector.get_system_health()

    def get_report(self, hours: int = 1) -> dict[str, Any]:
        """Get performance report."""
        return self.collector.get_performance_report(hours)


# Global metrics collector instance
_global_collector: MetricsCollector | None = None
_global_performance_logger: PerformanceLogger | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def get_performance_logger() -> PerformanceLogger:
    """Get the global performance logger instance."""
    global _global_performance_logger
    if _global_performance_logger is None:
        _global_performance_logger = PerformanceLogger(get_metrics_collector())
    return _global_performance_logger
