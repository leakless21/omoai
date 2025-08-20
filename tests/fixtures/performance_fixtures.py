"""
Performance testing and load testing utilities.

This module provides comprehensive performance testing capabilities including
load testing, stress testing, and performance regression detection.
"""
import asyncio
import concurrent.futures
import psutil
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
import sys

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation."""
    duration_ms: float
    memory_usage_mb: float
    cpu_percent: float
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    real_time_factor: Optional[float] = None
    throughput_ops_per_sec: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadTestResult:
    """Results from a load test."""
    test_name: str
    total_operations: int
    successful_operations: int
    failed_operations: int
    total_duration_seconds: float
    average_operation_time_ms: float
    min_operation_time_ms: float
    max_operation_time_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    throughput_ops_per_sec: float
    error_rate_percent: float
    peak_memory_mb: float
    peak_gpu_memory_mb: Optional[float]
    concurrent_users: int
    errors: List[str] = field(default_factory=list)


@dataclass
class StressTestConfig:
    """Configuration for stress testing."""
    max_concurrent_users: int = 10
    ramp_up_duration_seconds: int = 30
    steady_state_duration_seconds: int = 60
    ramp_down_duration_seconds: int = 10
    operation_timeout_seconds: int = 30
    memory_limit_mb: Optional[int] = None
    cpu_limit_percent: Optional[float] = None


class PerformanceMonitor:
    """Real-time performance monitoring during tests."""
    
    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self.monitoring = False
        self.samples: List[Dict[str, float]] = []
        self._monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.samples.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return collected metrics."""
        if not self.monitoring:
            return {}
        
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        if not self.samples:
            return {}
        
        # Calculate statistics
        cpu_values = [s["cpu_percent"] for s in self.samples]
        memory_values = [s["memory_mb"] for s in self.samples]
        
        results = {
            "peak_cpu_percent": max(cpu_values),
            "avg_cpu_percent": sum(cpu_values) / len(cpu_values),
            "peak_memory_mb": max(memory_values),
            "avg_memory_mb": sum(memory_values) / len(memory_values),
            "sample_count": len(self.samples),
            "duration_seconds": len(self.samples) * self.sample_interval
        }
        
        # Add GPU metrics if available
        if HAS_TORCH and torch.cuda.is_available():
            gpu_memory_values = [s.get("gpu_memory_mb", 0) for s in self.samples if "gpu_memory_mb" in s]
            if gpu_memory_values:
                results["peak_gpu_memory_mb"] = max(gpu_memory_values)
                results["avg_gpu_memory_mb"] = sum(gpu_memory_values) / len(gpu_memory_values)
        
        return results
    
    def _monitor_loop(self):
        """Monitoring loop running in separate thread."""
        while self.monitoring:
            try:
                sample = {
                    "timestamp": time.time(),
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "memory_mb": psutil.virtual_memory().used / 1024 / 1024
                }
                
                # Add GPU metrics if available
                if HAS_TORCH and torch.cuda.is_available():
                    try:
                        sample["gpu_memory_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
                        sample["gpu_utilization"] = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else None
                    except Exception:
                        pass
                
                self.samples.append(sample)
                time.sleep(self.sample_interval)
            except Exception:
                # Continue monitoring even if individual samples fail
                continue


class LoadTestRunner:
    """Comprehensive load testing runner."""
    
    def __init__(self, operation_func: Callable, operation_name: str = "test_operation"):
        self.operation_func = operation_func
        self.operation_name = operation_name
        self.monitor = PerformanceMonitor()
    
    def run_single_operation(self, *args, **kwargs) -> PerformanceMetrics:
        """Run a single operation with performance monitoring."""
        # Get initial resource usage
        initial_memory = psutil.virtual_memory().used / 1024 / 1024
        initial_gpu_memory = None
        
        if HAS_TORCH and torch.cuda.is_available():
            try:
                initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            except Exception:
                pass
        
        # Run operation with timing
        start_time = time.time()
        start_cpu = psutil.cpu_percent(interval=None)
        
        try:
            result = self.operation_func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            raise
        finally:
            end_time = time.time()
            end_cpu = psutil.cpu_percent(interval=None)
            
            # Calculate metrics
            duration_ms = (end_time - start_time) * 1000
            final_memory = psutil.virtual_memory().used / 1024 / 1024
            memory_usage = final_memory - initial_memory
            
            final_gpu_memory = None
            gpu_memory_usage = None
            
            if initial_gpu_memory is not None and HAS_TORCH and torch.cuda.is_available():
                try:
                    final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                    gpu_memory_usage = final_gpu_memory - initial_gpu_memory
                except Exception:
                    pass
            
            metrics = PerformanceMetrics(
                duration_ms=duration_ms,
                memory_usage_mb=memory_usage,
                cpu_percent=(start_cpu + end_cpu) / 2,
                gpu_memory_mb=gpu_memory_usage,
                metadata={"success": success}
            )
        
        return metrics
    
    def run_load_test(
        self,
        concurrent_users: int,
        operations_per_user: int,
        test_data: List[Any],
        timeout_seconds: int = 60
    ) -> LoadTestResult:
        """Run load test with specified concurrent users."""
        
        # Start system monitoring
        self.monitor.start_monitoring()
        
        start_time = time.time()
        results = []
        errors = []
        
        def run_user_operations(user_id: int) -> List[PerformanceMetrics]:
            """Run operations for a single user."""
            user_results = []
            for i in range(operations_per_user):
                try:
                    # Use test data in round-robin fashion
                    data_index = (user_id * operations_per_user + i) % len(test_data)
                    test_input = test_data[data_index]
                    
                    metrics = self.run_single_operation(test_input)
                    user_results.append(metrics)
                except Exception as e:
                    errors.append(f"User {user_id}, Op {i}: {str(e)}")
                    # Continue with other operations
            return user_results
        
        # Run concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [
                executor.submit(run_user_operations, user_id)
                for user_id in range(concurrent_users)
            ]
            
            # Wait for completion with timeout
            for future in concurrent.futures.as_completed(futures, timeout=timeout_seconds):
                try:
                    user_results = future.result()
                    results.extend(user_results)
                except concurrent.futures.TimeoutError:
                    errors.append("Operation timed out")
                except Exception as e:
                    errors.append(f"Unexpected error: {str(e)}")
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Stop monitoring and get system metrics
        system_metrics = self.monitor.stop_monitoring()
        
        # Calculate statistics
        total_ops = len(results)
        successful_ops = sum(1 for r in results if r.metadata.get("success", False))
        failed_ops = total_ops - successful_ops
        
        if total_ops > 0:
            durations = [r.duration_ms for r in results]
            durations.sort()
            
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
            
            # Calculate percentiles
            p50_idx = int(0.50 * len(durations))
            p90_idx = int(0.90 * len(durations))
            p95_idx = int(0.95 * len(durations))
            p99_idx = int(0.99 * len(durations))
            
            p50 = durations[p50_idx] if p50_idx < len(durations) else max_duration
            p90 = durations[p90_idx] if p90_idx < len(durations) else max_duration
            p95 = durations[p95_idx] if p95_idx < len(durations) else max_duration
            p99 = durations[p99_idx] if p99_idx < len(durations) else max_duration
            
            throughput = total_ops / total_duration if total_duration > 0 else 0
            error_rate = (failed_ops / total_ops) * 100
            
        else:
            avg_duration = min_duration = max_duration = 0
            p50 = p90 = p95 = p99 = 0
            throughput = 0
            error_rate = 100
        
        return LoadTestResult(
            test_name=f"{self.operation_name}_load_test",
            total_operations=total_ops,
            successful_operations=successful_ops,
            failed_operations=failed_ops,
            total_duration_seconds=total_duration,
            average_operation_time_ms=avg_duration,
            min_operation_time_ms=min_duration,
            max_operation_time_ms=max_duration,
            p50_ms=p50,
            p90_ms=p90,
            p95_ms=p95,
            p99_ms=p99,
            throughput_ops_per_sec=throughput,
            error_rate_percent=error_rate,
            peak_memory_mb=system_metrics.get("peak_memory_mb", 0),
            peak_gpu_memory_mb=system_metrics.get("peak_gpu_memory_mb"),
            concurrent_users=concurrent_users,
            errors=errors[:10]  # Keep only first 10 errors
        )
    
    def run_stress_test(
        self,
        config: StressTestConfig,
        test_data: List[Any]
    ) -> Dict[str, LoadTestResult]:
        """Run comprehensive stress test with ramp-up and ramp-down."""
        
        results = {}
        
        # Phase 1: Ramp-up
        print(f"Starting stress test ramp-up phase ({config.ramp_up_duration_seconds}s)")
        ramp_up_users = list(range(1, config.max_concurrent_users + 1, max(1, config.max_concurrent_users // 10)))
        
        for users in ramp_up_users[:3]:  # Test first few user counts during ramp-up
            result = self.run_load_test(
                concurrent_users=users,
                operations_per_user=5,  # Fewer operations during ramp-up
                test_data=test_data,
                timeout_seconds=config.operation_timeout_seconds
            )
            results[f"ramp_up_{users}_users"] = result
            
            # Check if we're hitting resource limits
            if config.memory_limit_mb and result.peak_memory_mb > config.memory_limit_mb:
                print(f"Memory limit exceeded at {users} users: {result.peak_memory_mb}MB")
                break
        
        # Phase 2: Steady state
        print(f"Starting steady state phase ({config.steady_state_duration_seconds}s)")
        steady_ops_per_user = max(1, config.steady_state_duration_seconds // 10)
        
        steady_result = self.run_load_test(
            concurrent_users=config.max_concurrent_users,
            operations_per_user=steady_ops_per_user,
            test_data=test_data,
            timeout_seconds=config.operation_timeout_seconds
        )
        results["steady_state"] = steady_result
        
        # Phase 3: Peak test (brief spike)
        print("Starting peak load test")
        peak_result = self.run_load_test(
            concurrent_users=min(config.max_concurrent_users * 2, 20),  # 2x users but cap at 20
            operations_per_user=3,  # Fewer operations for peak test
            test_data=test_data,
            timeout_seconds=config.operation_timeout_seconds
        )
        results["peak_load"] = peak_result
        
        return results


class PerformanceTestSuite:
    """Complete performance testing suite."""
    
    def __init__(self):
        self.baseline_results: Dict[str, PerformanceMetrics] = {}
        self.regression_threshold = 1.5  # 50% performance degradation threshold
    
    def run_baseline_tests(self, test_functions: Dict[str, Callable]) -> Dict[str, PerformanceMetrics]:
        """Run baseline performance tests to establish benchmarks."""
        results = {}
        
        for test_name, test_func in test_functions.items():
            print(f"Running baseline test: {test_name}")
            
            # Run test multiple times and take average
            metrics_list = []
            for i in range(3):  # Run 3 times
                runner = LoadTestRunner(test_func, test_name)
                metrics = runner.run_single_operation()
                metrics_list.append(metrics)
            
            # Average the results
            avg_duration = sum(m.duration_ms for m in metrics_list) / len(metrics_list)
            avg_memory = sum(m.memory_usage_mb for m in metrics_list) / len(metrics_list)
            avg_cpu = sum(m.cpu_percent for m in metrics_list) / len(metrics_list)
            
            baseline = PerformanceMetrics(
                duration_ms=avg_duration,
                memory_usage_mb=avg_memory,
                cpu_percent=avg_cpu,
                metadata={"runs": len(metrics_list), "test_type": "baseline"}
            )
            
            results[test_name] = baseline
            self.baseline_results[test_name] = baseline
            
            print(f"  Duration: {avg_duration:.2f}ms")
            print(f"  Memory: {avg_memory:.2f}MB")
            print(f"  CPU: {avg_cpu:.2f}%")
        
        return results
    
    def run_regression_tests(self, test_functions: Dict[str, Callable]) -> Dict[str, Dict[str, Any]]:
        """Run regression tests against baseline."""
        results = {}
        
        for test_name, test_func in test_functions.items():
            if test_name not in self.baseline_results:
                print(f"Warning: No baseline for {test_name}, skipping regression test")
                continue
            
            print(f"Running regression test: {test_name}")
            
            runner = LoadTestRunner(test_func, test_name)
            current_metrics = runner.run_single_operation()
            baseline = self.baseline_results[test_name]
            
            # Calculate regression ratios
            duration_ratio = current_metrics.duration_ms / baseline.duration_ms
            memory_ratio = current_metrics.memory_usage_mb / baseline.memory_usage_mb if baseline.memory_usage_mb > 0 else 1.0
            
            # Determine if regression occurred
            duration_regression = duration_ratio > self.regression_threshold
            memory_regression = memory_ratio > self.regression_threshold
            
            results[test_name] = {
                "current_metrics": current_metrics,
                "baseline_metrics": baseline,
                "duration_ratio": duration_ratio,
                "memory_ratio": memory_ratio,
                "duration_regression": duration_regression,
                "memory_regression": memory_regression,
                "overall_regression": duration_regression or memory_regression
            }
            
            print(f"  Duration ratio: {duration_ratio:.2f}x {'(REGRESSION)' if duration_regression else '(OK)'}")
            print(f"  Memory ratio: {memory_ratio:.2f}x {'(REGRESSION)' if memory_regression else '(OK)'}")
        
        return results
    
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive performance report."""
        report = ["Performance Test Report", "=" * 50, ""]
        
        if "baseline" in results:
            report.extend(["Baseline Performance:", "-" * 20])
            for test_name, metrics in results["baseline"].items():
                report.append(f"{test_name}:")
                report.append(f"  Duration: {metrics.duration_ms:.2f}ms")
                report.append(f"  Memory: {metrics.memory_usage_mb:.2f}MB")
                report.append(f"  CPU: {metrics.cpu_percent:.2f}%")
                report.append("")
        
        if "regression" in results:
            report.extend(["Regression Test Results:", "-" * 25])
            regressions_found = False
            
            for test_name, result in results["regression"].items():
                if result["overall_regression"]:
                    regressions_found = True
                    report.append(f"❌ {test_name}: REGRESSION DETECTED")
                    if result["duration_regression"]:
                        report.append(f"   Duration: {result['duration_ratio']:.2f}x slower")
                    if result["memory_regression"]:
                        report.append(f"   Memory: {result['memory_ratio']:.2f}x more")
                else:
                    report.append(f"✅ {test_name}: PASSED")
                report.append("")
            
            if not regressions_found:
                report.append("🎉 No performance regressions detected!")
                report.append("")
        
        if "load_test" in results:
            report.extend(["Load Test Results:", "-" * 18])
            for test_name, result in results["load_test"].items():
                report.append(f"{test_name}:")
                report.append(f"  Throughput: {result.throughput_ops_per_sec:.2f} ops/sec")
                report.append(f"  P95 Latency: {result.p95_ms:.2f}ms")
                report.append(f"  Error Rate: {result.error_rate_percent:.2f}%")
                report.append(f"  Peak Memory: {result.peak_memory_mb:.2f}MB")
                report.append("")
        
        return "\n".join(report)
