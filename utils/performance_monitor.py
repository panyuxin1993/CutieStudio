#!/usr/bin/env python3
"""
Performance monitoring utility for pairwise metrics calculation.
"""

import time
import threading
from typing import Dict, List, Optional
import numpy as np

# Try to import psutil, but make it optional
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Performance monitoring will be limited.")

class PerformanceMonitor:
    """Monitor CPU, memory, and processing performance"""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.monitoring = False
        self.monitor_thread = None
        self.stats = {
            'cpu_percent': [],
            'memory_percent': [],
            'memory_mb': [],
            'timestamps': [],
            'frame_counts': [],
            'processing_speeds': []
        }
        self.start_time = None
        self.last_frame_count = 0
        self.last_check_time = None
        
    def start_monitoring(self):
        """Start performance monitoring in background thread"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.start_time = time.time()
        self.last_check_time = self.start_time
        self.stats = {
            'cpu_percent': [],
            'memory_percent': [],
            'memory_mb': [],
            'timestamps': [],
            'frame_counts': [],
            'processing_speeds': []
        }
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"Performance monitoring started (interval: {self.interval}s)")
        
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("Performance monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                current_time = time.time()
                elapsed = current_time - self.start_time
                
                # Get current system stats (with fallback if psutil not available)
                if PSUTIL_AVAILABLE:
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory()
                    
                    # Store stats
                    self.stats['cpu_percent'].append(cpu_percent)
                    self.stats['memory_percent'].append(memory.percent)
                    self.stats['memory_mb'].append(memory.used / (1024 * 1024))
                else:
                    # Fallback: use placeholder values
                    self.stats['cpu_percent'].append(0.0)
                    self.stats['memory_percent'].append(0.0)
                    self.stats['memory_mb'].append(0.0)
                
                self.stats['timestamps'].append(elapsed)
                
                # Calculate processing speed if we have frame count updates
                if hasattr(self, 'current_frame_count'):
                    frames_processed = self.current_frame_count - self.last_frame_count
                    time_diff = current_time - self.last_check_time
                    if time_diff > 0:
                        speed = frames_processed / time_diff
                        self.stats['processing_speeds'].append(speed)
                        self.stats['frame_counts'].append(self.current_frame_count)
                    else:
                        self.stats['processing_speeds'].append(0)
                        self.stats['frame_counts'].append(self.current_frame_count)
                    
                    self.last_frame_count = self.current_frame_count
                    self.last_check_time = current_time
                
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"Performance monitoring error: {e}")
                time.sleep(self.interval)
                
    def update_frame_count(self, frame_count: int):
        """Update the current frame count for speed calculation"""
        self.current_frame_count = frame_count
        
    def get_current_stats(self) -> Dict:
        """Get current performance statistics"""
        if not self.stats['timestamps']:
            return {}
            
        return {
            'current_cpu_percent': self.stats['cpu_percent'][-1] if self.stats['cpu_percent'] else 0,
            'current_memory_percent': self.stats['memory_percent'][-1] if self.stats['memory_percent'] else 0,
            'current_memory_mb': self.stats['memory_mb'][-1] if self.stats['memory_mb'] else 0,
            'avg_cpu_percent': np.mean(self.stats['cpu_percent']) if self.stats['cpu_percent'] else 0,
            'avg_memory_percent': np.mean(self.stats['memory_percent']) if self.stats['memory_percent'] else 0,
            'avg_processing_speed': np.mean(self.stats['processing_speeds']) if self.stats['processing_speeds'] else 0,
            'max_processing_speed': np.max(self.stats['processing_speeds']) if self.stats['processing_speeds'] else 0,
            'total_frames_processed': self.stats['frame_counts'][-1] if self.stats['frame_counts'] else 0,
            'elapsed_time': self.stats['timestamps'][-1] if self.stats['timestamps'] else 0
        }
        
    def print_summary(self):
        """Print performance summary"""
        stats = self.get_current_stats()
        if not stats:
            print("No performance data available")
            return
            
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Total frames processed: {stats['total_frames_processed']}")
        print(f"Elapsed time: {stats['elapsed_time']:.2f} seconds")
        print(f"Average processing speed: {stats['avg_processing_speed']:.2f} frames/second")
        print(f"Peak processing speed: {stats['max_processing_speed']:.2f} frames/second")
        print(f"Average CPU usage: {stats['avg_cpu_percent']:.1f}%")
        print(f"Current CPU usage: {stats['current_cpu_percent']:.1f}%")
        if PSUTIL_AVAILABLE:
            print(f"Average memory usage: {stats['avg_memory_percent']:.1f}% ({stats['avg_memory_percent']*psutil.virtual_memory().total/(100*1024*1024):.1f} MB)")
            print(f"Current memory usage: {stats['current_memory_percent']:.1f}% ({stats['current_memory_mb']:.1f} MB)")
        else:
            print(f"Average memory usage: {stats['avg_memory_percent']:.1f}% (psutil not available)")
            print(f"Current memory usage: {stats['current_memory_percent']:.1f}% (psutil not available)")
        
        # Performance recommendations
        print("\nPERFORMANCE ANALYSIS:")
        if PSUTIL_AVAILABLE:
            if stats['avg_cpu_percent'] < 50:
                print("⚠️  Low CPU usage - consider increasing parallelization")
            elif stats['avg_cpu_percent'] > 90:
                print("⚠️  High CPU usage - consider reducing parallelization")
            else:
                print("✓ CPU usage is optimal")
                
            if stats['avg_memory_percent'] > 80:
                print("⚠️  High memory usage - consider reducing batch size")
            else:
                print("✓ Memory usage is acceptable")
        else:
            print("⚠️  Performance monitoring limited - psutil not available")
            print("   Install psutil for detailed CPU/memory analysis: pip install psutil")
            
        if stats['avg_processing_speed'] < 1.0:
            print("⚠️  Low processing speed - check for I/O bottlenecks")
        elif stats['avg_processing_speed'] > 10.0:
            print("✓ Processing speed is good")
        else:
            print("✓ Processing speed is acceptable")
            
        print("="*60)

def create_performance_monitor(interval: float = 1.0) -> PerformanceMonitor:
    """Create a performance monitor instance"""
    return PerformanceMonitor(interval)

# Global monitor instance for easy access
_global_monitor: Optional[PerformanceMonitor] = None

def start_global_monitoring(interval: float = 1.0):
    """Start global performance monitoring"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor(interval)
    _global_monitor.start_monitoring()

def stop_global_monitoring():
    """Stop global performance monitoring"""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop_monitoring()

def update_global_frame_count(frame_count: int):
    """Update global frame count"""
    global _global_monitor
    if _global_monitor:
        _global_monitor.update_frame_count(frame_count)

def get_global_stats() -> Dict:
    """Get global performance stats"""
    global _global_monitor
    if _global_monitor:
        return _global_monitor.get_current_stats()
    return {}

def print_global_summary():
    """Print global performance summary"""
    global _global_monitor
    if _global_monitor:
        _global_monitor.print_summary() 