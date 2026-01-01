"""
Shared Hardware Detection Module
Hardware detection for Windows/Linux/Colab/WSL

Migrated from cardiac-ml-research/shared/hardware/
"""

from cardiac_shared.hardware.detector import (
    # Data classes
    GPUInfo,
    CPUInfo,
    RAMInfo,
    EnvironmentInfo,
    HardwareInfo,

    # Detection functions
    detect_gpu,
    detect_cpu,
    detect_ram,
    detect_environment,
    detect_hardware,

    # Utility functions
    print_hardware_summary,
    get_optimal_config,
)

from cardiac_shared.hardware.cpu_optimizer import (
    # CPU Optimization
    CPUTier,
    CPUOptimizationConfig,
    CPUOptimizer,
    get_cpu_optimizer,
    get_optimal_cpu_config,
    apply_cpu_optimizations,
)

__all__ = [
    # Data classes
    'GPUInfo',
    'CPUInfo',
    'RAMInfo',
    'EnvironmentInfo',
    'HardwareInfo',

    # Detection functions
    'detect_gpu',
    'detect_cpu',
    'detect_ram',
    'detect_environment',
    'detect_hardware',

    # Utility functions
    'print_hardware_summary',
    'get_optimal_config',

    # CPU Optimization
    'CPUTier',
    'CPUOptimizationConfig',
    'CPUOptimizer',
    'get_cpu_optimizer',
    'get_optimal_cpu_config',
    'apply_cpu_optimizations',
]
