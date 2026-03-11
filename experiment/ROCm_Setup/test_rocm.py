#!/usr/bin/env python3
"""
ROCm Docker Test Script
Tests GPU availability and basic operations with PyTorch
"""

import torch
import torch.cuda as cuda
import platform
import sys
import subprocess
import psutil
from datetime import datetime

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {text}")
    print("="*60)

def check_rocm_smi():
    """Try to run rocm-smi to get GPU info"""
    try:
        result = subprocess.run(['rocm-smi', '--showproductname'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("\nROCm SMI Output:")
            print(result.stdout)
        else:
            print("⚠️  Could not run rocm-smi")
    except FileNotFoundError:
        print("⚠️  rocm-smi not found in PATH")

def test_pytorch_basic():
    """Test basic PyTorch operations on GPU"""
    print_header("PyTorch GPU Test")
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability (CUDA in PyTorch = ROCm support)
    cuda_available = torch.cuda.is_available()
    print(f"CUDA (ROCm) available: {cuda_available}")
    
    if cuda_available:
        # GPU count
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")
        
        # GPU properties
        for i in range(gpu_count):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  - Compute Capability: {props.major}.{props.minor}")
            print(f"  - Total Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  - Multiprocessors: {props.multi_processor_count}")
        
        # Basic tensor operations on GPU
        print("\nPerforming tensor operations on GPU...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        
        # Matrix multiplication
        z = torch.matmul(x, y)
        
        # Move back to CPU and compute sum
        result = z.cpu().sum().item()
        print(f"✓ Matrix multiplication successful")
        print(f"  Result sum: {result:.4f}")
        
        # Memory info
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
    else:
        print("❌ CUDA/ROCm not available!")
        print("  Check if:")
        print("  - GPU device passed correctly to container")
        print("  - ROCm drivers installed properly")

def test_tensorflow_if_available():
    """Test TensorFlow GPU support if installed"""
    try:
        import tensorflow as tf
        print_header("TensorFlow GPU Test")
        print(f"TensorFlow version: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ TensorFlow detected {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"  - {gpu}")
            
            # Simple tensor operation
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print(f"✓ TensorFlow GPU operation successful")
                print(f"  Result:\n{c.numpy()}")
        else:
            print("❌ No GPUs detected by TensorFlow")
    except ImportError:
        print("\nℹ️  TensorFlow not installed, skipping")

def test_memory_bandwidth():
    """Simple memory bandwidth test"""
    print_header("Simple Memory Bandwidth Test")
    
    if not torch.cuda.is_available():
        print("❌ Skipping - GPU not available")
        return
    
    # Create large tensors and measure transfer time
    sizes = [1000, 5000, 10000]
    
    for size in sizes:
        print(f"\nTesting {size}x{size} matrix ({size*size*4/1e6:.1f}M elements)...")
        
        # CPU tensor
        cpu_tensor = torch.randn(size, size)
        
        # Time CPU -> GPU transfer
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        gpu_tensor = cpu_tensor.cuda()
        end.record()
        
        torch.cuda.synchronize()
        transfer_time = start.elapsed_time(end)
        
        # Calculate bandwidth (in GB/s)
        data_size = cpu_tensor.element_size() * cpu_tensor.nelement() / 1e9  # GB
        bandwidth = data_size / (transfer_time / 1000)  # GB/s
        
        print(f"  Transfer time: {transfer_time:.2f} ms")
        print(f"  Bandwidth: {bandwidth:.2f} GB/s")
        
        # Simple computation
        start.record()
        result = torch.matmul(gpu_tensor, gpu_tensor.T)
        end.record()
        
        torch.cuda.synchronize()
        compute_time = start.elapsed_time(end)
        print(f"  Matrix multiply time: {compute_time:.2f} ms")

def main():
    """Main test function"""
    print_header("ROCm Docker Environment Test")
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {platform.python_version()}")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor()}")
    
    # System memory
    mem = psutil.virtual_memory()
    print(f"System Memory: {mem.total / 1e9:.2f} GB")
    
    # Check ROCm system info
    check_rocm_smi()
    
    # Run PyTorch tests
    test_pytorch_basic()
    
    # Run bandwidth test if GPU available
    if torch.cuda.is_available():
        test_memory_bandwidth()
    
    # Optional TensorFlow test
    test_tensorflow_if_available()
    
    print_header("Test Complete")
    
    # Return success if GPU available
    if torch.cuda.is_available():
        print("✅ ROCm setup is working correctly!")
        return 0
    else:
        print("❌ ROCm GPU not detected. Check your Docker command.")
        return 1

if __name__ == "__main__":
    sys.exit(main())