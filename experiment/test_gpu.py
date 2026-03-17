import torch
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    count = torch.cuda.device_count()
    print(f"Detected {count} device(s):")
    
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1e9
        name = props.name
        
        print(f"  GPU {i}: {name} ({mem_gb:.2f} GB)")
        
        # Logic check: If memory > 25GB, it's likely the CPU/iGPU appearing as a GPU
        if props.total_memory > 25e9:
            print(f"  ⚠️  WARNING: Device {i} has >25GB memory. This might be your CPU/iGPU, not the Radeon GPU.")
        else:
            print(f"  ✅ Device {i} looks like a dedicated GPU.")
            
    # Default to GPU 0 for the test tensor
    x = torch.randn(3,3).cuda()
    print(f"\nTest tensor created on: {x.device}")
