#!/usr/bin/env bash

# 1. Stop and remove old container
echo "🧹 Cleaning up old container..."
docker stop rocm-dev 2>/dev/null && docker rm rocm-dev 2>/dev/null

# 2. Start container
# FIX: Removed global -e HSA_OVERRIDE_GFX_VERSION to prevent CPU/iGPU detection
echo "🚀 Starting ROCm 7.1.1 container for Radeon RX 9070XT..."
docker run -d \
  --name rocm-dev \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --group-add=render \
  --ipc=host \
  --network=host \
  -v "$(pwd)":/workspace \
  -w /workspace \
  rocm/dev-ubuntu-24.04:7.1.1-complete \
  sleep infinity

# Wait for container to be ready
sleep 3

# 3. Install Python 3.10 and create virtual environment INSIDE CONTAINER (not in mounted volume)
echo "🐍 Setting up Python 3.10 virtual environment inside container..."
docker exec rocm-dev bash -c "
apt update &&
apt install -y software-properties-common &&
add-apt-repository -y ppa:deadsnakes/ppa &&
apt update &&
apt install -y python3.10 python3.10-venv python3.10-dev &&
# Create venv in container's root, NOT in mounted workspace
python3.10 -m venv /opt/rocm-venv &&
# Upgrade pip using the venv's Python
/opt/rocm-venv/bin/python -m pip install --upgrade pip setuptools wheel
"

# 4. Install PyTorch ROCm - USING ROCm 7.1 INDEX
echo "📦 Installing PyTorch ROCm (from ROCm 7.1 index)..."
docker exec rocm-dev bash -c "
export PATH=/opt/rocm-venv/bin:\$PATH &&
# First, uninstall any existing torch
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true &&
# Install from ROCm 7.1 index (version 2.10.0+rocm7.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.1 --no-cache-dir
"

# 4.1 FIX: Create Python wrapper to apply GFX override ONLY to Python
echo "🔧 Creating Python wrapper with targeted GFX override..."
docker exec rocm-dev bash -c "
cat > /opt/rocm-venv/bin/python_wrapper.sh << 'EOF'
#!/bin/bash
# Apply RDNA4 (GFX12) override for the GPU only
export HSA_OVERRIDE_GFX_VERSION=12.0.1
export HSA_ENABLE_SDMA=0
exec /opt/rocm-venv/bin/python \"\$@\"
EOF
chmod +x /opt/rocm-venv/bin/python_wrapper.sh
"

# 5. Install remaining requirements
echo "📦 Installing remaining Python packages..."
docker exec rocm-dev bash -c "
export PATH=/opt/rocm-venv/bin:\$PATH &&
pip install numpy==1.26.4 pandas==2.2.3 tqdm==4.66.5 psutil==6.1.0 scipy==1.11.4 scikit-learn==1.4.2 matplotlib==3.9.0 jupyter==1.1.1 ipykernel==6.29.5
"
x
# 6. Create helper script for aliases (UPDATED PATH)
# FIX: Pointed aliases to python_wrapper.sh instead of python
echo "📝 Creating aliases..."
cat > ~/.rocm_aliases << 'EOF'
# ROCm Docker aliases
alias rocm-python="docker exec -w /workspace rocm-dev /opt/rocm-venv/bin/python_wrapper.sh"
alias rocm-jupyter="docker exec -w /workspace rocm-dev /opt/rocm-venv/bin/python_wrapper.sh -m jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"
alias rocm-pip="docker exec -w /workspace rocm-dev /opt/rocm-venv/bin/pip"
alias rocm-bash="docker exec -it rocm-dev bash"
EOF

# 7. Set up aliases properly in current session and bashrc
echo "🔧 Setting up aliases in current session and bashrc..."

# Remove any existing alias definitions from bashrc
sed -i '/alias rocm-python=/d' ~/.bashrc 2>/dev/null
sed -i '/alias rocm-jupyter=/d' ~/.bashrc 2>/dev/null
sed -i '/alias rocm-pip=/d' ~/.bashrc 2>/dev/null
sed -i '/alias rocm-bash=/d' ~/.bashrc 2>/dev/null
sed -i '/source ~\/.rocm_aliases/d' ~/.bashrc 2>/dev/null

# Add source line to bashrc
echo "source ~/.rocm_aliases" >> ~/.bashrc

# Source the aliases for current session
source ~/.rocm_aliases

# Also set aliases directly in current session (backup)
# FIX: Updated these to use the wrapper too
alias rocm-python="docker exec -w /workspace rocm-dev /opt/rocm-venv/bin/python_wrapper.sh"
alias rocm-jupyter="docker exec -w /workspace rocm-dev /opt/rocm-venv/bin/python_wrapper.sh -m jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"
alias rocm-pip="docker exec -w /workspace rocm-dev /opt/rocm-venv/bin/pip"
alias rocm-bash="docker exec -it rocm-dev bash"

# 8. Verify aliases are set correctly
echo "✅ Aliases set:"
alias rocm-python 2>/dev/null && echo "  rocm-python ✓" || echo "  rocm-python ✗"
alias rocm-jupyter 2>/dev/null && echo "  rocm-jupyter ✓" || echo "  rocm-jupyter ✗"
alias rocm-pip 2>/dev/null && echo "  rocm-pip ✓" || echo "  rocm-pip ✗"
alias rocm-bash 2>/dev/null && echo "  rocm-bash ✓" || echo "  rocm-bash ✗"

# 9. Test the installation
# FIX: Using python_wrapper.sh for the test
echo "🧪 Testing PyTorch installation..."
docker exec rocm-dev /opt/rocm-venv/bin/python_wrapper.sh -c "
import torch
print(f'✅ PyTorch {torch.__version__} installed successfully')
print(f'   ROCm available: {torch.cuda.is_available()}')
print(f'   ROCm version: {getattr(torch.version, \"hip\", \"Not available\")}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
"

# 10. Also test with a simple tensor operation
# FIX: Using python_wrapper.sh for the test
echo "🧪 Testing GPU tensor operation..."
docker exec rocm-dev /opt/rocm-venv/bin/python_wrapper.sh -c "
import torch
if torch.cuda.is_available():
    x = torch.randn(3,3).cuda()
    print(f'✅ Successfully created tensor on GPU: {x.device}')
else:
    print('❌ GPU not available')
    exit(1)
"

# 11. Create a simple test file in current directory (this is fine - it's just a script)
# Added the memory check logic here too
cat > test_gpu.py << 'EOF'
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
EOF

# 12. Final warning about local venv
echo ""
echo "⚠️  IMPORTANT: Do NOT activate or use the local venv directory!"
echo "   The venv/ directory you see is just a mount point from the container."
echo "   Always use the 'rocm-python' alias to run code in the container."
echo ""

# Print success message
echo ""
echo "========================================================"
echo "✅ ROCm container for Radeon RX 9070XT (RDNA4) is ready!"
echo "========================================================"
echo ""
echo "Commands available NOW in this terminal:"
echo "  rocm-python script.py  - Run Python script in container"
echo "  rocm-jupyter           - Start Jupyter notebook"
echo "  rocm-pip install ...   - Install more packages"
echo "  rocm-bash              - Enter container interactively"
echo ""
echo "Test with: rocm-python test_gpu.py"
echo ""
echo "Access Jupyter at: http://localhost:8888"
echo ""
echo "To cleanup later:"
echo "  docker stop rocm-dev && docker rm rocm-dev"
echo "========================================================"