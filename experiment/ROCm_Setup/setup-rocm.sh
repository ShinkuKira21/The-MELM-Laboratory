#!/usr/bin/env bash

# 1. Stop and remove old container
echo "🧹 Cleaning up old container..."
docker stop rocm-dev 2>/dev/null && docker rm rocm-dev 2>/dev/null

# 3. Start container with proper RDNA4 support
echo "🚀 Starting ROCm 7.1.1 container for Radeon RX 9070XT..."
docker run -d \
  --name rocm-dev \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --group-add=render \
  --ipc=host \
  --network=host \
  -e HSA_OVERRIDE_GFX_VERSION=12.0.1 \
  -v "$(pwd)":/workspace \
  -w /workspace \
  rocm/dev-ubuntu-24.04:7.1.1-complete \
  sleep infinity

# Wait for container to be ready
sleep 3

# 4. Install Python 3.10 and create virtual environment INSIDE CONTAINER (not in mounted volume)
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

# 5. Install PyTorch ROCm - USING ROCm 7.1 INDEX
echo "📦 Installing PyTorch ROCm (from ROCm 7.1 index)..."
docker exec rocm-dev bash -c "
export PATH=/opt/rocm-venv/bin:\$PATH &&
# First, uninstall any existing torch
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true &&
# Install from ROCm 7.1 index (version 2.10.0+rocm7.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.1 --no-cache-dir
"

# 6. Install remaining requirements
echo "📦 Installing remaining Python packages..."
docker exec rocm-dev bash -c "
export PATH=/opt/rocm-venv/bin:\$PATH &&
pip install numpy==1.26.4 pandas==2.2.3 tqdm==4.66.5 psutil==6.1.0 scipy==1.11.4 scikit-learn==1.4.2 matplotlib==3.9.0 jupyter==1.1.1 ipykernel==6.29.5
"

# 7. Create helper script for aliases (UPDATED PATH)
echo "📝 Creating aliases..."
cat > ~/.rocm_aliases << 'EOF'
# ROCm Docker aliases
alias rocm-python="docker exec -w /workspace rocm-dev /opt/rocm-venv/bin/python"
alias rocm-jupyter="docker exec -w /workspace rocm-dev /opt/rocm-venv/bin/jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"
alias rocm-pip="docker exec -w /workspace rocm-dev /opt/rocm-venv/bin/pip"
alias rocm-bash="docker exec -it rocm-dev bash"
EOF

# 8. Set up aliases properly in current session and bashrc
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
alias rocm-python="docker exec -w /workspace rocm-dev /opt/rocm-venv/bin/python"
alias rocm-jupyter="docker exec -w /workspace rocm-dev /opt/rocm-venv/bin/jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"
alias rocm-pip="docker exec -w /workspace rocm-dev /opt/rocm-venv/bin/pip"
alias rocm-bash="docker exec -it rocm-dev bash"

# 9. Verify aliases are set correctly
echo "✅ Aliases set:"
alias rocm-python 2>/dev/null && echo "  rocm-python ✓" || echo "  rocm-python ✗"
alias rocm-jupyter 2>/dev/null && echo "  rocm-jupyter ✓" || echo "  rocm-jupyter ✗"
alias rocm-pip 2>/dev/null && echo "  rocm-pip ✓" || echo "  rocm-pip ✗"
alias rocm-bash 2>/dev/null && echo "  rocm-bash ✓" || echo "  rocm-bash ✗"

# 10. Test the installation
echo "🧪 Testing PyTorch installation..."
docker exec rocm-dev /opt/rocm-venv/bin/python -c "
import torch
print(f'✅ PyTorch {torch.__version__} installed successfully')
print(f'   ROCm available: {torch.cuda.is_available()}')
print(f'   ROCm version: {getattr(torch.version, \"hip\", \"Not available\")}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
"

# 11. Also test with a simple tensor operation
echo "🧪 Testing GPU tensor operation..."
docker exec rocm-dev /opt/rocm-venv/bin/python -c "
import torch
if torch.cuda.is_available():
    x = torch.randn(3,3).cuda()
    print(f'✅ Successfully created tensor on GPU: {x.device}')
else:
    print('❌ GPU not available')
    exit(1)
"

# 12. Create a simple test file in current directory (this is fine - it's just a script)
cat > test_gpu.py << 'EOF'
import torch
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    x = torch.randn(3,3).cuda()
    print(f"Tensor on: {x.device}")
EOF

# 13. Final warning about local venv
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