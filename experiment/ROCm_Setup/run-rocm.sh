#!/usr/bin/env bash

# run-rocm.sh - Start and connect to existing ROCm container

# ----------------------------------------
# Config
# ----------------------------------------
PROJECT_NAME=$(basename "$(dirname "$PWD")") 
CONTAINER_NAME="rocm-$PROJECT_NAME"

echo "🚀 Checking for existing ROCm container..."

# Check if container exists
if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
    echo "✅ Found container '$CONTAINER_NAME'"
    
    # Check if container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
        echo "🔄 Starting existing container..."
        docker start "$CONTAINER_NAME"
    else
        echo "✅ Container already running"
    fi
else
    echo "❌ No container found. Please run setup-rocm.sh first."
    exit 1
fi

# ----------------------------------------
# Temporary aliases for this session
# ----------------------------------------
echo "🔧 Setting up temporary aliases for this session..."

alias rocm-python="docker exec -it -w /workspace $CONTAINER_NAME /opt/rocm-venv/bin/python_wrapper.sh"
alias rocm-jupyter="docker exec -w /workspace $CONTAINER_NAME /opt/rocm-venv/bin/python_wrapper.sh -m jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"
alias rocm-pip="docker exec -w /workspace $CONTAINER_NAME /opt/rocm-venv/bin/pip"
alias rocm-bash="docker exec -it $CONTAINER_NAME bash"

# Verify aliases
echo "✅ Aliases set:"
alias rocm-python 2>/dev/null && echo "  rocm-python ✓" || echo "  rocm-python ✗"
alias rocm-jupyter 2>/dev/null && echo "  rocm-jupyter ✓" || echo "  rocm-jupyter ✗"
alias rocm-pip 2>/dev/null && echo "  rocm-pip ✓" || echo "  rocm-pip ✗"
alias rocm-bash 2>/dev/null && echo "  rocm-bash ✓" || echo "  rocm-bash ✗"

# ----------------------------------------
# Status and usage info
# ----------------------------------------
echo ""
echo "========================================================"
echo "✅ ROCm container is ready!"
echo "========================================================"
echo ""
echo "Available commands:"
echo "  rocm-python script.py  - Run Python script"
echo "  rocm-jupyter           - Start Jupyter notebook"
echo "  rocm-pip install ...   - Install Python packages"
echo "  rocm-bash              - Enter container shell"
echo ""
echo "Quick test:"
echo "  rocm-python -c \"import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')\""
echo ""
echo "Access Jupyter at: http://localhost:8888"
echo "========================================================"

# ----------------------------------------
# Optional: Quick GPU test
# ----------------------------------------
if [ "$1" == "--test" ]; then
    echo ""
    echo "🧪 Running quick GPU test..."
    rocm-python -c "
import torch
print(f'✅ PyTorch {torch.__version__}')
print(f'✅ ROCm available: {torch.cuda.is_available()}')
print(f'✅ GPU 0: {torch.cuda.get_device_name(0)}')
if torch.cuda.device_count() > 1:
    print(f'✅ GPU 1: {torch.cuda.get_device_name(1)}')
"
fi