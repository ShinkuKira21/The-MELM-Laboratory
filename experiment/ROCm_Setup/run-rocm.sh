#!/usr/bin/env bash

# run-rocm.sh - Start and connect to existing ROCm container

echo "🚀 Checking for existing ROCm container..."

# Check if container exists
if docker ps -a --format '{{.Names}}' | grep -q "^rocm-dev$"; then
    echo "✅ Found container 'rocm-dev'"
    
    # Check if container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^rocm-dev$"; then
        echo "🔄 Starting existing container..."
        docker start rocm-dev
    else
        echo "✅ Container already running"
    fi
else
    echo "❌ No container found. Please run setup-rocm.sh first."
    exit 1
fi

# Ensure aliases are loaded
if [ -f ~/.rocm_aliases ]; then
    source ~/.rocm_aliases
    echo "✅ Aliases loaded"
else
    # Create aliases if they don't exist
    cat > ~/.rocm_aliases << 'EOF'
# ROCm Docker aliases
alias rocm-python="docker exec -w /workspace rocm-dev /opt/rocm-venv/bin/python"
alias rocm-jupyter="docker exec -w /workspace rocm-dev /opt/rocm-venv/bin/jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"
alias rocm-pip="docker exec -w /workspace rocm-dev /opt/rocm-venv/bin/pip"
alias rocm-bash="docker exec -it rocm-dev bash"
EOF
    source ~/.rocm_aliases
    echo "✅ Aliases created and loaded"
fi

# Print status
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

# Optional: Run a quick test
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