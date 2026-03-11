#!/usr/bin/env bash

# cleanup.sh - Remove ROCm Docker container

echo "🧹 Cleaning up ROCm Docker container..."

# Check if container exists
if docker ps -a --format '{{.Names}}' | grep -q "^rocm-dev$"; then
    echo "Found container 'rocm-dev'"
    
    # Stop the container if it's running
    if docker ps --format '{{.Names}}' | grep -q "^rocm-dev$"; then
        echo "Stopping container..."
        docker stop rocm-dev
    fi
    
    # Remove the container
    echo "Removing container..."
    docker rm rocm-dev
    
    echo "✅ Container 'rocm-dev' has been removed"
else
    echo "❌ Container 'rocm-dev' not found"
fi

# Optional: Remove any dangling images or cleanup Docker system
# Uncomment the line below if you want to do a full cleanup
# echo "🧹 Cleaning up dangling images..."; docker image prune -f

echo "✨ Cleanup complete!"