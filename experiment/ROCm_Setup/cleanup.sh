#!/usr/bin/env bash

# cleanup.sh - Remove ROCm Docker container

PROJECT_NAME=$(basename "$(dirname "$PWD")") 
CONTAINER_NAME="rocm-$PROJECT_NAME"

echo "🧹 Cleaning up ROCm Docker container..."

# Check if container exists
if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
    echo "Found container '$CONTAINER_NAME'"
    
    # Stop the container if it's running
    if docker ps --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
        echo "Stopping container..."
        docker stop $CONTAINER_NAME
    fi
    
    # Remove the container
    echo "Removing container..."
    docker rm $CONTAINER_NAME
    
    echo "✅ Container '$CONTAINER_NAME' has been removed"
else
    echo "❌ Container '$CONTAINER_NAME' not found"
fi

# Optional: Remove any dangling images or cleanup Docker system
# Uncomment the line below if you want to do a full cleanup
# echo "🧹 Cleaning up dangling images..."; docker image prune -f

echo "✨ Cleanup complete!"