#!/usr/bin/env bash
PROJECT_NAME=$(basename "$(dirname "$PWD")") 
CONTAINER_NAME="rocm-$PROJECT_NAME"

docker stop $CONTAINER_NAME