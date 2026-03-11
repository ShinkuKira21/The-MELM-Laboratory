#!/usr/bin/env bash

alias rocm-python='docker exec -w $(pwd) rocm-dev python3' 
alias rocm-jupyter='docker exec -w $(pwd) rocm-dev jupyter notebook --ip=0.0.0.0 --no-browser --allow-root'