#!/usr/bin/env bash

# python example_RK.py --epochs 5000 --RK Gauss-Legendre --log-dir ./logs/Gauss-Legendre/ --h 0.1 --N 40 --no-cuda

python example_BE.py --epochs 5000 --log-dir ./logs/Backward-Euler/ --h 0.1 --N 40 --no-cuda
