#!/usr/bin/env bash

python example_RK.py --epochs 25000 --RK Gauss-Legendre --log-dir ./logs/Gauss-Legendre/ --h 0.1 --N 80 \
       --dyn-weight 64.0 --lr 1e-4 --start-from-best

# python example_BE.py --epochs 25000 --log-dir ./logs/Backward-Euler/ --h 0.1 --N 80 --no-cuda --start-from-best \
	#--dyn-weight 32.0 --lr 3e-4
