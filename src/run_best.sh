#!/usr/bin/env bash

python example_powerNet.py --log-dir ./logs/dae-pinns-best/ --num-test 500 --use-scheduler --patience 2000 --batch-size 1048 \
	--unstacked --dyn-depth 4 --dyn-width 100 --h 0.1 --N 80 --dyn-type attention --alg-type attention --dyn-activation sin \
	--alg-activation sin --test-every 1000 --scheduler-type plateau --alg-weight 1.0 --num-train 6000 --num-val 100 \
	--use-tqdm --num-test 500 \
	--dyn-weight 64.0 --epochs 30000 --start-from-best --lr 1e-4 
	
	#--dyn-weight 1.0 --epochs 50000 
	#--dyn-weight 2.0 --epochs 20000 --start-from-best 
	#--dyn-weight 4.0 --epochs 20000 --start-from-best --lr 8e-4
	#--dyn-weight 8.0 --epochs 25000 --start-from-best --lr 5e-4
	#--dyn-weight 16.0 --epochs 25000 --start-from-best --lr 4e-4
	#--dyn-weight 32.0 --epochs 25000 --start-from-best --lr 3e-4
	#--dyn-weight 64.0 --epochs 30000 --start-from-best --lr 1e-4 --num-train from 4000 to 6000
	
