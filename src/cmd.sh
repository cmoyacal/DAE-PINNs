#!/usr/bin/env bash

python example_powerNet.py --log-dir ./logs/dae-pinns-searching-model-w-o-fourier/ --num-train 2000 --num-test 500 --use-scheduler \
	--patience 2000 --use-tqdm --batch-size 1000 --unstacked --dyn-depth 4 --alg-width 100 --h 0.1 --N 40 \
	--epochs 20000 --alg-type attention --dyn-type attention \
	--dyn-weight 64.0 --start-from-best --lr 1e-5



#python example_powerNet.py --log-dir ./logs/dae-pinns-searching-model/ --num-train 2000 --num-test 500 --use-scheduler \
#	--patience 2000 --use-tqdm --batch-size 1000 --unstacked --dyn-depth 4 --alg-width 100 --h 0.1 --N 40 \
#	--alg-type fnn --dyn-type fnn --epochs 50000

# python example_powerNet.py --log-dir ./logs/dae-pinns-searching-model/ --num-train 2000 --num-test 500 --use-scheduler \
#	--patience 2000 --use-tqdm --batch-size 1000 --unstacked --dyn-depth 4 --alg-width 100 --h 0.1 --N 40 \
#	--epochs 40000 --alg-type attention --dyn-type attention --use-input-layer --dyn-weight 16.0 --start-from-best 
