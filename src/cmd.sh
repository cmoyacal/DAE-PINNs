#!/usr/bin/env bash

#python example_powerNet.py --log-dir ./logs/dae-pinns-searching-model-w-o-fourier/ --num-train 2000 --num-test 500 --use-scheduler \
#	--patience 2000 --use-tqdm --batch-size 1000 --unstacked --dyn-depth 4 --alg-width 100 --h 0.1 --N 40 \
#	--epochs 20000 --alg-type attention --dyn-type attention \
#	--dyn-weight 64.0 --start-from-best --lr 1e-5

#python example_powerNet.py --log-dir ./logs/dae-pinns-best-model/ --num-train 4000 --num-test 500 --use-scheduler \
#        --patience 2000 --use-tqdm --batch-size 1000 --unstacked --dyn-depth 5 --alg-width 100 --h 0.1 --N 80 \
#        --epochs 20000 --alg-type attention --dyn-type attention \
#        --dyn-weight 64.0 --lr 4e-6 --start-from-best 

#python example_powerNet.py --log-dir ./logs/dae-pinns-searching-model/ --num-train 2000 --num-test 500 --use-scheduler \
#	--patience 2000 --use-tqdm --batch-size 1000 --unstacked --dyn-depth 4 --alg-width 100 --h 0.1 --N 40 \
#	--alg-type fnn --dyn-type fnn --epochs 50000

# python example_powerNet.py --log-dir ./logs/dae-pinns-searching-model/ --num-train 2000 --num-test 500 --use-scheduler \
#	--patience 2000 --use-tqdm --batch-size 1000 --unstacked --dyn-depth 4 --alg-width 100 --h 0.1 --N 40 \
#	--epochs 40000 --alg-type attention --dyn-type attention --use-input-layer --dyn-weight 16.0 --start-from-best

#python run_stacked_vs_unstacked.py --log-dir ./logs/final-test-stacked-unstacked/ --dyn-width 100 --alg-width 25 --dyn-depth 3 \
#       	--alg-width 2 --h 0.1 --num-train 2000 --num-test 2500 --batch-size 2000 --use-tqdm --alg-type attention --dyn-type attention \
#	--epochs 50000 --test-every 1000

#python run_depth_analysis.py --epochs 50000 --unstacked --dyn-width 100 --alg-width 40 --dyn-type attention --alg-type attention \
#	--dyn-activation sin --alg-activation sin --num-train 1000 --num-test 1500 --batch 1000 --test-every 1000 \
#	--h 0.1 --use-tqdm --log ./logs/final-test-depth/

python run_width_analysis.py --epochs 50000 --unstacked --dyn-depth 4 --alg-depth 4 --dyn-type attention --alg-type attention \
	--dyn-activation sin --alg-activation sin --num-train 1000 --num-test 1500 --batch 1000 --test-every 1000 \
	--h 0.1 --use-tqdm --log ./logs/final-test-width/
