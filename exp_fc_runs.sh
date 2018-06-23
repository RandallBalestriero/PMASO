#!/bin/bash
#CUDA_VISIBLE_DEVICES=0;nohup bash -c "(python run_ortho.py CIFAR 0.0001 0 0 > cifar_0001_0_0.out) &> cpython_0001_0_0.out" &

export  CUDA_VISIBLE_DEVICES='1';

for k in 16 32 48 64
	do
	for ((i=0;i<=9;i++))
		do
			python exp_fc_runs.py 0 $k 0 $i;
			python exp_fc_runs.py 0 $k 1 $i;
			python exp_fc_runs.py 1 $k 0 $i;
			python exp_fc_runs.py 1 $k 1 $i;
		done
	done


