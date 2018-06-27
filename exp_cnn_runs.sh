#!/bin/bash
# supervised traindn randomthetaq neurons randomlearning numberrun
for k in 8 16
	do
	for ((i=0;i<=3;i++))
		do
			python exp_cnn_runs.py 1 $1 $k $2 $i;
			python exp_cnn_runs.py 0 $1 $k $2 $i;
		done
	done


