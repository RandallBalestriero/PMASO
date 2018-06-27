#!/bin/bash
# supervised traindn randomthetaq neurons randomlearning numberrun
for k in 16 32 48 64
	do
	for ((i=0;i<=9;i++))
		do
			python exp_fc_runs.py 1 0 $1 $k $2 $i;
			python exp_fc_runs.py 0 0 $1 $k $2 $i;
		done
	done


