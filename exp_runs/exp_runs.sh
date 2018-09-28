#!/bin/bash
# supervised traindn randomthetaq neurons randomlearning numberrun
for ((i=0;i<=9;i++))
	do
                python exp_fc_runs.py 1 $1 local local none $i;
#                python exp_fc_runs.py $1 $2 local local none $i;
#                python exp_fc_runs.py $1 $2 local none local $i;
#                python exp_fc_runs.py $1 $2 local none none $i;
	done


