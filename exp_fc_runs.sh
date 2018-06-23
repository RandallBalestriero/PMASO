export  CUDA_VISIBLE_DEVICES='1';

for k in 16 32 48 64
	do
	for ((i=0;i<=9;i++))
		do
			python exp_fc_runs.py $k 0 $i;
		done
	done
for k in 16 32 48 64
	do
	for ((i=0;i<=9;i++))
		do
			python exp_fc_runs.py $k 1 $i;
		done
	done





