#python exp_fc.py 0 1 local local local;

for run in 1 2 3 4 5 6 7 8 9
do
	python exp_nonlinearity.py $run $1 $2 0;
	python exp_nonlinearity.py $run $1 $2 0.01;
	python exp_nonlinearity.py $run $1 $2 -1;
	python exp_nonlinearity.py $run $1 $2 None;
done

