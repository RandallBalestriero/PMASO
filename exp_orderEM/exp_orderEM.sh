#perlayer randomm mpopt leakiness

for run in 3 4 5 6 7 8 9 
do
	python exp_orderEM.py $run $1 $2 0 0;
	python exp_orderEM.py $run $1 $2 1 0;
	python exp_orderEM.py $run $1 $2 2 0;
	python exp_orderEM.py $run $1 $2 3 0;
	
	python exp_orderEM.py $run $1 $2 0 None;
	python exp_orderEM.py $run $1 $2 1 None;
	python exp_orderEM.py $run $1 $2 2 None;
	python exp_orderEM.py $run $1 $2 3 None;
done


