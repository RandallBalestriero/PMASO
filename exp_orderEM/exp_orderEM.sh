#perlayer randomm mpopt leakiness
python exp_orderEM.py $1 $2 0 0;
python exp_orderEM.py $1 $2 1 0;
python exp_orderEM.py $1 $2 2 0;
python exp_orderEM.py $1 $2 3 0;

python exp_orderEM.py $1 $2 0 None;
python exp_orderEM.py $1 $2 1 None;
python exp_orderEM.py $1 $2 2 None;
python exp_orderEM.py $1 $2 3 None;



