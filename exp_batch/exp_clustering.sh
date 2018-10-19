#python exp_fc.py 0 1 local local local;
python exp_clustering.py $1 10;
python exp_clustering.py $1 30;
python exp_clustering.py $1 50;


