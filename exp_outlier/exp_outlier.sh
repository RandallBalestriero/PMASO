#python exp_fc.py 0 1 local local local;
python exp_outlier.py $1 $2 0;
python exp_outlier.py $1 $2 1;

#python exp_fa.py global 1 $1;
#python exp_fa.py global 2 $1;
#python exp_fa.py local 0 $1;
#python exp_fa.py local 1 $1;
#python exp_fa.py local 2 $1;


