#python exp_fc.py 0 1 local local local;
python exp_dense_cifar.py $1 $2 0;
python exp_dense_cifar.py $1 $2 1;
python exp_dense_cifar.py $1 $2 2;
python exp_dense_cifar.py $1 $2 3;
python exp_dense_cifar.py $1 $2 4;
python exp_dense_cifar.py $1 $2 5;
python exp_dense_cifar.py $1 $2 6;
python exp_dense_cifar.py $1 $2 7;
python exp_dense_cifar.py $1 $2 8;
python exp_dense_cifar.py $1 $2 9;

