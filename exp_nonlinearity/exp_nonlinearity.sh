#python exp_fc.py 0 1 local local local;
python exp_nonlinearity.py flippedMNIST $1 0;
python exp_nonlinearity.py MNIST $1 0;

python exp_nonlinearity.py flippedMNIST $1 0.01;
python exp_nonlinearity.py MNIST $1 0.01;

python exp_nonlinearity.py flippedMNIST $1 -1;
python exp_nonlinearity.py MNIST $1 -1;

python exp_nonlinearity.py flippedMNIST $1 None;
python exp_nonlinearity.py MNIST $1 None;


