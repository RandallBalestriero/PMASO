#neurons Nlayer RESIDUAL sigmas
python exp_resnet.py 32 1 $1 $2;
python exp_resnet.py 64 1 $1 $2;
python exp_resnet.py 32 2 $1 $2;
python exp_resnet.py 64 2 $1 $2;
python exp_resnet.py 32 3 $1 $2;
python exp_resnet.py 64 3 $1 $2;



