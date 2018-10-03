#python exp_neurons.py local 1;
#python exp_neurons.py global 1;
#python exp_neurons.py local 2;
#python exp_neurons.py global 2;

# DATASET CLASS MODEL OCLUSIONTYPE OCLUSIONSPEC KNOWNY

python exp_oclusion.py MNIST 0 $1 $2 $3 0
#python exp_oclusion.py MNIST 0 $1 $2 $3 1

python exp_oclusion.py CIFAR 0 $1 $2 $3 0
python exp_oclusion.py CIFAR 2 $1 $2 $3 0
python exp_oclusion.py CIFAR 4 $1 $2 $3 0

python exp_oclusion.py CIFAR 0 $1 $2 $3 1
python exp_oclusion.py CIFAR 2 $1 $2 $3 1
python exp_oclusion.py CIFAR 4 $1 $2 $3 1



