#python exp_neurons.py local 1;
#python exp_neurons.py global 1;
#python exp_neurons.py local 2;
#python exp_neurons.py global 2;

python exp_oclusion.py 0 $1 $2 local 1;
python exp_oclusion.py 0 $1 $2 global 1;

python exp_oclusion.py 1 $1 $2 local 1;
python exp_oclusion.py 1 $1 $2 global 1;

python exp_oclusion.py 0 $1 $2 local 2;
python exp_oclusion.py 0 $1 $2 global 2;

python exp_oclusion.py 1 $1 $2 local 2;
python exp_oclusion.py 1 $1 $2 global 2;

python exp_oclusion.py 0 $1 $2 local 3;
python exp_oclusion.py 0 $1 $2 global 3;

python exp_oclusion.py 1 $1 $2 local 3;
python exp_oclusion.py 1 $1 $2 global 3;





