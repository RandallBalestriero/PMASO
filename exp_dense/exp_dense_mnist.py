from pylab import *
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
import tensorflow as tf

from layers import *
from utils import *

import cPickle
import os
SAVE_DIR = os.environ['SAVE_DIR']

DATASET = 'MNIST'

neuronsss=10
sigmass=sys.argv[-2]
#sparsity  = float32(sys.argv[-1])
nonlinearity = int(sys.argv[-1])
print nonlinearity
if(nonlinearity=='none'):
	nonlinearity=None

supss     = 1


x_train,y_train,x_test,y_test = load_data(DATASET)

pp = permutation(x_train.shape[0])[:8000]
XX = x_train[pp]+randn(len(pp),1,28,28)*0.05
YY = y_train[pp]

XX = transpose(XX,[0,2,3,1])
x_test = transpose(x_test,[0,2,3,1])
input_shape = XX.shape

layers1 = [InputLayer(input_shape)]
layers1.append(DenseLayer(layers1[-1],K=32*nonlinearity,R=2,nonlinearity=None,sparsity_prior=0.,sigma=sigmass,learn_pi=1,p_drop=0.,bn=BN(0,0),U=0))
layers1.append(DenseLayer(layers1[-1],K=16*nonlinearity,R=2,nonlinearity=None,sparsity_prior=0.,sigma=sigmass,learn_pi=1,p_drop=0.,bn=BN(0,0),U=0))
layers1.append(FinalLayer(layers1[-1],R=neuronsss,sparsity_prior=0.00,sigma=sigmass,bn=BN(0,0)))

model1 = model(layers1)

if(supss):
    model1.init_dataset(XX,YY)
else:
    model1.init_dataset(XX)


samplesclass0 = []
samplesclass1 = []
W             = []
LOSSE         = []
reconstruction= []
for i in [2,3,5]+[20]*10:
	LOSSE.append(train_layer_model(model1,rcoeff_schedule=schedule(0.000000000001,'linear'),CPT=i,random=0,fineloss=0))
	reconstruction.append(model1.reconstruct()[:150])
	samplesclass0.append([model1.sampleclass(0,k)[:150] for k in xrange(neuronsss)])
	samplesclass1.append([model1.sampleclass(1,k)[:150] for k in xrange(neuronsss)])
	W.append(model1.get_Ws())

f=open(SAVE_DIR+'exp_neurons_'+sigmass+'_'+str(nonlinearity)+'.pkl','wb')
cPickle.dump([LOSSE,reconstruction,XX[:150],samplesclass0,samplesclass1,W],f)
f.close()




