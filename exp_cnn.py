from pylab import *
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
import tensorflow as tf

from layers import *
from utils import *

import cPickle

DATASET = 'MNIST'

supervised=int(sys.argv[-3])
fsize     = int(sys.argv[-1])
modeln    = int(sys.argv[-2])

x_train,y_train,x_test,y_test = load_data(DATASET)


pp = permutation(x_train.shape[0])[:2000]
XX = x_train[pp]+randn(len(pp),1,28,28)*0.0
YY = y_train[pp]

XX = transpose(XX,[0,2,3,1])
input_shape = XX.shape

if(modeln==0):
	layers1 = [InputLayer(input_shape)]
	layers1.append(ConvLayer(layers1[-1],stride=1,Ic=fsize,Jc=fsize,K=8,R=2,nonlinearity=None,sparsity_prior=0.))
	layers1.append(PoolLayer(layers1[-1],2))
	layers1.append(FinalLayer(layers1[-1],10))
else:
	layers1 = [InputLayer(input_shape)]
	layers1.append(ConvLayer(layers1[-1],stride=1,Ic=fsize,Jc=fsize,K=8,R=2,nonlinearity=None,sparsity_prior=0.))
	layers1.append(PoolLayer(layers1[-1],2))
	layers1.append(DenseLayer(layers1[-1],K=32,R=2,nonlinearity=None,sparsity_prior=0.00))
	layers1.append(FinalLayer(layers1[-1],10))



model1 = model(layers1,local_sigma=0)
if(supervised):
	model1.init_dataset(XX,YY)
else:
        model1.init_dataset(XX)

LOSSES  = train_model(model1,rcoeff=100,CPT=80,random=1,fineloss=0)
reconstruction = model1.reconstruct()[:150]
samplesclass1  = [model1.sampleclass(1,k) for k in xrange(10)]
samples1       = model1.sample(1)

W = model1.session.run(model1.layers[1].W)

f=open('BASE_EXP/exp_cnn_sup'+str(supervised)+'_'+str(modeln)+'_'+str(fsize)+'.pkl','wb')
cPickle.dump([LOSSES,reconstruction,XX[:150],samplesclass1,samples1,W],f)
f.close()




