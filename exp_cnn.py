from pylab import *
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
import tensorflow as tf

from layers import *
from utils import *

import cPickle

DATASET = 'MNIST'

modeln     = int(sys.argv[-1])
supervised = int(sys.argv[-2])

x_train,y_train,x_test,y_test = load_data(DATASET)


pp = permutation(x_train.shape[0])[:2000]
XX = x_train[pp]+randn(len(pp),1,28,28)*0.05
YY = y_train[pp]

XX = transpose(XX,[0,2,3,1])
input_shape = XX.shape

if(modeln==0):
	layers1 = [InputLayer(input_shape)]
	layers1.append(ConvLayer(layers1[-1],stride=1,Ic=5,Jc=5,K=8,R=2,nonlinearity=None,sparsity_prior=0.))
	layers1.append(PoolLayer(layers1[-1],4))
	layers1.append(FinalLayer(layers1[-1],10))
elif(modeln==1):
        layers1 = [InputLayer(input_shape)]
        layers1.append(ConvPoolLayer(layers1[-1],Ic=5,Jc=5,Ir=2,Jr=2,K=4,R=2,nonlinearity=None,sparsity_prior=0.0,sigma='local'))
#        layers1.append(ConvPoolLayer(layers1[-1],Ic=3,Jc=3,Ir=2,Jr=2,K=8,R=2,nonlinearity=None,sparsity_prior=0.00,sigma='channel'))
        layers1.append(DenseLayer(layers1[-1],K=32,R=2,nonlinearity=None,sparsity_prior=0.00,sigma='local'))
        layers1.append(FinalLayer(layers1[-1],10,sparsity_prior=0.00,sigma='local'))
else:
	layers1 = [InputLayer(input_shape)]
	layers1.append(ConvPoolLayer(layers1[-1],Ic=5,Jc=5,Ir=3,Jr=3,K=16,R=2,nonlinearity=None,sparsity_prior=0.0))
        layers1.append(ConvPoolLayer(layers1[-1],Ic=3,Jc=3,Ir=2,Jr=2,K=32,R=2,nonlinearity=None,sparsity_prior=0.0))
	layers1.append(DenseLayer(layers1[-1],K=128,R=2,nonlinearity=None,sparsity_prior=0.))
	layers1.append(FinalLayer(layers1[-1],10,sparsity_prior=0.0))



model1 = model(layers1)
if(supervised):
	model1.init_dataset(XX,YY)
else:
        model1.init_dataset(XX)

#init_model(model1,CPT=10)
LOSSES  = train_layer_model(model1,rcoeff=0.01,CPT=150,random=0,fineloss=0)
reconstruction = model1.reconstruct()[:150]
samplesclass1  = [model1.sampleclass(int(k<4),k)[:300] for k in xrange(10)]
samples1       = model1.sample(1)

W = model1.session.run(model1.layers[1].W)

f=open('BASE_EXP/exp_cnn_sup'+str(supervised)+'_'+str(modeln)+'.pkl','wb')
cPickle.dump([LOSSES,reconstruction,XX[:150],samplesclass1,samples1,W],f)
f.close()




