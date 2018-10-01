from pylab import *
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
import tensorflow as tf


import sys
sys.path.insert(0, '../utils')

from layers import *
from utils import *

import cPickle
import os
SAVE_DIR = os.environ['SAVE_DIR']



def doit(pred,y,K):
    label = asarray([argmax(bincount(y[pred==k],None,10)) for k in xrange(K)])
    yhat  = label[pred]
    print mean((yhat==y).astype('float32'))

DATASET = 'MNIST'

sigmass  = 'global'
neurons = int(sys.argv[-1])

if(sys.argv[-2]=='None'):
        leakiness=None
else:   
        leakiness = float(sys.argv[-2])

supss     = 0


x_train,y_train,x_test,y_test = load_data(DATASET)

pp = permutation(x_train.shape[0])[:8000]
XX = x_train[pp]+randn(len(pp),1,28,28)*0.01
YY = y_train[pp]

XX = transpose(XX,[0,2,3,1])
x_test = transpose(x_test,[0,2,3,1])
input_shape = XX.shape

layers1 = [InputLayer(input_shape)]
layers1.append(DenseLayer(layers1[-1],K=32*neurons,R=2,leakiness=leakiness,sparsity_prior=0.,sigma=sigmass))
layers1.append(DenseLayer(layers1[-1],K=16*neurons,R=2,leakiness=leakiness,sparsity_prior=0.,sigma=sigmass))
layers1.append(FinalLayer(layers1[-1],R=10*neurons,sparsity_prior=0.,sigma=sigmass))

model1 = model(layers1)

if(supss):
    model1.init_dataset(XX,YY)
else:
    model1.init_dataset(XX)


LOSSES  = train_layer_model(model1,rcoeff_schedule=schedule(0.00000000001,'sqrt'),CPT=100,random=0,fineloss=0,verbose=0)
y_hat   = argmax(model1.predict(),1)
CL      = doit(y_hat,YY,10*neurons)

reconstruction = model1.reconstruct()[:150]
samplesclass0=[model1.sampleclass(0,k)[:150] for k in xrange(10*neurons)]
samplesclass1=[model1.sampleclass(1,k)[:150] for k in xrange(10*neurons)]
samples1=model1.sample(1)[:300]

f=open(SAVE_DIR+'exp_unsup_'+sys.argv[-2]+'_'+str(neurons)+'.pkl','wb')
cPickle.dump([LOSSES,samplesclass0,samplesclass1,samples1,CL],f)
f.close()




