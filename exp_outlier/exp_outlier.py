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





DATASET= sys.argv[-3]
CLASS  = int(sys.argv[-2])

x_train,y_train,x_test,y_test = load_data(DATASET,CLASS)


if(DATASET=='MNIST'):
        CCC = 10
        K1 = 96
        EPS=0.002
        pp = permutation(x_train.shape[0])[:8000]
        XX = x_train[pp]/10
        YY = y_train[pp]
        XX = transpose(XX,[0,2,3,1])
        n_filters =4
        NORM = 10
elif(DATASET=='CIFAR'):
        K1 = 64
        CCC = 1
        EPS = 0
        pp = permutation(x_train.shape[0])[:1500]
        XX = x_train[pp]/30
        YY = y_train[pp]
        n_filters=6
        NORM = 30


input_shape = XX.shape
sigmass  = 'local'

MODEL    = int(sys.argv[-1])
NEURONS  = 32

leakiness=None

supss     = 0


if(MODEL==0):
    layers1 = [InputLayer(input_shape)]
    layers1.append(DenseLayer(layers1[-1],K=NEURONS,R=2,leakiness=leakiness,sparsity_prior=0.,sigma=sigmass))
    layers1.append(ContinuousLastLayer(layers1[-1],sigma=sigmass))
elif(MODEL==1):
    layers1 = [InputLayer(input_shape)]
    layers1.append(DenseLayer(layers1[-1],K=64,R=2,leakiness=leakiness,sparsity_prior=0.,sigma=sigmass))
    layers1.append(DenseLayer(layers1[-1],K=NEURONS,R=2,leakiness=leakiness,sparsity_prior=0.,sigma=sigmass))
    layers1.append(ContinuousLastLayer(layers1[-1],sigma=sigmass))
elif(MODEL==2):
    layers1 = [InputLayer(input_shape)]
    layers1.append(DenseLayer(layers1[-1],K=64,R=2,leakiness=leakiness,sparsity_prior=0.,sigma=sigmass))
    layers1.append(DenseLayer(layers1[-1],K=64,R=2,leakiness=leakiness,sparsity_prior=0.,sigma=sigmass))
    layers1.append(DenseLayer(layers1[-1],K=NEURONS,R=2,leakiness=leakiness,sparsity_prior=0.,sigma=sigmass))
    layers1.append(ContinuousLastLayer(layers1[-1],sigma=sigmass))




model1 = model(layers1)

if(supss):
    model1.init_dataset(XX,YY)
else:
    model1.init_dataset(XX)


LOSSES  = train_layer_model(model1,rcoeff_schedule=schedule(0.0000,'sqrt'),CPT=200,random=0,fineloss=0,verbose=0,per_layer=1,mp_opt=0)
#y_hat   = argmax(model1.predict(),1)
#CL      = doit(y_hat,YY,10*neurons)

#reconstruction = model1.reconstruct()[:3000]
#samplesclass0=[model1.sampleclass(0,k)[:150] for k in xrange(10*neurons)]
#samplesclass1=[model1.sampleclass(1,k)[:150] for k in xrange(10*neurons)]
#samples1=model1.sample(1)[:3000]
evidence = model1.get_evidence()

f=open(SAVE_DIR+'exp_outlier_'+DATASET+'_'+str(CLASS)+'_'+str(MODEL)+'.pkl','wb')
cPickle.dump([LOSSES,XX,evidence],f)
f.close()




