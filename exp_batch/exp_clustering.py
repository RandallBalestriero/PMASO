from pylab import *
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
import tensorflow as tf
import time


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
    return mean((yhat==y).astype('float32'))
    

DATASET = 'MNIST'

sigmass  = 'global'

MODEL    = 1
NEURONS  = 10

N = 50000
BS = 5000

leakiness = float32(0)

supss     = 0


x_train,y_train,x_test,y_test = load_data(DATASET)

pp = permutation(N)
XX = x_train[pp]+randn(len(pp),1,28,28)*0.00002
YY = y_train[pp]

XX = transpose(XX,[0,2,3,1])
x_test = transpose(x_test,[0,2,3,1])
input_shape = [BS,28,28,1]


if(MODEL==0):
    layers1 = [InputLayer(input_shape)]
    layers1.append(FinalLayer(layers1[-1],R=NEURONS,sigma=sigmass))
elif(MODEL==1):
    layers1 = [InputLayer(input_shape)]
#    layers1.append(DenseLayer(layers1[-1],K=32,R=2,leakiness=leakiness,sparsity_prior=0.,sigma=sigmass,batch=True,update_b=False))
    layers1.append(ConvLayer(layers1[-1],K=8,Ic=3,Jc=3,R=2,leakiness=0,sparsity_prior=0.,sigma=sigmass,batch=True))
    layers1.append(FinalLayer(layers1[-1],R=NEURONS,sigma=sigmass,batch=True,update_b=False))
elif(MODEL==2):
    layers1 = [InputLayer(input_shape)]
    layers1.append(DenseLayer(layers1[-1],K=64,R=2,leakiness=leakiness,sparsity_prior=0.,sigma=sigmass))
    layers1.append(DenseLayer(layers1[-1],K=64,R=2,leakiness=leakiness,sparsity_prior=0.,sigma=sigmass,residual=1))
    layers1.append(FinalLayer(layers1[-1],R=NEURONS,sigma=sigmass))




model1 = model(layers1,XX,batch=True)

#if(supss):
#    model1.init_dataset(XX,YY)
#else:
#    model1.init_dataset(XX)

y_hat   = argmax(model1.Y,1)
print shape(y_hat),y_hat
CL      = doit(y_hat,YY,NEURONS)
print CL
time.sleep(1)
LOSSES  = train_layer_model(model1,rcoeff_schedule=schedule(0.0000000000000001,'linear'),alpha_schedule=schedule(0.95,'mean'),CPT=170,random=0,fineloss=0,verbose=1,per_layer=1,mp_opt=0,partial_E=0)
y_hat1   = argmax(model1.Y,1)
CL      = doit(y_hat1,YY,NEURONS)
print CL
print y_hat
print y_hat1


reconstruction = model1.reconstruct()[:3000]
samplesclass0=[model1.sampleclass(0,k)[:150] for k in xrange(NEURONS)]
samplesclass1=[model1.sampleclass(1,k)[:150] for k in xrange(NEURONS)]
#samples1=model1.sample(1)[:3000]

f=open(SAVE_DIR+'exp_batch_'+str(N)+'_'+str(BS)+'_'+str(MODEL)+'_'+str(NEURONS)+'.pkl','wb')
cPickle.dump([LOSSES,reconstruction,XX[:3000],YY[:3000],samplesclass0,samplesclass1],f)
f.close()




