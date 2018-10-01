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





DATASET = 'MNIST'


if(sys.argv[-1]=='None'):
        leakiness=None
else:
        leakiness = float(sys.argv[-1])

supss     = 1

mp_opt    = int(sys.argv[-2])
randomm   = int(sys.argv[-3])
per_layer = int(sys.argv[-4])

x_train,y_train,x_test,y_test = load_data(DATASET)

pp = permutation(x_train.shape[0])[:8000]
XX = x_train[pp]*0.1+randn(len(pp),1,28,28)*0.002
YY = y_train[pp]

XX = transpose(XX,[0,2,3,1])
x_test = transpose(x_test,[0,2,3,1])
input_shape = XX.shape

layers1 = [InputLayer(input_shape)]
layers1.append(DenseLayer(layers1[-1],K=64,R=2,leakiness=leakiness,sparsity_prior=0.00,sigma='local'))
layers1.append(DenseLayer(layers1[-1],K=32,R=2,leakiness=leakiness,sparsity_prior=0.00,sigma='local'))
layers1.append(FinalLayer(layers1[-1],10,sparsity_prior=0.00,sigma='local'))


model1 = model(layers1)

if(supss):
    model1.init_dataset(XX,YY)
else:
    model1.init_dataset(XX)


LOSSES  = train_layer_model(model1,rcoeff_schedule=schedule(0.0000000000,'linear'),CPT=200,random=randomm,fineloss=0,verbose=0,mp_opt=mp_opt,per_layer=per_layer)
reconstruction=model1.reconstruct()[:1500]
samplesclass0=[model1.sampleclass(0,k)[:150] for k in xrange(10)]
samplesclass1=[model1.sampleclass(1,k)[:150] for k in xrange(10)]
samples1=model1.sample(1)[:300]

params = model1.get_params()

f=open(SAVE_DIR+'exp_orderEM_'+str(per_layer)+'_'+str(randomm)+'_'+str(mp_opt)+'_'+sys.argv[-1]+'.pkl','wb')
cPickle.dump([LOSSES,reconstruction,XX[:1500],samplesclass0,samplesclass1,samples1,params],f)
f.close()




