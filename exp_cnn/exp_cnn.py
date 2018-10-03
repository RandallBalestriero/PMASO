from pylab import *
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
import tensorflow as tf


sys.path.insert(0, '../utils')

from layers import *
from utils import *


import cPickle
import os
SAVE_DIR = os.environ['SAVE_DIR']

def normalize(x):
        return (x-x.min())/(x.max()-x.min())


DATASET = sys.argv[-2]

if(DATASET=='CIFAR'): neuronsss=1
else: neuronsss=10

sigmass='local'

classs       = int(sys.argv[-1])

supss   = 1

XX,YY,x_test,y_test = load_data(DATASET,classs)

if(DATASET=='CIFAR'):
    XX = XX[:1500]/30
    YY = YY[:1500]*0
    n_filters = 6
    CCC = 1
else:
    XX = XX[:2500]/10+randn(2500,XX.shape[1],XX.shape[2],XX.shape[3])*0.002
    XX = transpose(XX,[0,2,3,1])
    YY = YY[:2500]
    n_filters = 4
    CCC = 10


input_shape = XX.shape

print input_shape
layers1 = [InputLayer(input_shape)]
layers1.append(ConvPoolLayer(layers1[-1],Ic=5,Jc=5,Ir=2,Jr=2,K=n_filters,R=2,sigma='channel'))
layers1.append(DenseLayer(layers1[-1],K=96,R=2,leakiness=None,sparsity_prior=0.,sigma='local'))
layers1.append(DenseLayer(layers1[-1],K=32,R=2,leakiness=None,sparsity_prior=0.,sigma='local'))
layers1.append(FinalLayer(layers1[-1],R=CCC,sigma='local'))

model1 = model(layers1)

if(supss):
    model1.init_dataset(XX,YY)
else:
    model1.init_dataset(XX)


LOSSES = []
for i in xrange(10):
    LOSSES.append(train_layer_model(model1,rcoeff_schedule=schedule(0.00000,'linear'),CPT=20,random=0,fineloss=0,verbose=0,per_layer=1,mp_opt=0))
    reconstruction = model1.reconstruct()[:150]
    samplesclass0 = [model1.sampleclass(0,k)[:150] for k in xrange(neuronsss)]
    samplesclass1 = [model1.sampleclass(1,k)[:150] for k in xrange(neuronsss)]
    params=model1.get_params()
    f=open(SAVE_DIR+'exp_cnn_'+DATASET.lower()+'_'+str(classs)+'.pkl','wb')
    cPickle.dump([LOSSES,reconstruction,XX[:150],samplesclass0,samplesclass1,params],f)
    f.close()




