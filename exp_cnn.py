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

def normalize(x):
        return (x-x.min())/(x.max()-x.min())


DATASET = sys.argv[-3]

if(DATASET=='CIFAR'): neuronsss=1
else: neuronsss=10

sigmass=sys.argv[-1]

classs       = int(sys.argv[-2])

supss   = 1

XX,YY,x_test,y_test = load_data(DATASET,classs)

if(DATASET=='CIFAR'):
    XX = XX[:1500]
    YY = YY[:1500]*0
else:
    XX = XX[:2500]+randn(2500,XX.shape[1],XX.shape[2],XX.shape[3])*0.01
    XX = transpose(XX,[0,2,3,1])
    YY = YY[:2500]


input_shape = XX.shape

print input_shape
layers1 = [InputLayer(input_shape)]
layers1.append(ConvPoolLayer(layers1[-1],Ic=5,Jc=5,Ir=2,Jr=2,K=4,R=2,nonlinearity=None,sparsity_prior=0.0,sigma=sigmass))
layers1.append(DenseLayer(layers1[-1],K=96,R=2,nonlinearity=None,sparsity_prior=0.,sigma='local',learn_pi=1,p_drop=0.,bn=BN(0,0),U=0))
layers1.append(FinalLayer(layers1[-1],R=1,sparsity_prior=0.00,sigma='local',bn=BN(0,0)))

model1 = model(layers1)

if(supss):
    model1.init_dataset(XX,YY)
else:
    model1.init_dataset(XX)


LOSSE=train_layer_model(model1,rcoeff_schedule=schedule(0.000000000001,'linear'),CPT=140,random=0,fineloss=0,verbose=0)
reconstruction = model1.reconstruct()[:150]
samplesclass0 = [model1.sampleclass(0,k)[:150] for k in xrange(neuronsss)]
samplesclass1 = [model1.sampleclass(1,k)[:150] for k in xrange(neuronsss)]
W=model1.get_Ws()

f=open(SAVE_DIR+'exp_cnn_'+DATASET.lower()+'_'+sigmass+'_'+str(classs)+'.pkl','wb')
cPickle.dump([LOSSE,reconstruction,XX[:150],samplesclass0,samplesclass1,W],f)
f.close()




