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


runnb=int(sys.argv[-4])
DATASET = sys.argv[-3]
sigmass=sys.argv[-2]
if(sys.argv[-1]=='None'):
	leakiness=None
else:
	leakiness = float(sys.argv[-1])

supss     = 1


x_train,y_train,x_test,y_test = load_data(DATASET)

pp = permutation(x_train.shape[0])[:8000]
XX = x_train[pp]/10+randn(len(pp),1,28,28)*0.002
YY = y_train[pp]

XX = transpose(XX,[0,2,3,1])
x_test = transpose(x_test,[0,2,3,1])
input_shape = XX.shape

layers1 = [InputLayer(input_shape)]
layers1.append(DenseLayer(layers1[-1],K=64,R=2,leakiness=leakiness,sigma=sigmass,sparsity_prior=0.0))
layers1.append(DenseLayer(layers1[-1],K=32,R=2,leakiness=leakiness,sigma=sigmass))
layers1.append(FinalLayer(layers1[-1],R=10,sigma=sigmass,sparsity_prior=0.))

model1 = model(layers1)

if(supss):
    model1.init_dataset(XX,YY)
else:
    model1.init_dataset(XX)


LOSSES  = train_layer_model(model1,rcoeff_schedule=schedule(0.0000,'linear'),CPT=200,random=0,fineloss=0,verbose=0,mp_opt=0,per_layer=1)
reconstruction=model1.reconstruct()[:1500]
samplesclass0=[model1.sampleclass(0,k)[:150] for k in xrange(10)]
samplesclass1=[model1.sampleclass(1,k)[:150] for k in xrange(10)]
samples1=model1.sample(1)[:300]
params = model1.get_params()

f=open(SAVE_DIR+'exp_nonlinearity_'+DATASET+'_'+sigmass+'_'+sys.argv[-1]+'_run'+str(runnb)+'.pkl','wb')
cPickle.dump([LOSSES,reconstruction,XX[:1500],samplesclass0,samplesclass1,samples1,params],f)
f.close()




