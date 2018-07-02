from pylab import *
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
import tensorflow as tf

from layers import *
from utils import *

import cPickle

DATASET = 'MNIST'

neurons = int(sys.argv[-1])
unsup   = int(sys.argv[-2])

x_train,y_train,x_test,y_test = load_data(DATASET)


pp = permutation(x_train.shape[0])[:5000]
XX = x_train[pp]+randn(len(pp),1,28,28)*0.1
YY = y_train[pp]

XX = transpose(XX,[0,2,3,1])
input_shape = XX.shape

layers1 = [InputLayer(input_shape)]
layers1.append(DenseLayer(layers1[-1],K=neurons,R=2,nonlinearity=None,sparsity_prior=0.00))
layers1.append(DenseLayer(layers1[-1],K=neurons,R=2,nonlinearity=None,sparsity_prior=0.00))
layers1.append(FinalLayer(layers1[-1],10,sparsity_prior=0.00))

model1 = model(layers1,local_sigma=0)

if(unsup):
    model1.init_dataset(XX)
else:
    model1.init_dataset(XX,YY)


#L=model1.train_dn(100,YY)
#print L
reconstruction = []
samplesclass1  = []
samples1       = []
for cpt in [5,20,80,400,495]:
	LOSSES  = train_model(model1,rcoeff=5,CPT=cpt,random=1,fineloss=0)
	reconstruction.append(model1.reconstruct()[:150])
	samplesclass1.append([model1.sampleclass(0,k)[:300] for k in xrange(10)])
	samples1.append(model1.sample(1)[:300])

W = model1.session.run(model1.layers[1].W)
b = model1.session.run(model1.layers[1].b)

f=open('BASE_EXP/exp_fc_'+str(unsup)+'_'+str(neurons)+'.pkl','wb')
cPickle.dump([LOSSES,reconstruction,XX[:150],samplesclass1,samples1,W,b],f)
f.close()




