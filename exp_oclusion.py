from pylab import *
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
import tensorflow as tf

from layers import *
from utils import *

import cPickle

DATASET = 'MNIST'


def do_oclusion(x):
    mask=ones_like(x)
    for i in xrange(len(x)):
        m = randint(0,17)
        n = randint(0,17)
        mask[i,m:m+10,n:n+10]=0
    return x*mask


x_train,y_train,x_test,y_test = load_data(DATASET)


pp = permutation(x_train.shape[0])[:10000]
XX = x_train[pp]+randn(len(pp),1,28,28)*0.0
YY = y_train[pp]

XX = transpose(XX,[0,2,3,1])
input_shape = XX.shape

layers1 = [InputLayer(input_shape)]
layers1.append(DenseLayer(layers1[-1],K=32,R=2,nonlinearity=None))
#layers1.append(DenseLayer(layers1[-1],K=32,R=2,nonlinearity=None))
layers1.append(FinalLayer(layers1[-1],10))

model1 = model(layers1,local_sigma=0)
model1.init_dataset(XX,YY)
model1.init_model(random=1)
#model1.init_thetaq()

LOSSES  = train_model(model1,rcoeff=50,CPT=25)

XXmasked = do_oclusion(XX)

model1.init_dataset(XXmasked,YY)
model1.init_thetaq()

for i in xrange(10):
    model1.E_step(random=1)



reconstruction = model1.reconstruct()[:300]
#samplesclass1  = [model1.sampleclass(1,k) for k in xrange(10)]
#samples1       = model1.sample(1)




f=open('OCLUSION_EXP/exp_fc_oclusion.pkl','wb')
cPickle.dump([reconstruction,XX[:300],XXmasked[:300]],f)
f.close()




