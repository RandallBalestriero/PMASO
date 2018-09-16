from pylab import *
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
import tensorflow as tf

from layers import *
from utils import *

import cPickle

DATASET = 'MNIST'

neuronsss=10
sigmass=sys.argv[-2]
nonlinearity = int(sys.argv[-1])
print nonlinearity


x_train,y_train,x_test,y_test = load_data(DATASET)

x_train = transpose(x_train,[0,2,3,1])
x_test  = transpose(x_test,[0,2,3,1])

pp = permutation(x_train.shape[0])

x_train = x_train[pp]+randn(len(pp),28,28,1)*0.05
y_train = y_train[pp]

XX = x_train[:2000]
YY = y_train[:2000]

input_shape = XX.shape

layers1 = [InputLayer(input_shape)]
layers1.append(DenseLayer(layers1[-1],K=32*nonlinearity,R=2,nonlinearity=None,sparsity_prior=0.,sigma=sigmass,learn_pi=1,p_drop=0.,bn=BN(0,0),U=0))
layers1.append(DenseLayer(layers1[-1],K=16*nonlinearity,R=2,nonlinearity=None,sparsity_prior=0.,sigma=sigmass,learn_pi=1,p_drop=0.,bn=BN(0,0),U=0))
layers1.append(FinalLayer(layers1[-1],R=neuronsss,sparsity_prior=0.00,sigma=sigmass,bn=BN(0,0)))

model1 = model(layers1)

model1.init_dataset(XX,YY)


LOSSE=train_layer_model(model1,rcoeff_schedule=schedule(0.000000000001,'linear'),CPT=132,random=0,fineloss=0)
for i in xrange(1,10):
    model1.init_dataset(x_train[2000*i:2000*(i+1)])
    model1.init_thetaq()
    model1.E_step(10)
    y_hat = argmax(model1.predict(),1)
    print mean((y_hat==y_train[2000*i:2000*(i+1)]).astype('float32'))
    model1.E_step(1)
    y_hat = argmax(model1.predict(),1)
    print mean((y_hat==y_train[2000*i:2000*(i+1)]).astype('float32'))
    model1.E_step(0.001)
    y_hat = argmax(model1.predict(),1)
    print mean((y_hat==y_train[2000*i:2000*(i+1)]).astype('float32'))


