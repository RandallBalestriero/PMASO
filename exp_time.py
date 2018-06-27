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


pp = permutation(x_train.shape[0])[:10000]
XX = x_train[pp]+randn(len(pp),1,28,28)*0.0
YY = y_train[pp]

XX = transpose(XX,[0,2,3,1])
input_shape = XX.shape

layers1 = [InputLayer(input_shape)]
layers1.append(DenseLayer(layers1[-1],K=neurons,R=2,nonlinearity=None))
layers1.append(FinalLayer(layers1[-1],10))

model1 = model(layers1,local_sigma=0)

if(unsup):
    model1.init_dataset(XX)
else:
    model1.init_dataset(XX,YY)

LOSSES,timing  = train_model(model1,rcoeff=50,CPT=50,random=1,return_time=1,fineloss=1)

f=open('timing_'+str(neurons)+'.pkl','wb')
cPickle.dump(timing,f)
f.close()




