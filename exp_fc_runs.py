from pylab import *
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
import tensorflow as tf

from layers import *
from utils import *

import cPickle

DATASET = 'MNIST'

number = int(sys.argv[-1])
random = int(sys.argv[-2])
K = int(sys.argv[-3])


x_train,y_train,x_test,y_test = load_data(DATASET)


pp = permutation(x_train.shape[0])[:10000]
XX = x_train[pp]+randn(len(pp),1,28,28)*0.0
YY = y_train[pp]

XX = transpose(XX,[0,2,3,1])
input_shape = XX.shape

ALL = []

layers1 = [InputLayer(input_shape)]
layers1.append(DenseLayer(layers1[-1],K=K,R=2,nonlinearity=None))
layers1.append(FinalLayer(layers1[-1],10))

model1 = model(layers1,local_sigma=0)
model1.init_dataset(XX,YY)
model1.init_model(random=1)

LOSSES         = train_model(model1,rcoeff=50,CPT=25,random=random)
ALL.append(LOSSES)

f=open('BASE_EXP/exp_fc_run'+str(random)+'_'+str(K)+'_'+str(number)+'.pkl','wb')
cPickle.dump(ALL,f)
f.close()




