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

DATASET = 'MNIST'

neuronsss=10
p = float(sys.argv[-3])
sigmass=sys.argv[-2]
#sparsity  = float32(sys.argv[-1])
nonlinearity = int(sys.argv[-1])
print nonlinearity
if(nonlinearity=='none'):
	nonlinearity=None

supss     = 1


x_train,y_train,x_test,y_test = load_data(DATASET)

pp = permutation(x_train.shape[0])[:7000]
XX = x_train[pp]
YY = y_train[pp]
XX = transpose(XX,[0,2,3,1])


mask = concatenate([ones((6000,28,28,1)),binomial(1,p,1000*28*28).reshape((1000,28,28,1))],axis=0)
XX = XX*mask + randn(7000,28,28,1)*0.01

input_shape = XX.shape

layers1 = [InputLayer(input_shape,1-mask)]
layers1.append(DenseLayer(layers1[-1],K=32*nonlinearity,R=2,nonlinearity=None,sparsity_prior=0.,sigma=sigmass,learn_pi=1,p_drop=0.,bn=BN(0,0),U=0))
layers1.append(DenseLayer(layers1[-1],K=16*nonlinearity,R=2,nonlinearity=None,sparsity_prior=0.,sigma=sigmass,learn_pi=1,p_drop=0.,bn=BN(0,0),U=0))
layers1.append(FinalLayer(layers1[-1],R=neuronsss,sparsity_prior=0.00,sigma=sigmass,bn=BN(0,0)))

model1 = model(layers1)

if(supss):
    model1.init_dataset(XX,YY)
else:
    model1.init_dataset(XX)

model1.set_output_mask(concatenate([zeros(6000),ones(1000)]).astype('float32'))

samplesclass0 = []
samplesclass1 = []
W             = []
LOSSE         = []
reconstruction= []
reconstruction2 = []
for i in [15]*12:
	LOSSE.append(train_layer_model(model1,rcoeff_schedule=schedule(0.000000000001,'linear'),CPT=i,random=0,fineloss=0))
        reconstruction2.append(model1.get_input()[-1000:])


f=open(SAVE_DIR+'exp_oclusion2_'+str(p)+'_'+sigmass+'_'+str(nonlinearity)+'.pkl','wb')
cPickle.dump([LOSSE,reconstruction2,XX[-1000:]],f)
f.close()




