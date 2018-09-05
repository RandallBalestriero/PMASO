from pylab import *
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
import tensorflow as tf

from layers import *
from utils import *

import cPickle

DATASET = 'MNIST'

sigmass   = 'local'
supss     = 1
residual  = int(sys.argv[-1])
n_layers  = int(sys.argv[-2])

x_train,y_train,x_test,y_test = load_data(DATASET)

pp = permutation(x_train.shape[0])[:8000]
XX = x_train[pp]+randn(len(pp),1,28,28)*0.05
YY = y_train[pp]

XX = transpose(XX,[0,2,3,1])
x_test = transpose(x_test,[0,2,3,1])
input_shape = XX.shape

layers1 = [InputLayer(input_shape)]
layers1.append(DenseLayer(layers1[-1],K=32,R=2,nonlinearity=None,sparsity_prior=0.00,sigma='local',learn_pi=1,p_drop=0.))
for l in xrange(n_layers):
	layers1.append(DenseLayer(layers1[-1],K=32,R=2,nonlinearity=None,sparsity_prior=0.00,sigma=sigmass,learn_pi=1,p_drop=0.,residual=residual))
layers1.append(FinalLayer(layers1[-1],10,sparsity_prior=0.00,sigma='local'))

model1 = model(layers1)

if(supss):
    model1.init_dataset(XX,YY)
else:
    model1.init_dataset(XX)


LOSSES  = train_layer_model(model1,rcoeff=0.01,CPT=600,random=1,fineloss=0)
reconstruction=model1.reconstruct()[:150]
samplesclass0=[model1.sampleclass(0,k)[:150] for k in xrange(10)]
samplesclass1=[model1.sampleclass(1,k)[:150] for k in xrange(10)]
samples1=model1.sample(1)[:300]
sigmas=model1.get_sigmas()

W = model1.get_Ws()

f=open('/mnt/project2/rb42Data/PMASO/BASE_EXP/exp_resnet_'+str(residual)+'_'+str(n_layers)+'.pkl','wb')
cPickle.dump([LOSSES,reconstruction,XX[:150],samplesclass0,samplesclass1,samples1,W,sigmas],f)
f.close()




