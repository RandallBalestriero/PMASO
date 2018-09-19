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


DATASET = 'CIFAR'

neuronsss=1
sigmass='local'
#sparsity  = float32(sys.argv[-1])
classs       = int(sys.argv[-1])

supss   = 1

XX,YY,x_test,y_test = load_data(DATASET,classs)

XX = XX[:1500]
YY = YY[:1500]*0

#figure(figsize=(25,5))
#for i in xrange(15):
#	subplot(1,15,i+1)
#	imshow(normalize(XX[i]))
#show()
#sleep()

#pp = permutation(x_train.shape[0])
#XX = x_train[pp]#+randn(len(pp),32,32,3)*0.05
#YY = y_train[pp]
#print YY

input_shape = XX.shape

layers1 = [InputLayer(input_shape)]
layers1.append(ConvPoolLayer(layers1[-1],Ic=5,Jc=5,Ir=2,Jr=2,K=4,R=2,nonlinearity=None,sparsity_prior=0.0,sigma='channel'))
layers1.append(DenseLayer(layers1[-1],K=96,R=2,nonlinearity=None,sparsity_prior=0.,sigma=sigmass,learn_pi=1,p_drop=0.,bn=BN(0,0),U=0))
layers1.append(FinalLayer(layers1[-1],R=1,sparsity_prior=0.00,sigma=sigmass,bn=BN(0,0)))

model1 = model(layers1)

if(supss):
    model1.init_dataset(XX,YY)
else:
    model1.init_dataset(XX)

samplesclass0 = []
samplesclass1 = []
W             = []
LOSSE         = []
reconstruction= []

for i in [2,3,5]+[20]*6:
	LOSSE.append(train_layer_model(model1,rcoeff_schedule=schedule(0.000000000001,'linear'),CPT=i,random=0,fineloss=0,verbose=0))
	reconstruction.append(model1.reconstruct()[:150])
	samplesclass0.append([model1.sampleclass(0,k)[:150] for k in xrange(neuronsss)])
	samplesclass1.append([model1.sampleclass(1,k)[:150] for k in xrange(neuronsss)])
	W.append(model1.get_Ws())

f=open(SAVE_DIR+'exp_cnn_cifar_'+sigmass+'_'+str(nonlinearity)+'_'+str(classs)+'.pkl','wb')
cPickle.dump([LOSSE,reconstruction,XX[:150],samplesclass0,samplesclass1,W],f)
f.close()



