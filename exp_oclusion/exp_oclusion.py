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


print sys.argv

DATASET       = sys.argv[-6]
CLASS         = int(sys.argv[-5])
MODEL_TYPE    = sys.argv[-4]
OCLUSION_TYPE = sys.argv[-3]
OCLUSION_SPEC = float(sys.argv[-2])
KNOWN_Y       = int(sys.argv[-1])


sigmass='local'


supss     = 1



x_train,y_train,x_test,y_test = load_data(DATASET,CLASS)


if(DATASET=='MNIST'): 
	CCC = 10
	SIZE = 28
	CHANNEL = 1
	if(MODEL_TYPE=='CNN'):
                N1=3000
                N2=300
	else:
		N1=6000
		N2=600
	K1 = 96
	EPS=0.002
        pp = permutation(x_train.shape[0])[:N1+N2]
        XX = x_train[pp]/10
        YY = y_train[pp]
        XX = transpose(XX,[0,2,3,1])
	n_filters =4
elif(DATASET=='CIFAR'): 
	K1 = 64
	CCC = 1
	SIZE = 32
	CHANNEL = 3
	N1=1300
	N2=130
	EPS=0.0
        pp = permutation(x_train.shape[0])[:N1+N2]
        XX = x_train[pp]/30
        YY = y_train[pp]
#        XX = transpose(XX,[0,2,3,1])
	n_filters=6



if(OCLUSION_TYPE=='pixel'):
	mask = concatenate([ones((N1,SIZE,SIZE,CHANNEL)),binomial(1,OCLUSION_SPEC,N2*SIZE*SIZE*CHANNEL).reshape((N2,SIZE,SIZE,CHANNEL))],axis=0)
#	xx = XX*mask + randn(N1+N2,SIZE,SIZE,CHANNEL)*EPS
elif(OCLUSION_TYPE=='box'):
	mask = ones((N1+N2,SIZE,SIZE,CHANNEL))
	for i in xrange(N1,N1+N2):
		ii = randint(0,SIZE-OCLUSION_SPEC,2)
		mask[i,ii[0]:ii[0]+int(OCLUSION_SPEC),ii[1]:ii[1]+int(OCLUSION_SPEC),:]=0


xx = XX*mask + randn(N1+N2,SIZE,SIZE,CHANNEL)*EPS


input_shape = XX.shape

if(MODEL_TYPE=='MLP'):
	layers1 = [InputLayer(input_shape,1-mask)]
	layers1.append(DenseLayer(layers1[-1],K=K1,R=2,leakiness=None,sigma=sigmass))
	layers1.append(DenseLayer(layers1[-1],K=64,R=2,leakiness=None,sigma=sigmass))
	layers1.append(FinalLayer(layers1[-1],R=CCC,sigma=sigmass))
elif(MODEL_TYPE=='CNN'):
        layers1 = [InputLayer(input_shape,1-mask)]
	layers1.append(ConvPoolLayer(layers1[-1],Ic=5,Jc=5,Ir=2,Jr=2,K=n_filters,R=2,leakiness=None,sigma='channel'))
        layers1.append(DenseLayer(layers1[-1],K=64,R=2,leakiness=None,sigma=sigmass))
        layers1.append(FinalLayer(layers1[-1],R=CCC,sigma=sigmass))



model1 = model(layers1)
model1.init_dataset(xx,YY)
if(KNOWN_Y==0):
    model1.set_output_mask(concatenate([zeros(N1),ones(N2)]).astype('float32'))


samplesclass0   = []
samplesclass1   = []
W               = []
LOSSE           = []
reconstruction  = []
reconstruction2 = []
preds           = []

reconstruction2.append(model1.get_input()[-N2:])
preds.append(model1.predict()[-N2:])

for i in [20]*10:
	LOSSE.append(train_layer_model(model1,rcoeff_schedule=schedule(0.00,'linear'),CPT=i,random=0,per_layer=1,mp_opt=0,fineloss=0))
        reconstruction2.append(model1.get_input()[-N2:])
	preds.append(model1.predict()[-N2:])
	f=open(SAVE_DIR+'exp_oclusion_'+DATASET+'_'+str(CLASS)+'_'+MODEL_TYPE+'_'+OCLUSION_TYPE+'_'+str(OCLUSION_SPEC)+'_'+str(KNOWN_Y)+'.pkl','wb')
	cPickle.dump([LOSSE,reconstruction2,xx[-N2:],XX[-N2:],YY[-N2:],preds],f)
	f.close()




