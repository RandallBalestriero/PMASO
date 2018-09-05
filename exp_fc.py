from pylab import *
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
import tensorflow as tf

from layers import *
from utils import *

import cPickle

DATASET = 'MNIST'


sparsity  = float32(sys.argv[-1])
neuronsss = int(sys.argv[-2])
supss     = 1
sigmass='local'

x_train,y_train,x_test,y_test = load_data(DATASET)

pp = permutation(x_train.shape[0])[:10000]
XX = x_train[pp]+randn(len(pp),1,28,28)*0.000005
YY = y_train[pp]

XX = transpose(XX,[0,2,3,1])
x_test = transpose(x_test,[0,2,3,1])
input_shape = XX.shape

layers1 = [InputLayer(input_shape)]
layers1.append(DenseLayer(layers1[-1],K=32*neuronsss,R=2,nonlinearity=None,sparsity_prior=0.00,sigma=sigmass,learn_pi=1,p_drop=0.1))
layers1.append(DenseLayer(layers1[-1],K=32,R=2,nonlinearity=None,sparsity_prior=0.00,sigma=sigmass,learn_pi=1,p_drop=0.1))
layers1.append(FinalLayer(layers1[-1],10,sparsity_prior=0.00,sigma=sigmass,))

model1 = model(layers1)

if(supss):
    model1.init_dataset(XX,YY)
else:
    model1.init_dataset(XX)


reconstruction = []
samplesclass0  = []
samplesclass1  = []
samples1       = []
sigmas         = []
for cpt in [5,20,75,200,300]:
	LOSSES  = train_layer_model(model1,rcoeff=0.01,CPT=cpt,random=1,fineloss=0)
	reconstruction.append(model1.reconstruct()[:150])
	samplesclass0.append([model1.sampleclass(0,k)[:150] for k in xrange(10)])
        samplesclass1.append([model1.sampleclass(1,k)[:150] for k in xrange(10)])
	samples1.append(model1.sample(1)[:300])
	sigmas.append(model1.get_sigmas())


if(supss):
#    model1.init_dataset(x_test)
#    model1.init_thetaq()
#    model1.E_step(rcoeff=0.002,random=0,fineloss=0)
#    yhat = model1.predict()
#    accuracy = mean(asarray(argmax(yhat,1)==y_test).astype('float32'))
#    print "ACCURACY",accuracy
    accuracy = 0
else:
    accuracy = 0

W = model1.get_Ws()
#b = model1.get_bs()

f=open('/mnt/project2/rb42Data/PMASO/BASE_EXP/exp_fc_'+str(supss)+'_'+str(neuronsss)+'_'+sigmass+'.pkl','wb')
cPickle.dump([LOSSES,reconstruction,XX[:150],samplesclass0,samplesclass1,samples1,W,sigmas,accuracy],f)
f.close()




