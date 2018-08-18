from pylab import *
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
import tensorflow as tf

from layers import *
from utils import *

import cPickle

DATASET = 'MNIST'

doU     = int(sys.argv[-1])
#doBN    = int(sys.argv[-2])
neurons = int(sys.argv[-2])
sup     = int(sys.argv[-3])

x_train,y_train,x_test,y_test = load_data(DATASET)

pp = permutation(x_train.shape[0])[:10000]
XX = x_train[pp]+randn(len(pp),1,28,28)*0.05
YY = y_train[pp]

XX = transpose(XX,[0,2,3,1])
x_test = transpose(x_test,[0,2,3,1])
input_shape = XX.shape

layers1 = [InputLayer(input_shape)]
layers1.append(DenseLayer(layers1[-1],K=32*neurons,R=2,nonlinearity=None,sparsity_prior=0.00,batch_norm=BatchNorm(scale=0,center=0),sigma='local'))
layers1.append(DenseLayer(layers1[-1],K=32,R=2,nonlinearity=None,sparsity_prior=0.00,batch_norm=BatchNorm(scale=0,center=0),sigma='local'))
layers1.append(FinalLayer(layers1[-1],10,sparsity_prior=0.00,batch_norm=BatchNorm(scale=0,center=0),sigma='local'))

model1 = model(layers1)

if(sup):
    model1.init_dataset(XX,YY)
else:
    model1.init_dataset(XX)


#L=model1.train_dn(100,YY)
#print L
reconstruction = []
samplesclass0  = []
samplesclass1  = []
samples1       = []
sigmas         = []
for cpt in [5,20,75,200,300]:
	LOSSES  = train_layer_model(model1,rcoeff=0.002,CPT=cpt,random=0,fineloss=0,dob=1,doU=doU)
	reconstruction.append(model1.reconstruct()[:150])
	samplesclass0.append([model1.sampleclass(0,k)[:150] for k in xrange(10)])
        samplesclass1.append([model1.sampleclass(1,k)[:150] for k in xrange(10)])
	samples1.append(model1.sample(1)[:300])
	sigmas.append(model1.get_sigmas())


if(sup):
    model1.init_dataset(x_test)
    model1.init_thetaq()
    model1.E_step(rcoeff=0.002,random=0,fineloss=0)
    yhat = model1.predict()
    print "ACCURACY",mean(asarray(argmax(yhat,1)==y_test).astype('float32'))

W = model1.get_Ws()
b = model1.get_bs()

f=open('/mnt/project2/rb42Data/PMASO/BASE_EXP/exp_fc_'+str(sup)+'_'+str(neurons)+'_'+str(doU)+'.pkl','wb')
cPickle.dump([LOSSES,reconstruction,XX[:150],samplesclass0,samplesclass1,samples1,W,b,sigmas],f)
f.close()




