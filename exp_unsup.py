from pylab import *
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
import tensorflow as tf

from layers import *
from utils import *

import cPickle


def doit(pred,y,K):
    label = asarray([argmax(bincount(y[pred==k],None,10)) for k in xrange(K)])
    yhat  = label[pred]
    print mean((yhat==y).astype('float32'))

DATASET = 'MNIST'

sigmass=sys.argv[-1]
supss     = 0


x_train,y_train,x_test,y_test = load_data(DATASET)

pp = permutation(x_train.shape[0])[:1000]
XX = x_train[pp]+randn(len(pp),1,28,28)*0.05
YY = y_train[pp]

XX = transpose(XX,[0,2,3,1])
x_test = transpose(x_test,[0,2,3,1])
input_shape = XX.shape

layers1 = [InputLayer(input_shape)]
#layers1.append(DenseLayer(layers1[-1],K=16,R=2,nonlinearity=None,sparsity_prior=0.,sigma=sigmass,learn_pi=1,p_drop=0.,bn=BN(0,0),U=0))
layers1.append(ConvPoolLayer(layers1[-1],Ic=3,Jc=3,Ir=2,Jr=2,K=6,R=2,nonlinearity=None,sparsity_prior=0.0,sigma='local'))
#layers1.append(DenseLayer(layers1[-1],K=64,R=2,nonlinearity=None,sparsity_prior=0.,sigma=sigmass,learn_pi=1,p_drop=0.2,bn=BN(0,0),U=0))
layers1.append(FinalLayer(layers1[-1],R=10,sparsity_prior=0.,sigma=sigmass,bn=BN(0,0)))

model1 = model(layers1)

if(supss):
    model1.init_dataset(XX,YY)
else:
    model1.init_dataset(XX)


y_hat = argmax(model1.predict(),1)

doit(y_hat,YY,10)

for k in xrange(140):
    LOSSES  = train_layer_model(model1,rcoeff_schedule=schedule(0.0000000001,'sqrt'),CPT=1,random=1,fineloss=0,verbose=1)
    #model1.init_dataset(x_train[2000*i:2000*(i+1)])
    #model1.init_thetaq()
    #model1.E_step(10)
    y_hat = argmax(model1.predict(),1)
#    for i in xrange(10):
    print bincount(y_hat,None,10)
    doit(y_hat,YY,10)

reconstruction = model1.reconstruct()[:150]
for k in xrange(5):
    subplot(1,5,k+1)
    imshow(reconstruction[k,:,:,0])
show()
#f=open('/mnt/project2/rb42Data/PMASO/BASE_EXP/exp_unsup_'+str(neuronsss)+'.pkl','wb')
#cPickle.dump([LOSSES,reconstruction,XX[:2000],samplesclass0,samplesclass1,samples1,W,sigmas,pred],f)
#f.close()




