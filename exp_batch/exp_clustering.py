from pylab import *
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
import tensorflow as tf
import time
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


import sys

sys.path.insert(0, '../utils')

from layers import *
from utils import *

import cPickle
import os

SAVE_DIR = os.environ['SAVE_DIR']

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size





def doit(pred,y,K):
    label = asarray([argmax(bincount(y[pred==k],None,10)) for k in xrange(K)])
    yhat  = label[pred]
    return mean((yhat==y).astype('float32'))
    

DATASET = 'MNIST'

sigmass  = 'global'

MODEL    = 1
NEURONS  = 10

N = 64*6
BS =64

leakiness = None#float32(0.001)

supss     = 0

x_train,y_train,x_test,y_test,Y_mask = load_data(DATASET)


print x_train[0]


#x_train/=sqrt(sum(x_train**2,(1,2,3),keepdims=True))
pp = permutation(N)
XX = x_train[pp]#+randn(len(pp),1,28,28)*0.01
YY = y_train[pp]

XX = transpose(XX,[0,2,3,1])
x_test = transpose(x_test,[0,2,3,1])
input_shape = [BS,28,28,1]

#input_shape = [BS,32,32,3]
#




with tf.device('/device:GPU:0'): 
    MODEL1 = [InputLayer(input_shape)]
    MODEL1.append(ConvLayer(MODEL1[-1],K=6,Ic=5,Jc=5,R=2,sparsity_prior=0.000,sigma='channel',update_b='channel'))
    MODEL1.append(PoolLayer(MODEL1[-1],Ic=2,Jc=2,sigma='channel'))
#    MODEL1.append(ConvLayer(MODEL1[-1],K=32,Ic=3,Jc=3,R=2,leakiness=0,sparsity_prior=0.,sigma='channel',update_b='channel'))
#    MODEL1.append(PoolLayer(MODEL1[-1],Ic=2,Jc=2,sigma='channel'))
#    MODEL1.append(DenseLayer(MODEL1[-1],K=64,R=2,leakiness=leakiness,sparsity_prior=.00,sigma='global',update_b=True))
#    MODEL1.append(DenseLayer(MODEL1[-1],K=128,R=1,leakiness=leakiness,sparsity_prior=.001,sigma='local',update_b=True))
#    MODEL1.append(DenseLayer(MODEL1[-1],K=128,R=2,leakiness=leakiness,sparsity_prior=.00,sigma='local',update_b=False))
    MODEL1.append(CategoricalLastLayer(MODEL1[-1],R=NEURONS,sparsity_prior=.00,sigma='local',update_b=False))
#    MODEL1.append(ContinuousLastLayer(MODEL1[-1],128,'global',sparsity_prior=0.0))
#        layers1.append(DenseLayer(layers1[-1],K=128,R=2,leakiness=leakiness,sparsity_prior=0.,sigma=sigmass,update_b=True))
#        layers1.append(ConvLayer(layers1[-1],K=3,Ic=3,Jc=3,R=2,leakiness=0,sparsity_prior=0.,sigma='global'))
#    layers1.append(PoolLayer(layers1[-1],Ic=2,Jc=2,sigma='global'))
#    layers1.append(DenseLayer(layers1[-1],K=64,R=2,leakiness=leakiness,sparsity_prior=0.,sigma=sigmass,update_b=True,G=False))
#    layers1.append(DenseLayer(layers1[-1],K=64,R=2,leakiness=leakiness,sparsity_prior=0.001,sigma=sigmass,update_b=True,G=False))
#    layers1.append(DenseLayer(layers1[-1],K=32,R=2,leakiness=leakiness,sparsity_prior=0.00,sigma=sigmass,update_b=True,G=False))



model1 = model(MODEL1,XX,Y_mask=Y_mask)

#model2 = model(MODEL2,XX)



y_hat   = argmax(model1.layers_[model1.layers[-1]].p,1)
print shape(y_hat),y_hat
CL      = doit(y_hat,YY,NEURONS)
print CL
#time.sleep(1)
ACCU = [CL]

#LOSSES  = pretrain(model1,0)

for i in xrange(620):
    LOSSES  = train_layer_model(model1,rcoeff_schedule=schedule(.0001005,'linear'),alpha_schedule=schedule(0.85,'mean'),CPT=1,random=1,fineloss=0,verbose=0,per_layer=1,mp_opt=0,partial_E=0,PLOT=0)
    y_hat1   = argmax(model1.layers_[model1.layers[-1]].p,1)
    CL      = acc(YY,y_hat1)#doit(y_hat1,YY,NEURONS)
    print CL
    ACCU.append(CL)


reconstruction = model1.reconstruct()[:3000]
samplesclass0=[model1.sampleclass(0,k)[:150] for k in xrange(NEURONS)]
samplesclass1=[model1.sampleclass(1,k)[:150] for k in xrange(NEURONS)]
#samples1=model1.sample(1)[:3000]

f=open(SAVE_DIR+'exp_clustering2_'+str(N)+'_'+str(BS)+'_'+str(MODEL)+'_'+str(NEURONS)+'.pkl','wb')
cPickle.dump([LOSSES,reconstruction,XX[:3000],YY[:3000],samplesclass0,samplesclass1,ACCU],f)
f.close()




