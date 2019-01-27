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
    

DATASET = 'CIFAR'

NEURONS  = 10

N = 64*30
BS =64


x_train,y_train,x_test,y_test,Y_mask = load_data(DATASET)


#print x_train[0]


#x_train/=sqrt(sum(x_train**2,(1,2,3),keepdims=True))
pp = permutation(N)
XX = x_train[pp]#*1+randn(len(pp),1,28,28)*0.001
YY = y_train[pp]

#XX = transpose(XX,[0,2,3,1])
#x_test = transpose(x_test,[0,2,3,1])
#input_shape = [BS,28,28,1]

input_shape = [BS,32,32,3]


modelnb = int(sys.argv[-3])

if(int(sys.argv[-2])):
    EM = True
else:
    EM = False

sigma   = sys.argv[-1]

with tf.device('/device:GPU:0'): 
    MODEL1 = [InputLayer(input_shape)]
    if(modelnb==0):
        MODEL1.append(ConvLayer(MODEL1[-1],K=12,Ic=5,Jc=5,R=1,leakiness=0.0,sparsity_prior=0.000,sigma='global',update_b=False))
        MODEL1.append(PoolLayer(MODEL1[-1],Ic=2,Jc=2,Dc=1,sigma='channel'))
    elif(modelnb==1):
        MODEL1.append(ConvLayer(MODEL1[-1],K=12,Ic=5,Jc=5,R=2,leakiness=0.0,sparsity_prior=0.000,sigma='global',update_b=False))
        MODEL1.append(PoolLayer(MODEL1[-1],Ic=2,Jc=2,Dc=1,sigma='channel'))
        MODEL1.append(ConvLayer(MODEL1[-1],K=32,Ic=3,Jc=3,R=2,leakiness=0.0,sigma='channel',update_b=False))
        MODEL1.append(PoolLayer(MODEL1[-1],Ic=2,Jc=2,sigma='channel'))
    elif(modelnb==2):
        MODEL1.append(ConvLayer(MODEL1[-1],K=12,Ic=5,Jc=5,R=2,leakiness=0.0,sparsity_prior=0.000,sigma='global',update_b=False))
        MODEL1.append(PoolLayer(MODEL1[-1],Ic=2,Jc=2,Dc=1,sigma='channel'))
        MODEL1.append(ConvLayer(MODEL1[-1],K=32,Ic=3,Jc=3,R=2,leakiness=0.0,sigma='channel',update_b=False))
        MODEL1.append(PoolLayer(MODEL1[-1],Ic=2,Jc=2,sigma='channel'))
        MODEL1.append(ConvLayer(MODEL1[-1],K=64,Ic=1,Jc=1,R=2,leakiness=0.0,sigma='channel',update_b=False))
        MODEL1.append(PoolLayer(MODEL1[-1],Ic=1,Jc=1,Dc=2,sigma='channel'))
    MODEL1.append(CategoricalLastLayer(MODEL1[-1],R=NEURONS,sparsity_prior=.0001,sigma='local',update_b=False))



model1 = model(MODEL1,XX,Y_mask=Y_mask,sigma=sigma)


y_hat   = argmax(model1.layers_[model1.layers[-1]].p,1)
CL      = doit(y_hat,YY,NEURONS)
print CL
ACCU = [CL]

#LOSSES  = pretrain(model1,0)

for i in xrange(620):
    if(EM):
        LOSSES  = train_layer_model(model1,rcoeff_schedule=schedule(.01005,'linear'),alpha_schedule=schedule(0.5,'mean'),CPT=1,random=0,fineloss=0,verbose=0,per_layer=1,mp_opt=0,partial_E=0,PLOT=0)
    else:
        LOSSES  = train_layer_model(model1,rcoeff_schedule=schedule(.01005,'linear'),alpha_schedule=schedule(0.5,'exp'),CPT=1,random=0,fineloss=0,verbose=0,per_layer=1,mp_opt=0,partial_E=1,PLOT=0)
    y_hat1   = argmax(model1.layers_[model1.layers[-1]].p,1)
    CL      = doit(y_hat1,YY,NEURONS)
    print CL
    ACCU.append(CL)
    f=open(SAVE_DIR+'exp_clustering_'+str(modelnb)+'_'+str(EM)+'_'+str(sigma)+'.pkl','wb')
    cPickle.dump([ACCU],f)
    f.close()

reconstruction = model1.reconstruct()[:3000]
samplesclass0=[model1.sampleclass(0,k)[:150] for k in xrange(NEURONS)]
samplesclass1=[model1.sampleclass(1,k)[:150] for k in xrange(NEURONS)]
#samples1=model1.sample(1)[:3000]

f=open(SAVE_DIR+'exp_clustering_'+str(modelnb)+'_'+str(EM)+'_'+str(sigma)+'.pkl','wb')
cPickle.dump([LOSSES,reconstruction,XX[:3000],YY[:3000],samplesclass0,samplesclass1,ACCU],f)
f.close()




