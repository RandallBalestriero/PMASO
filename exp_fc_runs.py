from pylab import *
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
import tensorflow as tf

from layers import *
from utils import *

import cPickle

DATASET = 'MNIST'

run     = int(sys.argv[-1])
doU     = int(sys.argv[-2])
neurons = int(sys.argv[-3])
supervised   = int(sys.argv[-4])

x_train,y_train,x_test,y_test = load_data(DATASET)


XX,x_test,YY,y_test = train_test_split(x_train,y_train,train_size=10000,stratify=y_train)
XX=XX.astype('float32')
YY=YY.astype('int32')
XX = transpose(XX+0.05*randn(*shape(XX)),[0,2,3,1])
input_shape = XX.shape

layers1 = [InputLayer(input_shape)]
layers1.append(DenseLayer(layers1[-1],K=32*neurons,R=2,nonlinearity=None,sparsity_prior=0.00,batch_norm=BatchNorm(scale=0,center=0),sigma='local'))
layers1.append(DenseLayer(layers1[-1],K=32,R=2,nonlinearity=None,sparsity_prior=0.00,batch_norm=BatchNorm(scale=0,center=0),sigma='local'))
layers1.append(FinalLayer(layers1[-1],10,sparsity_prior=0.00,batch_norm=BatchNorm(scale=0,center=0),sigma='local'))


model1 = model(layers1)


if(supervised==0):
    model1.init_dataset(XX)
else:
    model1.init_dataset(XX,YY)

LOSSES  = train_layer_model(model1,rcoeff=0.002,CPT=600,random=0,fineloss=0,dob=1,doU=doU)

f=open('/mnt/project2/rb42Data/PMASO/BASE_EXP/exp_fc_run'+str(supervised)+'_'+str(neurons)+'_'+str(number)+'_'+str(doU)+'.pkl','wb')
cPickle.dump(LOSSES,f)
f.close()




