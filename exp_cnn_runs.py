from pylab import *
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
import tensorflow as tf

from layers import *
from utils import *

import cPickle

DATASET = 'MNIST'

number = int(sys.argv[-1])
random = int(sys.argv[-2])
neurons= int(sys.argv[-3])
fsize=int(sys.argv[-4])
supervised = int(sys.argv[-5])


x_train,y_train,x_test,y_test = load_data(DATASET)


XX,x_test,YY,y_test = train_test_split(x_train,y_train,train_size=2000,stratify=y_train)
XX=XX.astype('float32')
YY=YY.astype('int32')
XX = transpose(XX,[0,2,3,1])
input_shape = XX.shape

ALL = []

layers1 = [InputLayer(input_shape)]
layers1.append(ConvLayer(layers1[-1],stride=1,Ic=fsize,Jc=fsize,K=neurons,R=2,nonlinearity=None,sparsity_prior=0.))
layers1.append(PoolLayer(layers1[-1],2))
layers1.append(FinalLayer(layers1[-1],10))

model1 = model(layers1,local_sigma=0)
if(supervised==1):
    model1.init_dataset(XX,YY)
else:
    model1.init_dataset(XX)


LOSSES         = train_model(model1,rcoeff=50,CPT=40,random=random)
ALL.append(LOSSES)

#f=open('BASE_EXP/exp_fc_run'+str(supervised)+'_'+str(init)+'_'+str(random)+'_'+str(K)+'_'+str(number)+'.pkl','wb')
f=open('/mnt/project2/rb42Data/PMASO/BASE_EXP/exp_cnn_run'+str(supervised)+'_'+str(train_dn)+'_'+str(init_thetaq_random)+'_'+str(random)+'_'+str(neurons)+'_'+str(number)+'.pkl','wb')
cPickle.dump(ALL,f)
f.close()




