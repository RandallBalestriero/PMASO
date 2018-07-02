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
neurons= int(sys.argv[-2])
supervised = int(sys.argv[-3])
random = int(sys.argv[-4])

x_train,y_train,x_test,y_test = load_data(DATASET)


XX,x_test,YY,y_test = train_test_split(x_train,y_train,train_size=10000,stratify=y_train)
XX=XX.astype('float32')
YY=YY.astype('int32')
XX = transpose(XX,[0,2,3,1])
input_shape = XX.shape


layers1 = [InputLayer(input_shape)]
layers1.append(DenseLayer(layers1[-1],K=neurons,R=2,nonlinearity=None,sparsity_prior=0.00))
layers1.append(DenseLayer(layers1[-1],K=neurons,R=2,nonlinearity=None,sparsity_prior=0.00))
layers1.append(FinalLayer(layers1[-1],10,sparsity_prior=0.00))

model1 = model(layers1,local_sigma=0)

if(supervised==0):
    model1.init_dataset(XX)
else:
    model1.init_dataset(XX,YY)

LOSSES         = train_model(model1,rcoeff=5,CPT=1000,random=random,fineloss=1,return_infos=1)

f=open('/mnt/project2/rb42Data/PMASO/BASE_EXP/exp_fc_run'+str(supervised)+'_'+str(random)+'_'+str(neurons)+'_'+str(number)+'.pkl','wb')
cPickle.dump(LOSSES,f)
f.close()




