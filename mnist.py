from pylab import *
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
import tensorflow as tf

from layers import *
from utils import *

DATASET = 'MNIST'

x_train,y_train,x_test,y_test = load_data(DATASET)


pp = permutation(x_train.shape[0])[:100]
XX = x_train[pp]+randn(len(pp),1,28,28)*0.01
YY = y_train[pp]

XX = transpose(XX,[0,2,3,1])
input_shape = XX.shape

layers1 = [InputLayer(input_shape)]
layers1.append(DenseLayer(layers1[-1],K=32,R=2,nonlinearity=None))
layers1.append(DenseLayer(layers1[-1],K=32,R=2,nonlinearity=None))
layers1.append(SupFinalLayer(layers1[-1],10))

layers2 = [InputLayer(input_shape)]
layers2.append(ConvLayer(layers2[-1],stride=1,Ic=5,Jc=5,K=16,R=2,nonlinearity=None))
layers2.append(PoolLayer(layers2[-1],2))
layers2.append(ConvLayer(layers2[-1],stride=1,Ic=5,Jc=5,K=32,R=2,nonlinearity=None))
layers2.append(PoolLayer(layers2[-1],2))
layers2.append(SupFinalLayer(layers2[-1],10))

layers3 = [InputLayer(input_shape)]
layers3.append(ConvLayer(layers3[-1],stride=1,Ic=3,Jc=3,K=8,R=2,nonlinearity=None,sparsity_prior=0.001))
#layers3.append(PoolLayer(layers3[-1],2))
layers3.append(DenseLayer(layers3[-1],K=16,R=2,nonlinearity=None,sparsity_prior=0.001))
layers3.append(SupFinalLayer(layers3[-1],10))



model1 = model(layers3,local_sigma=0)
model1.init_dataset(XX,YY)
model1.init_model(random=1)
#model1.init_thetaq()

LIKELIHOOD,KL = train_model(model1,rcoeff=90,CPT=60)

figure()
plot(linspace(0,1,len(LIKELIHOOD)),LIKELIHOOD)
plot(linspace(0,1,len(KL)),KL)
U=model1.reconstruct()
figure()
for i in xrange(25):
    subplot(5,5,1+i)
    imshow(U[i,:,:,0],aspect='auto')
    xticks([])
    yticks([])
#    colorbar()

U=model1.sample(1)
figure()
for i in xrange(64):
    subplot(8,8,1+i)
    imshow(U[i,:,:,0],aspect='auto')
    xticks([])
    yticks([])
#    colorbar()

tight_layout()
show()






