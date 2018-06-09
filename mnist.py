from pylab import *
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
import tensorflow as tf

from layers import *
from utils import *

DATASET = 'MNIST'

x_train,y_train,x_test,y_test = load_data(DATASET)


pp = permutation(x_train.shape[0])[:1550]
XX = x_train[pp]+randn(len(pp),1,28,28)*0.05
YY = y_train[pp]

XX = transpose(XX,[0,2,3,1])
input_shape = XX.shape

layers = [InputLayer(input_shape)]
#layers.append(DenseLayer(layers[-1],K=64,R=2))
layers.append(ConvLayer(layers[-1],stride=1,Ic=3,Jc=3,K=8,R=2))
#layers.append(PoolLayer(layers[-1],2))
layers.append(UnsupFinalLayer(layers[-1],10))



model1 = model(layers)
model1.init_theta()
model1.init_dataset(XX,YY)
model1.init_thetaq()

LIKELIHOOD,KL = train_model(model1,1000,20)

figure()
plot(LIKELIHOOD)

U=model1.reconstruct()
figure()
for i in xrange(25):
    subplot(5,5,1+i)
    imshow(U[i,:,:,0],aspect='auto')
    xticks([])
    yticks([])
#    colorbar()

U=model1.sample(0)
figure()
for i in xrange(36):
    subplot(6,6,1+i)
    imshow(U[i,:,:,0],aspect='auto')
    xticks([])
    yticks([])
#    colorbar()

tight_layout()
show()






