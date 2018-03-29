from pylab import *
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
import tensorflow as tf

from layers import *
from utils import *



DATASET = 'MNIST'

if(DATASET=='MNIST'):
        batch_size = 50
        mnist         = fetch_mldata('MNIST original')
        x             = mnist.data.reshape(70000,1,28,28).astype('float32')
        y             = mnist.target.astype('int32')
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=10000,stratify=y)
        input_shape   = (batch_size,28,28,1)
#	x_train = transpose(x_train,[0,2,3,1])
#	x_test  = transpose(x_test,[0,2,3,1])
	c = 10
        n_epochs = 150

elif(DATASET == 'CIFAR'):
        batch_size = 50
        TRAIN,TEST = load_cifar(3)
        x_train,y_train = TRAIN
        x_test,y_test     = TEST
        input_shape       = (batch_size,32,32,3)
        x_train = transpose(x_train,[0,2,3,1])
        x_test  = transpose(x_test,[0,2,3,1])
	c=10
        n_epochs = 150

elif(DATASET == 'CIFAR100'):
	batch_size = 100
        TRAIN,TEST = load_cifar100(3)
        x_train,y_train = TRAIN
        x_test,y_test     = TEST
        input_shape       = (batch_size,32,32,3)
        x_train = transpose(x_train,[0,2,3,1])
        x_test  = transpose(x_test,[0,2,3,1])
        c=100
        n_epochs = 200

elif(DATASET=='IMAGE'):
	batch_size=200
        x,y           = load_imagenet()
	x = x.astype('float32')
	y = y.astype('int32')
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=20000,stratify=y)
        input_shape   = (batch_size,64,64,3)
	c=200
        n_epochs = 200

else:
        batch_size = 50
        TRAIN,TEST = load_svhn()
        x_train,y_train = TRAIN
        x_test,y_test     = TEST
        input_shape       = (batch_size,32,32,3)
        x_train = transpose(x_train,[0,2,3,1])
        x_test  = transpose(x_test,[0,2,3,1])
	c=10
        n_epochs = 150




x_train          -= x_train.mean((1,2,3),keepdims=True)
x_train          /= abs(x_train).max((1,2,3),keepdims=True)
x_test           -= x_test.mean((1,2,3),keepdims=True)
x_test           /= abs(x_test).max((1,2,3),keepdims=True)
x_train           = x_train.astype('float32')
x_test            = x_test.astype('float32')
y_train           = array(y_train).astype('int32')
y_test            = array(y_test).astype('int32')
 






XX = x_train[:1000]

input_shape = XX.shape

layers = [InputLayer(input_shape)]
layers.append(ConvLayer(layers[-1],K=15,I=3,J=3,R=2,stride=2))
#layers.append(DenseLayer(layers[-1],K=15,R=2))
layers.append(UnsupFinalLayer(layers[-1],10))

init_v2(layers)

x       = tf.placeholder(tf.float32,shape=layers[0].input_shape)

opti = tf.train.RMSPropOptimizer(0.100810005)
train_op1 = opti.minimize(-KL(layers),var_list=tf.get_collection('latent'))
print opti.variables()
train_op2 = opti.minimize(-likelihood(layers),var_list=tf.get_collection('params'))
print opti.variables()


session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)





session.run(init_latent(x,layers),feed_dict={x:XX})



U=session.run(sample(layers))
print shape(U)
figure()
for i in xrange(25):
    subplot(5,5,1+i)
    imshow(U[i,0],aspect='auto')
    colorbar()


for k in xrange(6):
	print tf.get_collection('latent')
	for i in xrange(16):
		session.run(train_op1)
		print "KL",session.run(KL(layers))
	for i in xrange(16):
	        session.run(train_op2)
	        print session.run(likelihood(layers))


U=session.run(sample(layers))
figure()
for i in xrange(25):
    subplot(5,5,1+i)
    imshow(U[i,0],aspect='auto')
    colorbar()

show()






