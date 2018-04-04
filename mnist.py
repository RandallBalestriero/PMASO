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
x_train          /= abs(x_train).max((1,2,3),keepdims=True)#/10
x_test           -= x_test.mean((1,2,3),keepdims=True)
x_test           /= abs(x_test).max((1,2,3),keepdims=True)
x_train           = x_train.astype('float32')
x_test            = x_test.astype('float32')
y_train           = array(y_train).astype('int32')
y_test            = array(y_test).astype('int32')
 







#XX = load_digits()['data']#x_train[permutation(x_train.shape[0])[:1000]]+randn(1000,1,28,28)*0.01
XX = x_train[permutation(x_train.shape[0])[:1500]]+randn(1500,1,28,28)*0.1
#XX-=XX.mean(1,keepdims=True)
#XX/=XX.max(1,keepdims=True)
XX = transpose(XX,[0,2,3,1])
#XX=XX.reshape((XX.shape[0],1,8,8))
input_shape = XX.shape

layers = [InputLayer(input_shape)]
#layers.append(DenseLayer(layers[-1],K=32,R=2))
layers.append(ConvLayer(layers[-1],stride=4,Ic=4,Jc=4,K=8,R=2))
layers.append(UnsupFinalLayer(layers[-1],10))

#init_v2(layers)

x       = tf.placeholder(tf.float32,shape=layers[0].input_shape)

#opti = tf.train.AdamOptimizer(10.20800810005)
#train_op1 = opti.minimize(-KL(layers)/200,var_list=tf.get_collection('latent'))
#print opti.variables()
#train_op2 = opti.minimize(-likelihood(layers),var_list=tf.get_collection('params'))

session_config = tf.ConfigProto(allow_soft_placement=False,
                                          log_device_placement=True)


session = tf.Session(config=session_config)
init = tf.global_variables_initializer()
session.run(init)





session.run(init_latent(x,layers),feed_dict={x:XX})



U=session.run(sample(layers))
#print shape(U)
#figure()
#for i in xrange(25):
#    subplot(5,5,1+i)
#    imshow(U[i,0],aspect='auto')
#    colorbar()


updates_v2 = update_v2(layers)
#updates_m = update_m(layers)
updates_mk = update_mk(layers)
updates_sigma = update_sigma(layers)
updates_p = update_p(layers)
updates_pk= update_pk(layers)
updates_W = update_W(layers)
updates_Wk = update_Wk(layers)
updates_pi = update_pi(layers)

#updates_p = update_p(layers)

PI1=[]
PI2 =[]
PI1.append(session.run(layers[1].pi))
PI2.append(session.run(layers[2].pi))


P1 = []
P2 = []

KLl=KL(layers) 
LIKl=likelihood(layers)

for k in xrange(0):
#	print tf.get_collection('latent')
        print "KL",session.run(KLl)
#        session.run(updates_sigma)
        session.run(updates_v2)
#        print session.run(layers[1].m)
        print "AFTER V",session.run(KLl)
	for i in xrange(3):
                session.run(updates_m)
                print "AFTER M",session.run(KLl)
                session.run(updates_p)
                print "AFTER P",session.run(KLl)
        print "LIKELI",session.run(LIKl)
#        session.run(updates_pi)
        for i in xrange(3):
#                print "INIT LIKE",session.run(likelihood(layers))
#                for k in xrange(2):
                session.run(updates_W)
                print "AFTER W",session.run(likelihood(layers))
#                imshow(reshape(session.run(layers[1].W)[0,0],(28,28)))
#                show()
#                for k in xrange(2):
                session.run(updates_pi)
                print "AFTER P",session.run(likelihood(layers))
#                session.run(updates_sigma)
                print "AFTER S",session.run(likelihood(layers))
#                print session.run([layers[1].sigmas2,layers[2].sigmas2])
#                PI1.append(session.run(layers[1].pi))
#                PI2.append(session.run(layers[2].pi))
#	        session.run(train_op2)
#                print session.run(likelihood(layers))




samples = sample(layers)


for k in xrange(10):
        print "KL",session.run(KLl)
        session.run(updates_v2)
        print "AFTER V",session.run(KLl)
        session.run(updates_sigma)
        print "AFTER S",session.run(KLl)
        for i in xrange(2):
                for l in xrange(1):
                        for k in permutation(layers[l+1].K).astype('int32'):
                                session.run(updates_mk[l][k])
                                print "AFTER M",l,k,session.run(KLl)
                for l in xrange(1):
                        for k in permutation(layers[l+1].K).astype('int32'):
                                session.run(updates_pk[l][k])
                                print "AFTER P",l,k,session.run(KLl)
                session.run(updates_p[-1])
                print "AFTER P last",session.run(KLl)
                session.run(updates_v2)
                print "AFTER V",l,session.run(KLl)
        print "LIKELI",session.run(LIKl)
        session.run(updates_pi)
        print "AFTER pi",session.run(LIKl)
        session.run(updates_sigma)
        print "AFTER S",session.run(LIKl),session.run([layers[1].sigmas2,layers[2].sigmas2])
        for i in xrange(2):
                for l in xrange(1):
                        for k in permutation(layers[l+1].K).astype('int32'):
                                session.run(updates_Wk[l][k])
                                print "AFTER W",l,k,session.run(LIKl)
                session.run(updates_W[-1])
                print "AFTER W",session.run(LIKl)
#                session.run(updates_sigma)
#                print "AFTER S",session.run(likelihood(layers))
#                print session.run([layers[1].sigmas2,layers[2].sigmas2])
#                PI1.append(session.run(layers[1].pi))
#                PI2.append(session.run(layers[2].pi))
#	        session.run(train_op2)
#                print session.run(likelihood(layers))



#PI1=asarray(PI1)
#PI2=asarray(PI2)

#figure()
#
#subplot(121)
#for i in xrange(32):
#    for j in xrange(2):
#        plot(PI1[:,i,j])
#subplot(122)
#for i in xrange(10):
#    plot(PI2[:,0,i])
#show()



#samplet = sampletrue(layers)
#samples = sample(layers)
#U=session.run(samplet)
#figure()
#for i in xrange(25):
#    subplot(5,5,1+i)
#    imshow(U[i,0],aspect='auto')
#    colorbar()

U=session.run(samples)
figure()
for i in xrange(36):
    subplot(6,6,1+i)
    imshow(U[i,:,:,0],aspect='auto')
    xticks([])
    yticks([])
    colorbar()

tight_layout()
show()






