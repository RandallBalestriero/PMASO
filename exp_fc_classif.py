from pylab import *
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
import tensorflow as tf

from layers import *
from utils import *

import cPickle

DATASET = 'MNIST'

def trainit(dx,dy,bs):
    nb = len(dx)/bs
    for i in xrange(nb):
        session.run(updates,feed_dict={x:dx[i*bs:(i+1)*bs],y:dy[i*bs:(i+1)*bs]})

def predict(dx,bs):
    nb = len(dx)/bs
    yhat = []
    for i in xrange(nb):
        yhat.append(argmax(session.run(z2,feed_dict={x:dx[i*bs:(i+1)*bs]}),axis=1))
    return concatenate(yhat)


rnnb=int(sys.argv[-1])
mode=int(sys.argv[-4])
nonlinearity=sys.argv[-3]
if(nonlinearity=='None'):
    nonlinearity=None
neurons=int(sys.argv[-2])
N = int(sys.argv[-5])
seed(rnnb)
x_train,y_train,x_test,y_test = load_data(DATASET)
x_train = transpose(x_train,[0,2,3,1])
x_test  = transpose(x_test,[0,2,3,1])
pp = permutation(x_train.shape[0])
x_train=x_train[pp]
y_train=y_train[pp]
XX = x_train[:N]
YY = y_train[:N]
input_shape = XX.shape

if(mode==1):
    layers1 = [InputLayer(input_shape)]
    layers1.append(DenseLayer(layers1[-1],K=neurons,R=2,nonlinearity=nonlinearity,sparsity_prior=0.1))
    layers1.append(FinalLayer(layers1[-1],10))
    model1   = model(layers1,local_sigma=0)
    model1.init_dataset(XX,YY)
    model1.init_thetaq()
    model1.init_dataset(XX,YY)
    LOSSES   = train_model(model1,rcoeff=50,CPT=50,random=1,fineloss=0)
    accuracy = [] 
    for i in xrange(1,len(x_train)/len(XX)):
        model1.init_dataset(x_train[len(XX)*i:(i+1)*len(XX)])
        model1.init_thetaq()
        for j in xrange(50):
            model1.E_step(random=1,fineloss=0)
            yhat=model1.predict()
            print mean((argmax(yhat,1)==y_train[len(XX)*i:(i+1)*len(XX)]).astype('float32'))
        yhat=model1.predict()
        accuracy.append(mean((argmax(yhat,1)==y_train[len(XX)*i:(i+1)*len(XX)]).astype('float32')))
    for i in xrange(len(x_test)/len(XX)):
        model1.init_dataset(x_test[len(XX)*i:(i+1)*len(XX)])
        model1.init_thetaq()
        for j in xrange(50):
            model1.E_step(random=1,fineloss=0)
        yhat=model1.predict()
        accuracy.append(mean((argmax(yhat,1)==y_test[len(XX)*i:(i+1)*len(XX)]).astype('float32')))
    accuracy=mean(accuracy)
else:
    lr=0.0001
    batch_size=50
    x=tf.placeholder(shape=[batch_size,28,28,1],dtype=tf.float32)
    y=tf.placeholder(shape=[batch_size],dtype=tf.int32)
#    h1 = tf.layers.dense(tf.layers.flatten(x),neurons)
#    mu=tf.Variable(tf.random_normal([28*28,neurons]))
#    distances = tf.reduce_mean(tf.square(tf.expand_dims(tf.layers.flatten(x),-1)-tf.expand_dims(mu,0)),axis=1)
#    k = tf.Variable(tf.random_normal([neurons,neurons]))
#    z1 = h1+tf.tensordot(distances*tf.log(distances+0.001),k,[[1],[0]])
    h1 = tf.layers.dense(tf.layers.flatten(x),neurons)
    if(nonlinearity is None):
        z1 = tf.maximum(h1,tf.layers.dense(tf.layers.flatten(x),neurons))
    elif(nonlinearity=='relu'):
        z1 = tf.nn.relu(h1)
    else:
        z1 = tf.abs(h1)
    z2=tf.layers.dense(z1,10)
    loss = tf.losses.softmax_cross_entropy(tf.one_hot(y,10),z2)
    learner     = tf.train.AdamOptimizer(lr)
    updates     = learner.minimize(loss)
    init        = tf.initialize_all_variables()
    n_epochs    = 50
    session_config = tf.ConfigProto(allow_soft_placement=False,log_device_placement=True)
    session_config.gpu_options.allow_growth=True
    session        = tf.Session(config=session_config)
    session.run(init)
    for n in xrange(n_epochs):
        trainit(XX,YY,batch_size)
    accuracy = mean(equal(concatenate([predict(x_train[len(XX):],batch_size),predict(x_test,batch_size)],axis=0),concatenate([y_train[len(XX):],y_test],axis=0)).astype('float32'))


print accuracy

f=open('EXP_CLASSIF/exp_fc_classif_'+str(rnnb)+'_'+str(nonlinearity)+'_'+str(neurons)+'_'+str(rnnb),'wb')
cPickle.dump(accuracy,f)
f.close()

