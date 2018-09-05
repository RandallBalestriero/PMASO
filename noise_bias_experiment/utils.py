import tensorflow as tf
import cPickle
from pylab import *
import glob
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

###################################################################
#
#
#                       UTILITY FOR CIFAR10 & MNIST
#
#
###################################################################
def load_utility(DATASET):
	if(DATASET=='MNIST'):
	        batch_size = 50
	        mnist         = fetch_mldata('MNIST original')
	        x             = mnist.data.reshape(70000,1,28,28).astype('float32')
	        y             = mnist.target.astype('int32')
	        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=10000,stratify=y)
	        input_shape   = (batch_size,28,28,1)
		x_train = transpose(x_train,[0,2,3,1])
		x_test  = transpose(x_test,[0,2,3,1])
		c = 10
        	n_epochs = 100

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
	return x_train,x_test,y_train,y_test,c,n_epochs,input_shape 


def principal_components(x):
    x = x.transpose(0, 2, 3, 1)
    flatx = numpy.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
    sigma = numpy.dot(flatx.T, flatx) / flatx.shape[1]
    U, S, V = numpy.linalg.svd(sigma)
    eps = 0.0001
    return numpy.dot(numpy.dot(U, numpy.diag(1. / numpy.sqrt(S + eps))), U.T)


def zca_whitening(x, principal_components):
#    x = x.transpose(1,2,0)
    flatx = numpy.reshape(x, (x.size))
    whitex = numpy.dot(flatx, principal_components)
    x = numpy.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
    return x

def load_imagenet():
        import scipy.misc
        classes = glob.glob('../../DATASET/tiny-imagenet-200/train/*')
        x_train,y_train = [],[]
        cpt=0
        for c,name in zip(range(200),classes):
                print name
                files = glob.glob(name+'/images/*.JPEG')
                for f in files:
                        x_train.append(scipy.misc.imread(f, flatten=False, mode='RGB'))
                        y_train.append(c)
	return asarray(x_train),asarray(y_train)



def load_svhn():
        import scipy.io as sio
        train_data = sio.loadmat('../../DATASET/train_32x32.mat')
        x_train = train_data['X'].transpose([3,2,0,1]).astype('float32')
        y_train = concatenate(train_data['y']).astype('int32')-1
        test_data = sio.loadmat('../../DATASET/test_32x32.mat')
        x_test = test_data['X'].transpose([3,2,0,1]).astype('float32')
        y_test = concatenate(test_data['y']).astype('int32')-1
        print y_test
        return [x_train,y_train],[x_test,y_test]



def unpickle100(file,labels,channels):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    if(channels==1):
        p=dict['data'][:,:1024]*0.299+dict['data'][:,1024:2048]*0.587+dict['data'][:,2048:]*0.114
        p = p.reshape((-1,1,32,32))#dict['data'].reshape((-1,3,32,32))
    else:
        p=dict['data']
        p = p.reshape((-1,channels,32,32)).astype('float64')#dict['data'].reshape((-1,3,32,32))
    if(labels == 0 ):
        return p
    else:
        return asarray(p),asarray(dict['fine_labels'])



def unpickle(file,labels,channels):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    if(channels==1):
        p=dict['data'][:,:1024]*0.299+dict['data'][:,1024:2048]*0.587+dict['data'][:,2048:]*0.114
        p = p.reshape((-1,1,32,32))#dict['data'].reshape((-1,3,32,32))
    else:
        p=dict['data']
        p = p.reshape((-1,channels,32,32)).astype('float64')#dict['data'].reshape((-1,3,32,32))
    if(labels == 0 ):
        return p
    else:
        return asarray(p),asarray(dict['labels'])





def load_mnist():
        mndata = file('../DATASET/MNIST.pkl','rb')
        data=cPickle.load(mndata)
        mndata.close()
        return [concatenate([data[0][0],data[1][0]]).reshape(60000,1,28,28),concatenate([data[0][1],data[1][1]])],[data[2][0].reshape(10000,1,28,28),data[2][1]]

def load_cifar(channels=1):
        path = '../../DATASET/cifar-10-batches-py/'
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for i in ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']:
                PP = unpickle(path+i,1,channels)
                x_train.append(PP[0])
                y_train.append(PP[1])
        x_test,y_test = unpickle(path+'test_batch',1,channels)
        x_train = concatenate(x_train)
        y_train = concatenate(y_train)
        return [x_train,y_train],[x_test,y_test]



def load_cifar100(channels=1):
        path = '../../DATASET/cifar-100-python/'
        PP = unpickle100(path+'train',1,channels)
        x_train = PP[0]
        y_train = PP[1]
        PP = unpickle100(path+'test',1,channels)
        x_test = PP[0]
        y_test = PP[1]
        return [x_train,y_train],[x_test,y_test]




def train(x_train,y_train,x_test,y_test,session,train_opt,loss,accu,x,y_,test,name='caca',n_epochs=5):
        n_train = x_train.shape[0]/batch_size
        n_test  = x_test.shape[0]/batch_size
        train_loss          = []
        test_loss           = []
        for e in xrange(n_epochs):
                print 'epoch',e
                for i in xrange(n_train):
                        session.run(train_opt,feed_dict={x:x_train[batch_size*i:batch_size*(i+1)],y_:y_train[batch_size*i:batch_size*(i+1)],test:True})
                        train_loss.append(session.run(loss,feed_dict={x:x_train[batch_size*i:batch_size*(i+1)],y_:y_train[batch_size*i:batch_size*(i+1)],test:True}))
                acc1 = 0
                acc2 = 0
                for i in xrange(n_test):
                        acc1+=session.run(accu,feed_dict={x:x_test[batch_size*i:batch_size*(i+1)],y_:y_test[batch_size*i:batch_size*(i+1)],test:False})
                test_loss.append(acc1/n_test)
		print test_loss[-1]
	return train_loss,test_loss





##################################################
def compute_loss(logits, labels):
	labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
#	tf.add_to_collection('losses', cross_entropy_mean)
	return cross_entropy_mean#tf.add_n(tf.get_collection('losses'), name='total_loss')







