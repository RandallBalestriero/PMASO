import tensorflow as tf
from pylab import *
import layers as layers_


def init_latent(x,layers):
	new_v = []
	for i in xrange(len(layers)):
		if(isinstance(layers[i],layers_.InputLayer)):
                        new_v.append(tf.assign(layers[i].m,x))
                else:
                        new_v+=layers[i].init_latent()
	return new_v



def update_v2(layers):
        v2 = []
        for l in layers:
                v2+=l.update_v2()
        return v2


def update_p(layers):
        v2 = []
        for l in layers:
                v2+=l.update_p()
        return v2



def update_W(layers):
        v2 = []
        for l in layers:
                v2+=l.update_W()
        return v2



def update_m(layers):
        v2 = []
        for l in layers:
                v2+=l.update_m()
        return v2




def update_pk(layers):
        v2 = []
        for l,i in zip(layers[1:-1],range(len(layers)-2)):
                v2.append([])
                for k in xrange(l.K):
                    v2[i]+=l.update_pk(k)
        return v2

def update_mk(layers):
        v2 = []
        for l,i in zip(layers[1:-1],range(len(layers)-2)):
                v2.append([])
                for k in xrange(l.K):
                    v2[i]+=l.update_mk(k)
        return v2


def update_Wk(layers):
        v2 = []
        for l,i in zip(layers[1:-1],range(len(layers)-2)):
                v2.append([])
                for k in xrange(l.K):
                    v2[i]+=l.update_Wk(k)
        return v2






def update_W(layers):
        v2 = []
        for l in layers:
                v2+=l.update_W()
        return v2



def update_m(layers):
        v2 = []
        for l in layers:
                v2+=l.update_m()
        return v2









def update_sigma(layers):
        v2 = []
        for l in layers:
                v2+=l.update_sigma()
        return v2


def update_pi(layers):
        v2 = []
        for l in layers:
                v2+=l.update_pi()
        return v2









def sample(layers):
	s=float32(1)
        for i in xrange(1,len(layers)):
		s = layers[-i].sample(s)
	return s



def sampletrue(layers):
        s=float32(1)
        return layers[1].backward(0)


def SSE(x,y):
	return tf.reduce_sum(tf.pow(x-y,2))
	
		


def likelihood(layers):
        like = 0
        for l in layers:
                like+=l.likelihood()
    #	# FIRST LAYER
#	like=0# a1+a2+a3+a4+a5
#	for l in xrange(1,len(layers)-1):
#	        a1 = -SSE(layers[l-1].M,layers[l].backward())/(2*layers[l].sigmas2[0])
#		if(isinstance(layers[l],DenseLayer)):
#	                k  = layers[l].bs*layers[l].D*(tf.log(layers[l].sigmas2[0]+eps)/2+tf.log(2*3.14159)/2)+tf.reduce_sum(layers[l].p*tf.expand_dims(tf.log(layers[l].pi+eps),0))
#			a2 = tf.reduce_sum(layers[l].W*layers[l].W,axis=2)/(2*layers[l].sigmas2)+1/(2*layers[l+1].sigmas2)
#		else:
#	                k  = layers[l].bs*layers[l].D*(tf.log(layers[l].sigmas2[0]+eps)/2+tf.log(2*3.14159)/2)+tf.reduce_sum(layers[l].p*tf.reshape(tf.log(layers[l].pi+eps),(1,layers[l].K,layers[l].R,1,1)))
#			a2 = tf.reshape(tf.transpose(tf.reduce_sum(layers[l].W*layers[l].W,axis=[1,2,3])),(1,layers[l].K,layers[l].R,1,1))
#		a3 = tf.reduce_sum(a2*tf.reduce_sum((tf.pow(layers[l].m,2))*(tf.pow(layers[l].p,2)-layers[l].p),axis=0)/(2*layers[l].sigmas2[0]))
#		like+=a1+a3+k#-tf.reduce_sum(layers[l].p)##tf.reduce_sum(tf.expand_dims(a2,0)*layers[l].p*layers[l].v2)
#	l+=1
#	# LAST LAYER
#        k  = layers[l].bs*layers[l].D*(tf.log(layers[l].sigmas2+eps)/2+tf.log(2*3.14159)/2)+tf.reduce_sum(layers[l].p*tf.expand_dims(tf.log(layers[l].pi+eps),0))
#        if(isinstance(layers[-2],DenseLayer)):
#                a1 = -tf.reduce_sum(tf.reduce_sum(tf.pow(tf.expand_dims(layers[l-1].M,1)-tf.expand_dims(layers[l].W[0],0),2),axis=2)*layers[-1].p[:,0,:])/(2*layers[l].sigmas2)
#        else:
#                a1 = -tf.reduce_sum(tf.reduce_sum(tf.pow(tf.expand_dims(tf.reshape(layers[-2].M,(layers[l].bs,layers[-2].D)),1)-tf.expand_dims(layers[-1].W[0],0),2),axis=2)*layers[-1].p[:,0,:])/(2*layers[-1].sigmas2)                       
#        like+=a1+k
	return like
	

def KL(layers):
	kl = 0
        for l in layers:
                kl+=l.KL()
#        for l in xrange(1,len(layers)-1):
#	        kl += tf.reduce_sum(layers[l].p*(tf.log(layers[l].p+eps)-tf.log(layers[l].v2+eps)/2))
#	kl += tf.reduce_sum(layers[-1].p*(tf.log(layers[-1].p+eps)))
	return kl#likelihood(layers)-kl







def get_p(layers):
	p=[]
	for l in layers[1:]:
		p.append(l.p)
	return p

def get_m(layers):
        p=[]
        for l in layers[1:]:
                p.append(l.M)
        return p

def get_v(layers):
        p=[]
        for l in layers[1:]:
                p.append(l.v2)
        return p





import cPickle
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







