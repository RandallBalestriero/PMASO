import tensorflow as tf
from pylab import *
import layers as layers_



def init_dataset(x,layers,y=None):
	updates_op = []
        updates_op.append(tf.assign(layers[0].m,x))
        if(isinstance(layers[-1],layers_.SupFinalLayer)):
                updates_op.append(tf.assign(layers[-1].p_,tf.expand_dims(tf.one_hot(y,layers[-1].R),0)))
	return updates_op

def init_theta(layers):
	updates_op = []
	for i in xrange(1,len(layers)):
                updates_op+=layers[i].init_theta()
	return updates_op

def init_thetaq(layers):
	updates_op = []
	for i in xrange(1,len(layers)):
                updates_op+=layers[i].init_thetaq()
	return updates_op




def update_v2(layers):
        v2 = []
        for l in layers:
                v2+=l.update_v2()
        return v2

def update_vmpk(layers):
        v2 = []
        for l in layers[1:]:
                v2+=[l.update_vmpk()]
        return v2

def update_Wk(layers):
        v2 = []
        for l in layers[1:]:
                v2+=l.update_Wk()
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




def sample(layers,Ks=None,sigma=1):
    """Ks is used if one wants to pre imposed some 
    t variables at different layers, one can provide
    a specific set of t for any layer but must be
    of the same shape as the inner layer variable"""
    if(Ks == None):
        Ks = [None]*len(layers)
    samples=0 # variables that carries the per layer samples representation going from layer to layer
    for i in xrange(len(layers)-1,0,-1):
	samples = layers[i].sample(samples,Ks[i],sigma)
    return samples



def sampletrue(layers):
    s=float32(1)
    try:
        return layers[1].deconv()
    except:
        return layers[1].backward(0)


def SSE(x,y):
    return tf.reduce_sum(tf.square(x-y))
	
		


def likelihood(layers):
    """ gather all the per layer likelihoods
    and add them together as derived in the paper"""
    like = 0
    for l in layers:
        like+=l.likelihood()
    return like
	

def KL(layers):
    """gather the KL divergence by
    summing the per layers one as derived"""
    kl = 0
    for l in layers:
        kl+=l.KL()
    return kl



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



def load_data(DATASET):
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
  
#    x_train          -= x_train.mean((1,2,3),keepdims=True)
    x_train          /= abs(x_train).max((1,2,3),keepdims=True)#/10
#    x_test           -= x_test.mean((1,2,3),keepdims=True)
    x_test           /= abs(x_test).max((1,2,3),keepdims=True)
    x_train           = x_train.astype('float32')
    x_test            = x_test.astype('float32')
    y_train           = array(y_train).astype('int32')
    y_test            = array(y_test).astype('int32')
    return x_train,y_train,x_test,y_test



class model:
    def __init__(self,layers):
        self.layers    = layers
        self.L         = len(layers)
        self.x         = tf.placeholder(tf.float32,shape=layers[0].input_shape)
        self.y         = tf.placeholder(tf.int32,shape=[layers[0].input_shape[0]]) # MIGHT NOT BE USED DEPENDING ON SETTING
        self.sigma     = tf.placeholder(tf.float32)
        session_config = tf.ConfigProto(allow_soft_placement=False,log_device_placement=True)
        session_config.gpu_options.allow_growth=True
        session        = tf.Session(config=session_config)
        init           = tf.global_variables_initializer()
        session.run(init)
        self.session=session
        ## INITIALIZATION
        self.dataset_init_op = init_dataset(self.x,layers,self.y)
        self.theta_inits_op  = init_theta(layers)
        self.thetaq_inits_op = init_thetaq(layers)
        ### GATHER UPDATES
        self.updates_v2      = update_v2(layers)
        self.updates_vmpk    = update_vmpk(layers)
        self.updates_sigma   = update_sigma(layers)
        self.updates_Wk      = update_Wk(layers)
        self.updates_pi      = update_pi(layers)
        ## GATHER LOSSES
        self.KL              = KL(layers)
        self.like            = likelihood(layers)
        # GATHER SAMPLING
        self.samples         = sample(layers,sigma=self.sigma)
        self.samplet         = sampletrue(layers)
    def init_theta(self):
        self.session.run(self.theta_inits_op)
    def init_dataset(self,x,y):
        self.session.run(self.dataset_init_op,feed_dict={self.x:x,self.y:y})
    def init_thetaq(self):
        for op in self.thetaq_inits_op:
            self.session.run(op)
    def sample(self,sigma):
        return self.session.run(self.samples,feed_dict={self.sigma:float32(sigma)})
    def reconstruct(self):
        return self.session.run(self.samplet)


def train_model(model,rcoeff,CPT):
    global_global_L = 0
    cpt             = 0
    LIKELIHOOD      = []
    KL              = []
    rcoeff          = 4000
    L               = rcoeff*2
    while((L-global_global_L)>rcoeff and cpt<CPT):
        cpt+=1
        global_global_L=model.session.run(model.like)
        LIKELIHOOD.append(global_global_L)
        print "CACA",global_global_L,model.session.run(model.KL)
#        session.run(updates_sigma)
#        L = session.run(LIKl)
#        LIKELIHOOD.append(L)
#        print "AFTER S",L
#        session.run(updates_v2)
#        print "AFTER V",session.run(KLl)
        ########################################### E STEP
        L        = rcoeff*2
        global_L = 0
        while((L-global_L)>rcoeff):
            global_L = model.session.run(model.KL)
            for l in xrange(model.L-1):
                if(isinstance(model.layers[l+1],layers_.PoolLayer)):
                    session.run(model.updates_vmpk[l])
                    L = model.session.run(model.KL)
                    print "AFTER POOL M",l,L
                else:
                    L = rcoeff*2
                    prev_L = 0
                    while((L-prev_L)>rcoeff):
                        prev_L = model.session.run(model.KL)
                        for kk in permutation(model.layers[l+1].K).astype('int32'):
                            model.session.run(model.updates_vmpk[l],feed_dict={model.layers[l+1].k_:int32(kk)})
                            L = model.session.run(model.KL)
                            print "AFTER M",l,kk,L
            if(isinstance(model.layers[-1],layers_.UnsupFinalLayer)):
                model.session.run(model.updates_vmpk[-1])
                L = model.session.run(model.KL)
                print "AFTER LAST P",L
            print L,">",global_L
        ########################################### E STEP
        L        = rcoeff*2
        global_L = 0
        while((L-global_L)>rcoeff):
            global_L = model.session.run(model.like)
            model.session.run(model.updates_pi)
            L = model.session.run(model.like)
            LIKELIHOOD.append(L)
            print "AFTER pi",L
            model.session.run(model.updates_sigma)
            L = model.session.run(model.like)
            LIKELIHOOD.append(L)
            print "AFTER S",L,model.session.run([l.sigmas2 for l in model.layers[1:]])
            for l in xrange(model.L-1):
                if(isinstance(model.layers[l+1],layers_.PoolLayer)):
                    0
                else:
                    L = rcoeff*2
                    prev_L = 0
                    while((L-prev_L)*rcoeff):
                        prev_L = model.session.run(model.like)
                        for kk in permutation(model.layers[l+1].K).astype('int32'):
                            model.session.run(model.updates_Wk[l],feed_dict={model.layers[l+1].k_:int32(kk)})
                            L = model.session.run(model.like)
                            LIKELIHOOD.append(L)
                            print "AFTER W",l,kk,L
                            model.session.run(model.updates_sigma[l])
                            L = model.session.run(model.like)
                            LIKELIHOOD.append(L)
                            print "AFTER S",L
                    model.session.run(model.updates_Wk[-1])
                    L = model.session.run(model.like)
                    LIKELIHOOD.append(L)
                    print "AFTER W",L
            model.session.run(model.updates_sigma)
            L = model.session.run(model.like)
            LIKELIHOOD.append(L)
            print "AFTER S",L,model.session.run([l.sigmas2 for l in model.layers[1:]])
    return LIKELIHOOD,KL





###################################################################
#
#
#                       UTILITY FOR CIFAR10 & MNIST
#
#
###################################################################



import cPickle
import glob
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split



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











