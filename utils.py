import tensorflow as tf
import time
from pylab import *
import layers as layers_
import itertools
from random import shuffle





#############################################################################################
#
#
#                       UPDATE and SAMPLE HELPER
#
#
#############################################################################################




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

def sampleclass(layers,K,sigma=1):
    """Ks is used if one wants to pre imposed some 
    t variables at different layers, one can provide
    a specific set of t for any layer but must be
    of the same shape as the inner layer variable"""
    Ks  = [None]*(len(layers)-1)
    pp = zeros((layers[-1].input_shape[0],1,layers[-1].R),dtype='float32')
    pp[:,:,K]=1
    Ks.append(pp)
    samples=0 # variables that carries the per layer samples representation going from layer to layer
    for i in xrange(len(layers)-1,0,-1):
	samples = layers[i].sample(samples,Ks[i],sigma)
    return samples


def sampletrue(layers):
    s=float32(1)
    try:
        return layers[1].deconv()
    except:
        return layers[1].sample(0,deterministic=True)


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



class model:
    def __init__(self,layers):
        self.layers    = layers
        self.L         = len(layers)
        self.x         = tf.placeholder(tf.float32,shape=layers[0].input_shape)
        self.y         = tf.placeholder(tf.int32,shape=[layers[0].input_shape[0]]) # MIGHT NOT BE USED DEPENDING ON SETTING
        self.sigma     = tf.placeholder(tf.float32)
        # INIT SESSION
        session_config = tf.ConfigProto(allow_soft_placement=False,log_device_placement=True)
        session_config.gpu_options.allow_growth=True
        session        = tf.Session(config=session_config)
        init           = tf.global_variables_initializer()
        session.run(init)
        self.session=session
        ### INITIALIZATION OP
        self.initx              = tf.assign(self.layers[0].m,self.x)
        self.inity              = tf.assign(self.layers[-1].p_,tf.expand_dims(tf.one_hot(self.y,self.layers[-1].R),0))
        self.initop_thetaq_random  = [l.init_thetaq() for l in layers[1:]]
        ### WEIGHTS UPDATES OP
        self.updates_m       = [l.update_m() for l in layers[1:-1]]
        self.updates_p       = [l.update_p() for l in layers[1:-1]]
        self.updates_rho     = [l.update_rho() for l in layers[1:-1]]
	self.updates_v2      = [l.update_v2() for l in layers[1:-1]]
	self.update_last_p   = layers[-1].update_p() # SEPARATE DEPENDING ON SUP UNSUP TRAINING
#self.updates_DD      = [l.update_DD() for l in layers[1:]]
        self.updates_d       = [l.update_dropout() for l in layers[1:-1]]
#self.updates_U       = [l.update_U() for l in layers[1:]]
#        self.updates_sigma_global = update_sigma(layers,0)
        self.updates_sigma   = [l.update_sigma() for l in layers[1:]]
        self.updates_Wk      = [l.update_Wk() for l in layers[1:]]
        self.updates_pi      = [l.update_pi() for l in layers[1:]]
	self.updates_V       = [l.update_V() for l in layers[1:]]
#       self.updates_b       = [l.update_b() for l in layers[1:]]
        ## GATHER LOSSES
        self.KL              = KL(layers)
        self.like            = likelihood(layers)
        # GATHER SAMPLING
        self.samplesclass    = [sampleclass(layers,k,sigma=self.sigma) for k in xrange(layers[-1].R)]
        self.samples         = sample(layers,sigma=self.sigma)
        self.samplet         = sampletrue(layers)
    def get_sigmas(self):
	return self.session.run([l.sigmas2_ for l in self.layers[1:]])
#   def get_bs(self):
#       return self.session.run([l.b_ for l in self.layers[1:]])
    def get_Ws(self):
	Ws = []
	for l in self.layers[1:]:
	    if(isinstance(l,layers_.ConvPoolLayer)):
		Ws.append(self.session.run(l.W))
	    else:
                Ws.append(self.session.run(l.W))
	return Ws
    def layer_E_step(self,lay,random=1,fineloss=1):
        loss = []
	l    = lay
        GAIN = self.session.run(self.KL)
        print 'BEFORE',l,GAIN,
	if(l<len(self.updates_v2)):
            t = time.time()
            self.session.run(self.updates_v2[lay])
            print 'AFTER V2',l,self.session.run(self.KL),self.session.run(self.like),time.time()-t,
        if(random==0):
            iih = self.layers[l+1].m_indices
        else:
            perm = permutation(len(self.layers[l+1].m_indices))
            iih = self.layers[l+1].m_indices[perm]
	if(isinstance(self.layers[l+1],layers_.ConvPoolLayer)): # ALL BUT THE LAST LAYER
	    t = time.time()
            for i in iih:
                self.session.run(self.updates_m[l],feed_dict={self.layers[l+1].i_:int32(i[0]),
                                                                        self.layers[l+1].j_:int32(i[1]),
                                                                        self.layers[l+1].k_:int32(i[2])})
                if(fineloss):
                    t=time.time()
                    L = self.session.run(self.KL)
                    loss.append(L)
            print 'AFTER CONV ',str(l),' M',self.session.run(self.KL),self.session.run(self.like),time.time()-t,
#	    t = time.time()
#            for i in iih:
#                self.session.run(self.updates_p[l],feed_dict={self.layers[l+1].i_:int32(i[0]),
#                                                                        self.layers[l+1].j_:int32(i[1]),
#                                                                        self.layers[l+1].k_:int32(i[2])})
#                if(fineloss):
#                    t=time.time()
#                    L = self.session.run(self.KL)
#                    loss.append(L)
#            print 'AFTER CONV P',self.session.run(self.KL),time.time()-t,
#	    t = time.time()
#            for i in iih:
#                self.session.run(self.updates_rho[l],feed_dict={self.layers[l+1].i_:int32(i[0]),
#                                                                        self.layers[l+1].j_:int32(i[1]),
#                                                                        self.layers[l+1].k_:int32(i[2])})
#                if(fineloss):
#                    t=time.time()
#                    L = self.session.run(self.KL)
#                    loss.append(L)
#            print 'AFTER RHO',self.session.run(self.KL),time.time()-t
	elif(isinstance(self.layers[l+1],layers_.DenseLayer)):
            for i in iih:
		self.session.run(self.updates_m[l],feed_dict={self.layers[l+1].k_:int32(i)})
                if(fineloss):
                    t=time.time()
                    L = self.session.run(self.KL)
                    loss.append(L)
            print 'AFTER DENS ',str(l),' M',self.session.run(self.KL),self.session.run(self.like)
#	    for i in iih:
#                self.session.run(self.updates_p[l],feed_dict={self.layers[l+1].k_:int32(i)})
#                if(fineloss):
#                    t=time.time()
#                    L = self.session.run(self.KL)
#                    loss.append(L)
#            print 'AFTER DENS P',self.session.run(self.KL),
#            for i in iih:
#               self.session.run(self.updates_d[l],feed_dict={self.layers[l+1].k_:int32(i)})
#                if(fineloss):
#                    t=time.time()
#                    L = self.session.run(self.KL)
#                    loss.append(L)
#            print 'AFTER DENS D',self.session.run(self.KL)
	else:
	    if(self.hold_last_p==0):
	        self.session.run(self.update_last_p)
	        L = self.session.run(self.KL)
	        loss.append(L)
#                print 'AFTER LAST P',self.session.run(self.KL)
        L = self.session.run(self.KL)
#	GAIN = L-GAIN
        return L-GAIN
    def layer_M_step(self,lay,random=1,fineloss=1):
        loss = []
	l    = lay
	GAIN = self.session.run(self.like)
        print 'BEFORE M',GAIN
        self.session.run(self.updates_sigma[l])
 #       L = self.session.run(self.like)
        print 'AFTER SIGMA ',str(l),self.session.run(self.like)
        self.session.run(self.updates_V[l])
#        print 'AFTER V',self.session.run(self.like)
        if(random==0):
            iih = self.layers[l+1].W_indices
        else:
            perm = permutation(len(self.layers[l+1].W_indices))
            iih = self.layers[l+1].W_indices[perm]
        if(isinstance(self.layers[l+1],layers_.DenseLayer)):
            t=time.time()
            for kk in iih:
                self.session.run(self.updates_Wk[l],feed_dict={self.layers[l+1].k_:int32(kk)})
                newtime = time.time()-t
                if(fineloss):
                    t=time.time()
                    L = self.session.run(self.like)
                    loss.append(L)
            print 'AFTER DENS ',str(l),' W',self.session.run(self.like)
        elif(isinstance(self.layers[l+1],layers_.ConvPoolLayer)):
            t=time.time()
            for kk in iih:
                self.session.run(self.updates_Wk[l],feed_dict={self.layers[l+1].r_:int32(kk[1]),
								self.layers[l+1].k_:int32(kk[0]),
								self.layers[l+1].i_:int32(kk[2]),
								self.layers[l+1].j_:int32(kk[3])})
                if(fineloss):
                    t=time.time()
                    L = self.session.run(self.like)
                    loss.append(L)
            newtime = time.time()-t
            print 'AFTER CONV W',self.session.run(self.like),newtime
        else:# LAST LAYER
            self.session.run(self.updates_Wk[-1])
            if(fineloss):
                L = self.session.run(self.like)
                loss.append(L)
            print 'AFTER LAST W',self.session.run(self.like)
#        self.session.run(self.updates_b[l])
#        if(fineloss):
#            L = self.session.run(self.like)
#            loss.append(L)
#        print 'AFTER BBBBB',self.session.run(self.like)
        self.session.run(self.updates_pi[l])
        print 'AFTER PI',self.session.run(self.like)
        if(fineloss):
            L = self.session.run(self.like)
            loss.append(L) 
#        self.session.run(self.updates_sigma[l])
        L = self.session.run(self.like)
#	GAIN = L-GAIN
#        print 'AFTER SIGMA',self.session.run(self.like)
        return L-GAIN
    def E_step(self,rcoeff,fineloss=0,random=0):
        GAINS      = 0
#	LAYER_LOSS = []
	LAYER_GAIN =rcoeff*2
	mini_cpt   = 0
        while(LAYER_GAIN>rcoeff and mini_cpt<80):
            mini_cpt  +=1
            LAYER_GAIN = 0
            for l in range(self.L-1):
                layer_cpt,g_ = 0,rcoeff
#		self.session.run(self.updates_DD[l:])
                while(g_>rcoeff/(self.L-1.0) and layer_cpt<40):
                    layer_cpt += 1
                    l_,g_=self.layer_E_step(l,random=random,fineloss=fineloss)
#                    LAYER_LOSS.append(l_)
                    LAYER_GAIN+=g_
                    print "E,",l,'       ',g_,">",rcoeff/(self.L-1)
	    GAINS+= LAYER_GAIN
	return GAINS
    def M_step(self,rcoeff,fineloss=0,random=0):
#        LAYER_LOSS = []
        GAINS = 0
        for l in range(self.L-1):
            layer_cpt,g_ = 0,rcoeff
            while(g_>rcoeff/(self.L-1.0) and layer_cpt<40):
                g_=self.layer_M_step(l,random=random,fineloss=fineloss)
#                LAYER_LOSS.append(l_)
                GAINS+=g_
#                print "M,",l,'          ',g_,">",rcoeff/(self.L-1)
	return GAINS
    def E_step2(self,rcoeff,fineloss=0,random=0):
        GAINS      = 0
#	LAYER_LOSS = []
	LAYER_GAIN =rcoeff*2
	mini_cpt   = 0
        while(LAYER_GAIN>rcoeff and mini_cpt<80):
            mini_cpt  +=1
            LAYER_GAIN = 0
	    g = rcoeff+1
	    layer_cpt = 0
            while(g>rcoeff and layer_cpt<40):
		layer_cpt+=1
		g=0
#		self.session.run(self.updates_DD[l:])
                for l in range(self.L-1):
#		    print l,l
                    g_=self.layer_E_step(l,random=random,fineloss=fineloss)
#                    LAYER_LOSS.append(l_)
                    g+=g_
#                    print "E,",l,'       ',g_
	    LAYER_GAIN+=g
	    GAINS+= LAYER_GAIN
	return GAINS
    def init_dataset(self,x,y=None):
        """this function initializes the variable of the
        first layer with x and possibly the last layer
        with y. Depending if y is given or not, an extra variable
        is updated that will impact the E-step. If no y is given the E step
        is done also on the last layer, to infer the p values"""
        self.session.run(self.initx,feed_dict={self.x:x})
        if(y is not None):
            self.session.run(self.inity,feed_dict={self.y:y})
            self.hold_last_p = 1
        else:
            self.hold_last_p = 0
    def init_thetaq(self):
        """this function is used alone when for example testing
        the model on a new dataset (using init_dataset),
        then the parameters of the model are kept as the 
        trained ones but one aims at correct initialization 
        of the Q fistribution parameters. 
        For initialization of the whole model see the below fun"""
	for l in xrange(self.L-1-self.hold_last_p):
            self.session.run(self.initop_thetaq_random[l])
    def sample(self,sigma):
        return self.session.run(self.samples,feed_dict={self.sigma:float32(sigma)})
    def sampleclass(self,sigma,k):
        return self.session.run(self.samplesclass[k],feed_dict={self.sigma:float32(sigma)})
    def reconstruct(self):
        return self.session.run(self.samplet)
    def predict(self):
        return self.session.run(self.layers[-1].p_)[0]


def train_layer_model(model,rcoeff,CPT,random=0,fineloss=1,return_time=0):
    cpt  = 0
    LIKE = [model.session.run(model.like)]
    GAIN = 1
    print "INIT",model.session.run(model.KL),model.session.run(model.like)
    while(cpt<CPT and GAIN>0):
        print cpt
        cpt         += 1
	g = model.E_step2(rcoeff=rcoeff,random=random,fineloss=fineloss)
	print "AFTER E",model.session.run(model.KL)
        g = model.M_step(rcoeff=rcoeff,random=random,fineloss=fineloss)
	LIKE.append(model.session.run(model.like))
        print "AFTER M",LIKE[-3:]
	GAIN = g#LIKE[-1]-LIKE[-2]
    return LIKE#LOSSES,GAINS





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
    x_train          /= abs(x_train).max((1,2,3),keepdims=True)
#    x_test           -= x_test.mean((1,2,3),keepdims=True)
    x_test           /= abs(x_test).max((1,2,3),keepdims=True)
    x_train           = x_train.astype('float32')
    x_test            = x_test.astype('float32')
    y_train           = array(y_train).astype('int32')
    y_test            = array(y_test).astype('int32')
    return x_train,y_train,x_test,y_test



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











