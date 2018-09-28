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

class schedule:
    def __init__(self,lr_,opt):
	self.lr_ = lr_
	self.opt = opt
    def lr(self,t):
	if(self.opt=='linear'):
	    return self.lr_
	elif(self.opt=='sqrt'):
	    return self.lr_/sqrt(1+t)




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
        self.initop_thetaq      = [l.init_thetaq() for l in layers]
        ### WEIGHTS UPDATES OP
	self.updates_BV      = [l.update_BV() for l in layers[1:]]
        self.updates_m       = [l.update_m() for l in layers[1:-1]]
	self.updates_v2      = [l.update_v2() for l in layers[1:-1]]
	self.updates_firstlayer = tf.group(layers[0].update_v2(),layers[0].update_m())
	self.update_last_p   = layers[-1].update_p() # SEPARATE DEPENDING ON SUP UNSUP TRAINING
        self.updates_sigma   = [l.update_sigma() for l in layers[1:]]
        self.updates_Wk      = [l.update_Wk() for l in layers[1:]]
        self.updates_pi      = [l.update_pi() for l in layers[1:]]
        ## GATHER LOSSES
        self.KL              = KL(layers)
        self.like            = likelihood(layers)
        # GATHER SAMPLING
        self.samplesclass    = [sampleclass(layers,k,sigma=self.sigma) for k in xrange(layers[-1].R)]
        self.samples         = sample(layers,sigma=self.sigma)
        self.samplet         = sampletrue(layers)
    def get_sigmas(self):
	return self.session.run([l.sigmas2_ for l in self.layers[1:]])
    def get_Ws(self):
	Ws = []
	for l in self.layers[1:]:
	    if(isinstance(l,layers_.ConvPoolLayer)):
		Ws.append(self.session.run(l.W))
	    else:
                Ws.append(self.session.run(l.W))
	return Ws
    def get_params(self):
        params = []
        for l in self.layers[1:]:
            if(isinstance(l,layers_.ConvPoolLayer) or isinstance(l,layers_.FinalLayer)):
                params.append([self.session.run(l.W),self.session.run(l.sigmas2_),self.session.run(l.pi),self.session.run(l.b_)])
            else:
                params.append([self.session.run(l.W),self.session.run(l.sigmas2_),self.session.run(l.pi),self.session.run(l.b_),self.session.run(l.V_)])
        return params
    def layer_E_step(self,l,random=1,fineloss=1,verbose=0):
        loss = []
        GAIN = self.session.run(self.KL)
	if(l==0): self.session.run(self.updates_firstlayer)
	if(l<len(self.updates_v2)):
            self.session.run(self.updates_v2[l])
            if(verbose): print 'V2',l,self.session.run(self.KL)
        if(random==0): iih = self.layers[l+1].m_indices
        else:   iih = self.layers[l+1].m_indices[permutation(len(self.layers[l+1].m_indices))]
	if(isinstance(self.layers[l+1],layers_.ConvPoolLayer)): # ALL BUT THE LAST LAYER
            for i in iih:
                self.session.run(self.updates_m[l],feed_dict={self.layers[l+1].i_:int32(i[0]),
                                                                        self.layers[l+1].j_:int32(i[1]),
                                                                        self.layers[l+1].k_:int32(i[2])})
                if(fineloss):loss.append(self.session.run(self.KL))
            if(verbose): print 'M',l,self.session.run(self.KL)
	elif(isinstance(self.layers[l+1],layers_.DenseLayer)):
            for i in iih:
		self.session.run(self.updates_m[l],feed_dict={self.layers[l+1].k_:int32(i)})
                if(fineloss):loss.append(self.session.run(self.KL))
            if(verbose): print 'M',l,self.session.run(self.KL)
	else:
            self.session.run(self.update_last_p)
            loss.append(self.session.run(self.KL))
            if(verbose): print 'P',l,loss[-1]
        L = self.session.run(self.KL)
        return L-GAIN
    def layer_M_step(self,lay,random=1,fineloss=1,verbose=0):
        loss = []
	l    = lay
	GAIN = self.session.run(self.like)
	if(verbose): print 'BEFORE M ',str(l),GAIN
        self.session.run(self.updates_sigma[l])
        if(verbose): print 'SIGMA ',str(l),self.session.run(self.like)
        self.session.run(self.updates_BV[l])
	if(verbose): print 'BV',self.session.run(self.like)
        if(random==0): iih = self.layers[l+1].W_indices
        else: iih = self.layers[l+1].W_indices[permutation(len(self.layers[l+1].W_indices))]
        if(isinstance(self.layers[l+1],layers_.DenseLayer)):
            t=time.time()
            for kk in iih:
                self.session.run(self.updates_Wk[l],feed_dict={self.layers[l+1].k_:int32(kk)})
                if(fineloss): loss.append(self.session.run(self.like))
	    if(verbose): print 'DW',self.session.run(self.like)
        elif(isinstance(self.layers[l+1],layers_.ConvPoolLayer)):
            t=time.time()
            for kk in iih:
                self.session.run(self.updates_Wk[l],feed_dict={self.layers[l+1].r_:int32(kk[1]),
								self.layers[l+1].k_:int32(kk[0]),
								self.layers[l+1].i_:int32(kk[2]),
								self.layers[l+1].j_:int32(kk[3])})
                if(fineloss):loss.append(self.session.run(self.like))
	    if(verbose): print 'CW',self.session.run(self.like),
        else:# LAST LAYER
            self.session.run(self.updates_Wk[-1])
            if(fineloss): loss.append(self.session.run(self.like))
	    if(verbose): print 'LW',self.session.run(self.like),
        self.session.run(self.updates_pi[l])
	if(verbose): print 'PI',self.session.run(self.like)
        if(fineloss): loss.append(self.session.run(self.like)) 
        L = self.session.run(self.like)
        return L-GAIN
    def E_step(self,rcoeff,fineloss=0,random=0,verbose=0):
        GAINS      = 0
	LAYER_GAIN = rcoeff*2
	mini_cpt   = 0
        print 'Beginning of E-step...',self.session.run(self.KL),self.session.run(self.like)
        while(LAYER_GAIN>rcoeff and mini_cpt<80):
            mini_cpt  +=1
            LAYER_GAIN = 0
            for l in range(self.L-1):
                layer_cpt,g_ = 0,rcoeff+1
                while(g_>rcoeff and layer_cpt<80):
                    layer_cpt += 1
                    g_=self.layer_E_step(l,random=random,fineloss=fineloss,verbose=verbose)
                    LAYER_GAIN+=g_
	    GAINS+= LAYER_GAIN
	return GAINS
    def M_step(self,rcoeff,fineloss=0,random=0,verbose=0):
        GAINS = 0
	print 'Beginning of M-step...',self.session.run(self.like),self.session.run(self.KL)
        for l in range(self.L-1):
            layer_cpt,g_ = 0,rcoeff+1
            while(g_>rcoeff and layer_cpt<80):
                g_=self.layer_M_step(l,random=random,fineloss=fineloss,verbose=verbose)
                GAINS+=g_
	return GAINS
    def E_step2(self,rcoeff,fineloss=0,random=0,verbose=0):
        GAINS      = 0
        g = rcoeff+1
        layer_cpt = 0
        while(g>rcoeff and layer_cpt<40):
	    layer_cpt+=1
	    g=0
            for l in range(self.L-1):
                g_=self.layer_E_step(l,random=random,fineloss=fineloss,verbose=verbose)
                g+=g_
	    GAINS+=g
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
            self.set_output_mask(zeros(x.shape[0]).astype('float32'))
        else:
	    self.set_output_mask(ones(x.shape[0]).astype('float32'))
    def init_thetaq(self):
        """this function is used alone when for example testing
        the model on a new dataset (using init_dataset),
        then the parameters of the model are kept as the 
        trained ones but one aims at correct initialization 
        of the Q fistribution parameters. 
        For initialization of the whole model see the below fun"""
	for l in xrange(self.L):
            self.session.run(self.initop_thetaq[l])
    def sample(self,sigma):
        return self.session.run(self.samples,feed_dict={self.sigma:float32(sigma)})
    def sampleclass(self,sigma,k):
        return self.session.run(self.samplesclass[k],feed_dict={self.sigma:float32(sigma)})
    def get_input(self):
        return self.session.run(self.layers[0].m)
    def reconstruct(self):
        return self.session.run(self.samplet)
    def predict(self):
        return squeeze(self.session.run(self.layers[-1].p_))
    def set_input_mask(self,mask):
	self.session.run(tf.assign(self.layers[0].mask,mask))
    def set_output_mask(self,mask):
        self.session.run(tf.assign(self.layers[-1].mask,mask))



def train_layer_model(model,rcoeff_schedule,CPT,random=0,fineloss=1,return_time=0,verbose=0):
    cpt  = 0
    LIKE = [model.session.run(model.like)]
    GAIN = 1
    print "INIT",model.session.run(model.KL),model.session.run(model.like)
    while(cpt<CPT and GAIN>0):
        print cpt
        cpt         += 1
	g = model.E_step(rcoeff=rcoeff_schedule.lr(cpt),random=random,fineloss=fineloss,verbose=verbose)
	print "AFTER E",model.session.run(model.KL)
        g = model.M_step(rcoeff=rcoeff_schedule.lr(cpt),random=random,fineloss=fineloss,verbose=verbose)
	LIKE.append(model.session.run(model.like))
        print "AFTER M",LIKE[-3:]
	GAIN = g
    return LIKE


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



def load_data(DATASET,k=-1):
    if(DATASET=='MNIST'):
        batch_size = 50
        mnist         = fetch_mldata('MNIST original')
        x             = mnist.data.reshape(70000,1,28,28).astype('float32')
        y             = mnist.target.astype('int32')
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=10000,stratify=y)
        input_shape   = (batch_size,28,28,1)
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
	if(k>=0):
	    x_train = x_train[y_train==k]
	    y_train = y_train[y_train==k]
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
    if(DATASET=='CIFAR'):
        x_train          -= x_train.mean((1,2,3),keepdims=True)
        x_test           -= x_test.mean((1,2,3),keepdims=True)
    x_train          /= abs(x_train).max((1,2,3),keepdims=True)
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
        classes = glob.glob('../DATASET/tiny-imagenet-200/train/*')
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
        train_data = sio.loadmat('../DATASET/train_32x32.mat')
        x_train = train_data['X'].transpose([3,2,0,1]).astype('float32')
        y_train = concatenate(train_data['y']).astype('int32')-1
        test_data = sio.loadmat('../DATASET/test_32x32.mat')
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
        path = '../DATASET/cifar-10-batches-py/'
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
        path = '../DATASET/cifar-100-python/'
        PP = unpickle100(path+'train',1,channels)
        x_train = PP[0]
        y_train = PP[1]
        PP = unpickle100(path+'test',1,channels)
        x_test = PP[0]
        y_test = PP[1]
        return [x_train,y_train],[x_test,y_test]











