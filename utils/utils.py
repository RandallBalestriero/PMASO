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
    def get(self,t):
        if(t==0): return 1
	if(self.opt=='linear'):
	    return self.lr_
	elif(self.opt=='sqrt'):
	    return self.lr_/sqrt(1+t)
        elif(self.opt=='exp'):
            return (t+1)**(-self.lr_)
        elif(self.opt=='mean'):
            return float32(1.0/(t+1))


def generate_batch_indices(N,bs):
    p=permutation(N)
    l=[]
    for i in xrange(N/bs):
        l.append(p[i*bs:(i+1)*bs])
    return l


def my_onehot(X,k):
    out = zeros((len(X),k),dtype='float32')
    out[range(len(X)),X]=1
    return out



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
    pp = zeros((layers[-1].input_shape[0],layers[-1].R),dtype='float32')
    pp[:,K]=1
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
	
		
def init_latent_variables(layers,N):
    #m p v
    P = dict()
    for l in layers:
        if(isinstance(l,layers_.DenseLayer)):
            P[l]=[(randn(l.K,N)*0.).astype('float32'),(ones((l.K,N,l.R))/l.R).astype('float32'),ones((l.K,N),dtype='float32')]
    return P



def likelihood(layers,batch):
    """ gather all the per layer likelihoods
    and add them together as derived in the paper"""
    like = 0
    for l in layers:
        like+=l.likelihood(batch)
    return like
	

def KL(layers):
    """gather the KL divergence by
    summing the per layers one as derived"""
    kl = 0
    for l in layers:
        kl+=l.KL()
    return kl



class model:
    def __init__(self,layers,X,X_mask=None,Y_mask=None,Y=None,batch=False):
        self.layers    = layers
        self.batch     = batch
        self.N         = len(X)
        self.bs        = layers[1].bs
        self.L         = len(layers)
        #
        self.x         = tf.placeholder(tf.float32,shape=layers[0].input_shape)
        self.y         = tf.placeholder(tf.float32,shape=[layers[0].input_shape[0],layers[-1].R]) # MIGHT NOT BE USED DEPENDING ON SETTING
        self.x_mask    = tf.placeholder(tf.float32,shape=layers[0].input_shape)
        self.y_mask    = tf.placeholder(tf.float32,shape=[layers[0].input_shape[0]])
        self.X          = X
        if(Y is None):
            if(isinstance(layers[-1],layers_.ContinuousLastLayer)): self.Y = zeros((X.shape[0],layers[-1].K),dtype='float32')
            else: self.Y = ones((X.shape[0],layers[-1].R),dtype='float32')/layers[-1].R
            self.Y_mask = ones(X.shape[0],dtype='float32')
        else:
            self.Y = Y
            if(Y_mask is None): self.Y_mask = zeros(X.shape[0],dtype='float32')
            else:               self.Y_mask = Y_mask
        if(X_mask is None): self.X_mask = zeros_like(X,dtype='float32')
        else:               self.X_mask = X_mask
        self.initx              = tf.assign(self.layers[0].m,self.x)
        if(isinstance(layers[-1],layers_.ContinuousLastLayer)):
            self.inity  = tf.assign(self.layers[-1].m_,self.y)
        else:self.inity = tf.assign(self.layers[-1].p_,self.y)
        self.initx_mask = tf.assign(self.layers[0].mask,self.x_mask)
        self.inity_mask = tf.assign(self.layers[-1].mask,self.y_mask)
        #
        self.alpha     = tf.placeholder(tf.float32)
        self.m_        = dict()
        self.p_        = dict()
        self.v2_       = dict()
        self.initm_    = dict()
        self.initp_    = dict()
        self.initv2_   = dict()
        self.initalpha = []
        for l in layers:
            if(isinstance(l,layers_.DenseLayer)): 
                self.m_[l]  = tf.placeholder(tf.float32,shape=[l.K,l.bs])
                self.p_[l]  = tf.placeholder(tf.float32,shape=[l.K,l.bs,l.R])
                self.v2_[l] = tf.placeholder(tf.float32,shape=[l.K,l.bs])
                self.initm_[l]  = tf.assign(l.m_,self.m_[l])
                self.initp_[l]  = tf.assign(l.p_,self.p_[l])
                self.initv2_[l] = tf.assign(l.v2_,self.v2_[l])
                self.initalpha.append(tf.assign(l.alpha,self.alpha))
        self.latent_variables = init_latent_variables(layers,self.N)
        self.sigma     = tf.placeholder(tf.float32)
        # INIT SESSION
        session_config = tf.ConfigProto(allow_soft_placement=False,log_device_placement=True)
        session_config.gpu_options.allow_growth=True
        session        = tf.Session(config=session_config)
        init           = tf.global_variables_initializer()
        session.run(init)
        self.session=session
        self.initop_thetaq      = [l.init_thetaq() for l in layers]
        ### WEIGHTS UPDATES OP
        self.updates_S       = [l.update_S() for l in layers[1:]]
	self.updates_BV      = [l.update_BV() for l in layers[1:]]
        self.updates_m       = [l.update_m(0) for l in layers[1:]]
        self.updates_p       = [l.update_m(1) for l in layers[1:]]
        self.updates_rho     = [l.update_m(2) for l in layers[1:]]
	self.updates_v2      = [l.update_v2() for l in layers[1:]]
	self.updates_firstlayer = tf.group(layers[0].update_v2(),layers[0].update_m())
	self.update_last_p   = layers[-1].update_p() # SEPARATE DEPENDING ON SUP UNSUP TRAINING
        self.updates_sigma   = [l.update_sigma() for l in layers[1:]]
        self.updates_Wk      = [l.update_Wk() for l in layers[1:]]
        self.updates_pi      = [l.update_pi() for l in layers[1:]]
        self.evidence        = sum([l.evidence() for l in layers[1:]])
        ## GATHER LOSSES
        self.KL              = KL(layers)
        self.like0           = likelihood(layers,0)
        self.like1           = likelihood(layers,1)
        # GATHER SAMPLING
        if(not isinstance(layers[-1],layers_.ContinuousLastLayer)):
            self.samplesclass    = [sampleclass(layers,k,sigma=self.sigma) for k in xrange(layers[-1].R)]
        self.samples         = sample(layers,sigma=self.sigma)
        self.samplet         = sampletrue(layers)
    def set_batch(self):
        # input-output
        self.session.run(self.initx,feed_dict={self.x:self.X[self.indices]})
        self.session.run(self.inity,feed_dict={self.y:self.Y[self.indices]})
        # masks
        self.session.run(self.initx_mask,feed_dict={self.x_mask:self.X_mask[self.indices]})
        self.session.run(self.inity_mask,feed_dict={self.y_mask:self.Y_mask[self.indices]})
        #latent variables
        for l in self.layers:
            if(isinstance(l,layers_.DenseLayer)):
                self.session.run(self.initm_[l],feed_dict={self.m_[l]:self.latent_variables[l][0][:,self.indices]})
                self.session.run(self.initp_[l],feed_dict={self.p_[l]:self.latent_variables[l][1][:,self.indices]})
                self.session.run(self.initv2_[l],feed_dict={self.v2_[l]:self.latent_variables[l][2][:,self.indices]})
    def save_batch(self):
        #input-output
        self.X[self.indices]=self.session.run(self.layers[0].m)
        if(isinstance(self.layers[-1],layers_.ContinuousLastLayer)):
            self.Y[self.indices]=self.session.run(self.layers[-1].m)
        else: self.Y[self.indices]=self.session.run(self.layers[-1].p_)
        #latent variables
        for l in self.layers:
            if(isinstance(l,layers_.DenseLayer)):
                self.latent_variables[l][0][:,self.indices]=self.session.run(l.m_)
                self.latent_variables[l][1][:,self.indices]=self.session.run(l.p_)
                self.latent_variables[l][2][:,self.indices]=self.session.run(l.v2_)
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
    def layer_E_step(self,l,random=1,fineloss=1,verbose=0,mp_opt=0):
        loss = []
        GAIN = self.session.run(self.KL)
	if(l==0): self.session.run(self.updates_firstlayer)
	if(verbose): print 'V2 BEFORE',l,self.session.run(self.KL),self.session.run(self.like0)
        self.session.run(self.updates_v2[l])
        if(verbose): print 'V2',l,self.session.run(self.KL),self.session.run(self.like0)
        if(random==0): iih = self.layers[l+1].m_indices
        else:   iih = self.layers[l+1].m_indices[permutation(len(self.layers[l+1].m_indices))]
	if(isinstance(self.layers[l+1],layers_.ConvPoolLayer)):
            for i in iih:
                self.session.run(self.updates_m[l],feed_dict={self.layers[l+1].i_:int32(i[0]),
                                                                        self.layers[l+1].j_:int32(i[1]),
                                                                        self.layers[l+1].k_:int32(i[2])})
                if(fineloss):loss.append(self.session.run(self.KL)) 
            for i in iih:
                self.session.run(self.updates_p[l],feed_dict={self.layers[l+1].i_:int32(i[0]),
                                                                        self.layers[l+1].j_:int32(i[1]),
                                                                        self.layers[l+1].k_:int32(i[2])})
                if(fineloss):loss.append(self.session.run(self.KL))
            for i in iih:
                self.session.run(self.updates_rho[l],feed_dict={self.layers[l+1].i_:int32(i[0]),
                                                                        self.layers[l+1].j_:int32(i[1]),
                                                                        self.layers[l+1].k_:int32(i[2])})
                if(fineloss):loss.append(self.session.run(self.KL))
            if(verbose): print 'P',l,self.session.run(self.KL)
	elif(isinstance(self.layers[l+1],layers_.DenseLayer)):
	    if(verbose): print 'BEFORE M',l,self.session.run(self.KL),self.session.run(self.like0)
	    if(mp_opt==0):
                for i in iih:
                    self.session.run(self.updates_m[l],feed_dict={self.layers[l+1].k_:int32(i)})
                    if(fineloss):loss.append(self.session.run(self.KL))
                if(verbose): print 'M',l,self.session.run(self.KL),self.session.run(self.like0)
                for i in iih:
                    self.session.run(self.updates_p[l],feed_dict={self.layers[l+1].k_:int32(i)})
                    if(fineloss):loss.append(self.session.run(self.KL))
                if(verbose): print 'P',l,self.session.run(self.KL),self.session.run(self.like0)
	    elif(mp_opt==1):
                for i in iih:
                    self.session.run(self.updates_p[l],feed_dict={self.layers[l+1].k_:int32(i)})
                if(verbose): print 'M',l,self.session.run(self.KL),self.session.run(self.like0)
                for i in iih:
                    self.session.run(self.updates_m[l],feed_dict={self.layers[l+1].k_:int32(i)})
                if(verbose): print 'P',l,self.session.run(self.KL),self.session.run(self.like0)
	    elif(mp_opt==2):
                for i in iih:
                    self.session.run(self.updates_m[l],feed_dict={self.layers[l+1].k_:int32(i)})
                    self.session.run(self.updates_p[l],feed_dict={self.layers[l+1].k_:int32(i)})
                if(verbose): print 'P',l,self.session.run(self.KL),self.session.run(self.like0)
	    elif(mp_opt==3):
                for i in iih:
                    self.session.run(self.updates_p[l],feed_dict={self.layers[l+1].k_:int32(i)})
                    self.session.run(self.updates_m[l],feed_dict={self.layers[l+1].k_:int32(i)})
                if(verbose): print 'P',l,self.session.run(self.KL),self.session.run(self.like0)
	else:
            self.session.run(self.updates_m[-1])
            self.session.run(self.update_last_p)
            loss.append(self.session.run(self.KL))
            if(verbose): print 'LAST P',l,loss[-1]
        L = self.session.run(self.KL)
        return L-GAIN
    def layer_M_step(self,lay,random=1,fineloss=1,verbose=0):
        loss = []
	l    = lay
	GAIN = self.session.run(self.like1)
	if(verbose): print 'BEFORE M ',str(l),GAIN
        self.session.run(self.updates_sigma[l])
        if(verbose): print 'SIGMA ',str(l),self.session.run(self.like1)
        self.session.run(self.updates_BV[l])
	if(verbose): print 'BV',l,self.session.run(self.like1)
        if(random==0): iih = self.layers[l+1].W_indices
        else: iih = self.layers[l+1].W_indices[permutation(len(self.layers[l+1].W_indices))]
        if(isinstance(self.layers[l+1],layers_.DenseLayer)):
            for kk in iih:
                self.session.run(self.updates_Wk[l],feed_dict={self.layers[l+1].k_:int32(kk)})
                if(fineloss): loss.append(self.session.run(self.like1))
	    if(verbose): print 'DW',l,self.session.run(self.like1)
        elif(isinstance(self.layers[l+1],layers_.ConvPoolLayer)):
            for kk in iih:
                self.session.run(self.updates_Wk[l],feed_dict={self.layers[l+1].r_:int32(kk[1]),
								self.layers[l+1].k_:int32(kk[0]),
								self.layers[l+1].i_:int32(kk[2]),
								self.layers[l+1].j_:int32(kk[3])})
                if(fineloss):loss.append(self.session.run(self.like1))
	    if(verbose): print 'CW',l,self.session.run(self.like1),
        else:# LAST LAYER
            self.session.run(self.updates_Wk[-1])
            if(fineloss): loss.append(self.session.run(self.like1))
	    if(verbose): print 'LW',l,self.session.run(self.like1),
        self.session.run(self.updates_pi[l])
	if(verbose): print 'PI',l,self.session.run(self.like1)
        if(fineloss): loss.append(self.session.run(self.like1)) 
        L = self.session.run(self.like1)
        return L-GAIN
    def E_step(self,rcoeff,fineloss=0,random=0,verbose=0,mp_opt=0):
        GAINS      = 0
	LAYER_GAIN = rcoeff+1
        while(LAYER_GAIN>rcoeff):
            LAYER_GAIN = 0
            for l in range(self.L-1):
                g_ = rcoeff+1
                while(g_>rcoeff):
                    g_=self.layer_E_step(l,random=random,fineloss=fineloss,verbose=verbose,mp_opt=mp_opt)
                    LAYER_GAIN+=g_
	    GAINS+= LAYER_GAIN
	return GAINS
    def M_step(self,rcoeff,fineloss=0,random=0,verbose=0):
        GAINS = 0
#        if(self.batch):
#            self.session.run(self.updates_S)
        for l in range(self.L-1):
            g_ = rcoeff+1
            while(g_>rcoeff):
                g_=self.layer_M_step(l,random=random,fineloss=fineloss,verbose=verbose)
                GAINS+=g_
	return GAINS
    def E_step2(self,rcoeff,fineloss=0,random=0,verbose=0,mp_opt=0):
        GAINS      = 0
        g = rcoeff+1
        while(g>rcoeff):
	    g=0
            for l in range(self.L-1):
                g_=self.layer_E_step(l,random=random,fineloss=fineloss,verbose=verbose,mp_opt=mp_opt)
                g+=g_
	    GAINS+=g
        return GAINS
    def set_alpha(self,alpha):
        self.session.run(self.initalpha,feed_dict={self.alpha:alpha})
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
    def get_evidence(self):
        return self.session.run(self.evidence)
    def reconstruct(self):
        return self.session.run(self.samplet)
    def predict(self):
        return squeeze(self.session.run(self.layers[-1].p_))



def train_layer_model(model,rcoeff_schedule,alpha_schedule,CPT,random=0,fineloss=1,return_time=0,verbose=0,per_layer=0,mp_opt=0,partial_E=False):
    """ mp_opt : { 0,1,2,3}, m then p, p then m, mpmpmp, pmpmpm"""
    cpt  = 0
    LIKE = [model.session.run(model.like0)]
    GAIN = 1
    print "INIT",model.session.run(model.KL),model.session.run(model.like0)
    cpt_ = 0
    while(cpt<CPT):# and GAIN>0):
        print 'Epoch...',cpt
        cpt    += 1
        indices = generate_batch_indices(model.N,model.bs)
        for i in range(len(indices)):
            print '  Batch...',i
            model.indices=indices[i]
            model.set_batch()
            print "\tBEFORE E",model.session.run(model.KL),model.session.run(model.like0)
            if(alpha_schedule.opt=='mean'):  model.set_alpha(float32(alpha_schedule.get(i)))
            else:                            model.set_alpha(float32(alpha_schedule.get(cpt_)))
	    if(per_layer):f = model.E_step(rcoeff=rcoeff_schedule.get(cpt),random=random,fineloss=fineloss,verbose=verbose,mp_opt=mp_opt)
	    else:  g = model.E_step2(rcoeff=rcoeff_schedule.get(cpt),random=random,fineloss=fineloss,verbose=verbose,mp_opt=mp_opt)
            print "\tAFTER E",model.session.run(model.KL),model.session.run(model.like0)
            model.save_batch()
            model.session.run(model.updates_S)
            if(partial_E):
                print "\tBEFORE M",model.session.run(model.like1)
                g = model.M_step(rcoeff=rcoeff_schedule.get(cpt),random=random,fineloss=fineloss,verbose=verbose)
                LIKE.append(model.session.run(model.like1))
                print "\tAFTER M",LIKE[-3:]
	        GAIN = g
	        print "\tgain",g
            cpt_+=1
        if(partial_E==0):
            print "\tBEFORE M",model.session.run(model.like1)
            g = model.M_step(rcoeff=rcoeff_schedule.get(cpt),random=random,fineloss=fineloss,verbose=verbose)
            LIKE.append(model.session.run(model.like1))
            print "\tAFTER M",LIKE[-3:]
            GAIN = g
	    print "\tgain",g
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
    elif(DATASET=='flippedMNIST'):
        batch_size = 50
        mnist         = fetch_mldata('MNIST original')
        x             = mnist.data.reshape(70000,1,28,28).astype('float32')
        y             = mnist.target.astype('int32')
	signs = randint(0,2,len(x))*2-1
        x_train,x_test,y_train,y_test = train_test_split(x*signs.reshape((-1,1,1,1)),y,test_size=10000,stratify=y)
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
	    y_train = y_train[y_train==k]*0
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
    if(1):#DATASET=='CIFAR'):
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











