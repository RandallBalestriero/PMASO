import tensorflow as tf
import time
from pylab import *
import layers as layers_
import itertools
from random import shuffle
import zipfile
from sklearn.decomposition import PCA
from sklearn.feature_extraction.image import extract_patches_2d
def mynormalize(x):
    return (x-x.min())/(x.max()-x.min())


def collect_sigmas(layers):
    a=[]
    cpt=0
    for l in layers[1:]:
        a.append(tf.reduce_sum(l.update_sigma(True)))
        cpt+=prod(l.input_shape[1:])
    S=tf.clip_by_value(tf.add_n(a)/cpt,0.000000001,10)
    a=[[]]
    for l in layers[1:]:
        a.append(tf.assign(l.sigmas2_,tf.ones_like(l.sigmas2_)*S))
    return a

def softmax(x,axis=-1):
    m=x.max(axis=axis,keepdims=True)
    return exp(x-m)/exp(x-m).sum(axis=axis,keepdims=True)



def PCA(X,K,opt=False):
    mu = X.mean(0)
    if(opt==False):
        p = permutation(X.shape[0])[:K]
        return X[p],mu#/norm(X[p].reshape((K,-1)),2,axis=1,keepdims=True)
    Xm  = X-mu.reshape((1,-1))
    v,w = eigh(dot(Xm.T,Xm))
    w   = w[:,-K:]
#    print v/v.sum()
    return w.T,mu#*mean(norm(X,2,axis=1)),mu

def extract_patches(M,S):
    N,_,_,C = M.shape
    PATCHES = []
    for n in xrange(N):
        PATCHES.append(stack([extract_patches_2d(M[n,:,:,c],(S,S)) for c in xrange(C)],2))
    return asarray(PATCHES).reshape((-1,S*S*C))


def plot_layer(model,l,n_,filters=1):
    if(l==1):
        figure()
        imshow(mynormalize(model.layers_[model.layers[l-1]].m[n_]),interpolation='nearest',aspect='auto')
    if(isinstance(model.layers[l],layers_.ConvLayer)):
        figure()
#        subplot(3,model.layers[l].K,1)
#        imshow(model.layers_[model.layers[l]].m[n_,:,:,0],interpolation='nearest',aspect='auto')
        for k in xrange(model.layers[l].K):
            subplot(2,model.layers[l].K,1+model.layers[l].K*0+k)
            imshow(model.layers_[model.layers[l]].m[n_,k],interpolation='nearest',aspect='auto')
            subplot(2,model.layers[l].K,1+model.layers[l].K+k)
            imshow(model.layers_[model.layers[l]].p[n_,k,:,:,0],interpolation='nearest',aspect='auto')
        suptitle('Convolutional input and m,p variables')
        if(filters):
            figure()
            W = model.session.run(model.layers[l].W_)
            for k in xrange(model.layers[l].K):
                for c in xrange(model.layers[l].C):
                    subplot(model.layers[l].C,model.layers[l].K,k*model.layers[l].C+c+1)
                    imshow(W[k,:,:,c],interpolation='nearest',aspect='auto',vmin=W.min(),vmax=W.max())
            suptitle('Convolutional Filters')
    elif(isinstance(model.layers[l],layers_.PoolLayer)):
        figure()
        for k in xrange(model.layers[l].K):
            subplot(2,model.layers[l].K,1+k)
            imshow(model.layers_[model.layers[l]].m[n_,k],interpolation='nearest',aspect='auto')
            subplot(2,model.layers[l].K,1+model.layers[l].K+k)
            imshow(model.layers_[model.layers[l]].p[n_,k,:,:,0],interpolation='nearest',aspect='auto')
        suptitle('Pooling Layer m and p')
    elif(isinstance(model.layers[l],layers_.DenseLayer)):
        if(filters):
            figure()
            W = model.session.run(model.layers[l].W)
            imshow(W[:,0],interpolation='nearest',aspect='auto')
            title('Filter of the fully connected layer')





#############################################################################################
#
#
#                       UPDATE and SAMPLE HELPER
#
#
#############################################################################################

class schedule:
    def __init__(self,lr_,opt):
	self.lr_      = lr_
	self.opt      = opt
        self.counter  = 0.
    def get(self):
        self.counter += 1 
        if(self.counter==1): return 1
	if(self.opt=='linear'):
	    return self.lr_
	elif(self.opt=='sqrt'):
	    return self.lr_/sqrt(self.counter)
        elif(self.opt=='exp'):
            return (self.counter)**(-self.lr_)
        elif(self.opt=='mean'):
            return float32(1.0/self.counter)
    def reset(self):
        self.counter = 0.


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


class latent_variable_placeholder:
    def __init__(self,m=0,p=0,v2=0):
        self.m=m
        self.v2=v2
        self.p=p
		
def init_latent_variables(X,X_V2,X_MASK,Y,Y_V2,Y_MASK,layers):
    #m p v
    N = X.shape[0]
    P = dict()
    for l in layers:
        if(isinstance(l,layers_.DenseLayer)):
            P[l]=latent_variable_placeholder((randn(N,l.K)*0.5/sqrt(l.K)).astype('float32'),softmax(randn(N,l.K,l.R)*0.5,-1).astype('float32'),ones((N,l.K),dtype='float32')/prod(l.input_shape[1:]))
            # placeholders
            P[l].m_placeholder  = tf.placeholder(tf.float32,shape=[l.bs,l.K])
            P[l].p_placeholder  = tf.placeholder(tf.float32,shape=[l.bs,l.K,l.R])
            P[l].v2_placeholder = tf.placeholder(tf.float32,shape=[l.bs,l.K])
            # assign operators
            P[l].m_assign_op    = tf.assign(l.m_,tf.transpose(P[l].m_placeholder))
            P[l].p_assign_op    = tf.assign(l.p_,tf.transpose(P[l].p_placeholder,[1,0,2]))
            P[l].v2_assign_op   = tf.assign(l.v2_,tf.transpose(P[l].v2_placeholder))
        elif(isinstance(l,layers_.ConvLayer)):
            P[l]=latent_variable_placeholder((randn(N,l.K,l.I,l.J)*1).astype('float32'),softmax(randn(N,l.K,l.I,l.J,l.R)*0.5,-1).astype('float32'),ones((N,l.K,l.I,l.J),dtype='float32')/(l.Ic*l.Jc*l.C))
            # placeholders
            P[l].m_placeholder  = tf.placeholder(tf.float32,shape=[l.bs,l.K,l.I,l.J])
            P[l].p_placeholder  = tf.placeholder(tf.float32,shape=[l.bs,l.K,l.I,l.J,l.R])
            P[l].v2_placeholder = tf.placeholder(tf.float32,shape=[l.bs,l.K,l.I,l.J])
            # assign operators
            P[l].m_assign_op    = tf.assign(l.m_,tf.transpose(P[l].m_placeholder,[1,2,3,0]))
            P[l].p_assign_op    = tf.assign(l.p_,tf.transpose(P[l].p_placeholder,[1,2,3,4,0]))
            P[l].v2_assign_op   = tf.assign(l.v2_,tf.transpose(P[l].v2_placeholder,[1,2,3,0]))
        elif(isinstance(l,layers_.PoolLayer)):
            P[l]=latent_variable_placeholder((randn(N,l.K,l.I,l.J)).astype('float32'),softmax(randn(N,l.K,l.I,l.J,l.R),-1).astype('float32'),ones((N,l.K,l.I,l.J),dtype='float32'))
            # placeholders
            P[l].m_placeholder  = tf.placeholder(tf.float32,shape=[l.bs,l.K,l.I,l.J])
            P[l].p_placeholder  = tf.placeholder(tf.float32,shape=[l.bs,l.K,l.I,l.J,l.R])
            P[l].v2_placeholder = tf.placeholder(tf.float32,shape=[l.bs,l.K,l.I,l.J])
            # assign operators
            P[l].m_assign_op    = tf.assign(l.m_,tf.transpose(P[l].m_placeholder,[1,2,3,0]))
            P[l].p_assign_op    = tf.assign(l.p_,tf.transpose(P[l].p_placeholder,[1,2,3,4,0]))
            P[l].v2_assign_op   = tf.assign(l.v2_,tf.transpose(P[l].v2_placeholder,[1,2,3,0]))
        elif(isinstance(l,layers_.InputLayer)):
            P[l]      = latent_variable_placeholder(X,0,X_V2)
            P[l].mask = X_MASK
            # placeholders
            P[l].m_placeholder    = tf.placeholder(tf.float32,shape=l.output_shape)
            P[l].v2_placeholder   = tf.placeholder(tf.float32,shape=l.output_shape)
            P[l].mask_placeholder = tf.placeholder(tf.float32,shape=l.output_shape)
            # assign operators
            P[l].m_assign_op      = tf.assign(l.m,P[l].m_placeholder)
            P[l].mask_assign_op   = tf.assign(l.mask,P[l].mask_placeholder)
            P[l].v2_assign_op     = tf.assign(l.v2,P[l].v2_placeholder)
        elif(isinstance(l,layers_.CategoricalLastLayer)):
            P[l]      = latent_variable_placeholder(0,Y,0)
            P[l].mask = Y_MASK
            # placeholders
            P[l].p_placeholder    = tf.placeholder(tf.float32,shape=[l.bs,l.R])
            P[l].mask_placeholder = tf.placeholder(tf.float32,shape=[l.bs])
            # assign operators
            P[l].p_assign_op      = tf.assign(l.p_,P[l].p_placeholder)
            P[l].mask_assign_op   = tf.assign(l.mask,P[l].mask_placeholder)
        elif(isinstance(l,layers_.ContinuousLastLayer)):
            P[l]      = latent_variable_placeholder(Y,0,ones((l.K,),dtype='float32')/prod(l.input_shape[1:]))
#            P[l].mask = Y_MASK
            # placeholders
            P[l].m_placeholder    = tf.placeholder(tf.float32,shape=[l.bs,l.K])
            P[l].v2_placeholder   = tf.placeholder(tf.float32,shape=[l.K])
#            P[l].mask_placeholder = tf.placeholder(tf.float32,shape=[l.bs])
            # assign operators
            P[l].m_assign_op      = tf.assign(l.m_,P[l].m_placeholder)
            P[l].v2_assign_op     = tf.assign(l.v2_,P[l].v2_placeholder)
#            P[l].mask_assign_op   = tf.assign(l.mask,P[l].mask_placeholder)
    return P



def likelihood(layers,batch):
    """ gather all the per layer likelihoods
    and add them together as derived in the paper"""
    like = []
    for l in layers:
        like.append(l.likelihood(batch))
    likes = []
    for l in xrange(len(layers)):
        likes.append(tf.reduce_sum(like[:l])+layers[l].likelihood(batch,True))
    return likes,tf.add_n(like)
	

def KL(layers):
    """gather the KL divergence by
    summing the per layers one as derived"""
    kl = []
    for l in layers:
        kl.append(l.KL())
    kls = []
#    for l in xrange(len(layers)):
#        kls.append(tf.reduce_sum(kl[:l])+layers[l].KL(True))
    return kls,tf.add_n(kl)



class model:
    def __init__(self,layers,X,X_mask=None,Y_mask=None,Y=None,batch=False):
        self.layers    = layers
        self.batch     = batch
        self.N         = len(X)
        self.bs        = layers[1].bs
        self.L         = len(layers)
        #
        if(Y is None):
            if(isinstance(layers[-1],layers_.ContinuousLastLayer)): 
                Y = randn(X.shape[0],layers[-1].K).astype('float32')
                Y_mask = ones(X.shape[0],dtype='float32')
            else:                                                   
                Y = softmax(randn(X.shape[0],layers[-1].R)*0,-1).astype('float32')
                Y_mask = ones(X.shape[0],dtype='float32')
        else:
            if(Y_mask is None): Y_mask = zeros(X.shape[0],dtype='float32')
            else:               Y_mask = Y_mask
        if(X_mask is None):     X_mask = zeros_like(X,dtype='float32')
        else:                   X_mask = X_mask
        self.layers_ = init_latent_variables(X,ones_like(X)*X_mask,X_mask,Y,ones_like(Y)*Y_mask.reshape((-1,1)),Y_mask,layers)
        #
        self.alpha   = tf.placeholder(tf.float32)
        for l in layers:
                self.layers_[l].alpha_assign_op=tf.assign(l.alpha,self.alpha)
        self.sigma     = tf.placeholder(tf.float32)
        # INIT SESSION
        session_config = tf.ConfigProto(allow_soft_placement=False,log_device_placement=True)
        session_config.gpu_options.allow_growth=True
        session        = tf.Session(config=session_config)
        self.session=session
        with tf.device('/device:GPU:0'):
#            self.BNBN = layers[1].BN()
            ### WEIGHTS UPDATES OP
            self.updates_S       = [l.update_S() for l in layers]
    	    self.updates_BV      = [l.update_BV() for l in layers]
            self.updates_m       = [l.update_m() for l in layers]
            self.updates_m_pre   = [l.update_m(0,True) for l in layers]
            self.updates_p       = [l.update_p() for l in layers]
	    self.updates_v2      = [l.update_v2() for l in layers]
            self.updates_v2_pre  = [l.update_v2(True) for l in layers]
            self.updates_sigma   = collect_sigmas(layers)#[l.update_sigma() for l in layers]
            self.updates_Wk      = [l.update_Wk() for l in layers]
            self.updates_pi      = [l.update_pi() for l in layers]
            self.updates_BN      = [l.update_BN() for l in layers]
            self.evidence        = sum([l.evidence() for l in layers])
        ## GATHER LOSSES
            self.KLs,self.KL       = KL(layers)
            self.like0s,self.like0 = likelihood(layers,0)
            self.like1s,self.like1 = likelihood(layers,1)
        # GATHER GRAD OPS
#            self.ops_KL = tf.train.AdamOptimizer(0.1).minimize(-self.KL,var_list=tf.get_collection('LATENT'))
#            self.ops_like1 = tf.train.AdamOptimizer(0.001).minimize(-self.like1,var_list=tf.get_collection('PARAMS'))
        # GATHER SAMPLING
        if(not isinstance(layers[-1],layers_.ContinuousLastLayer)):
            self.samplesclass    = [sampleclass(layers,k,sigma=self.sigma) for k in xrange(layers[-1].R)]
        self.samples         = sample(layers,sigma=self.sigma)
        self.samplet         = sampletrue(layers)
        init           = tf.global_variables_initializer()
        session.run(init)
    def set_alpha(self,alpha):
        for l in self.layers:
            self.session.run(self.layers_[l].alpha_assign_op,feed_dict={self.alpha:alpha})
    def set_batch(self,indices):
        for l in self.layers:
            if(isinstance(l,layers_.InputLayer)):
                self.session.run(self.layers_[l].m_assign_op,feed_dict={self.layers_[l].m_placeholder:self.layers_[l].m[indices]})
                self.session.run(self.layers_[l].v2_assign_op,feed_dict={self.layers_[l].v2_placeholder:self.layers_[l].v2[indices]})
                self.session.run(self.layers_[l].mask_assign_op,feed_dict={self.layers_[l].mask_placeholder:self.layers_[l].mask[indices]})
            elif(isinstance(l,layers_.ContinuousLastLayer)):
                self.session.run(self.layers_[l].m_assign_op,feed_dict={self.layers_[l].m_placeholder:self.layers_[l].m[indices]})
                self.session.run(self.layers_[l].v2_assign_op,feed_dict={self.layers_[l].v2_placeholder:self.layers_[l].v2})
#                self.session.run(self.layers_[l].mask_assign_op,feed_dict={self.layers_[l].mask_placeholder:self.layers_[l].mask[indices]})
            elif(isinstance(l,layers_.CategoricalLastLayer)):
                self.session.run(self.layers_[l].p_assign_op,feed_dict={self.layers_[l].p_placeholder:self.layers_[l].p[indices]})
                self.session.run(self.layers_[l].mask_assign_op,feed_dict={self.layers_[l].mask_placeholder:self.layers_[l].mask[indices]})
            else:
                self.session.run(self.layers_[l].m_assign_op,feed_dict={self.layers_[l].m_placeholder:self.layers_[l].m[indices]})
                self.session.run(self.layers_[l].p_assign_op,feed_dict={self.layers_[l].p_placeholder:self.layers_[l].p[indices]})
                self.session.run(self.layers_[l].v2_assign_op,feed_dict={self.layers_[l].v2_placeholder:self.layers_[l].v2[indices]})
    def save_batch(self,indices):
        for l in self.layers:
            if(isinstance(l,layers_.DenseLayer)):
                self.layers_[l].m[indices]  = transpose(self.session.run(l.m_))
                self.layers_[l].p[indices]  = transpose(self.session.run(l.p_),[1,0,2])
                self.layers_[l].v2[indices] = transpose(self.session.run(l.v2_))
            elif(isinstance(l,layers_.InputLayer)):
                self.layers_[l].m[indices]  = self.session.run(l.m)
                self.layers_[l].v2[indices] = self.session.run(l.v2)
            elif(isinstance(l,layers_.CategoricalLastLayer)):
                self.layers_[l].p[indices]  = self.session.run(l.p_)
            elif(isinstance(l,layers_.ContinuousLastLayer)):
                self.layers_[l].m[indices]  = self.session.run(l.m_)
                self.layers_[l].v2          = self.session.run(l.v2_)
            else:
                self.layers_[l].m[indices]  = transpose(self.session.run(l.m_),[3,0,1,2])
                self.layers_[l].p[indices]  = transpose(self.session.run(l.p_),[4,0,1,2,3])
                self.layers_[l].v2[indices] = transpose(self.session.run(l.v2_),[3,0,1,2])
    def get_params(self):
        params = []
        for l in self.layers[1:]:
            params.append([self.session.run(l.W),self.session.run(l.sigmas2_),self.session.run(l.pi),self.session.run(l.b_),self.session.run(l.V_)])
        return params
    def layer_E_step(self,l,random=0,fineloss=0,verbose=2,mp_opt=0,pretraining=False):
        if(pretraining): updates_m = self.updates_m_pre;updates_v2 = self.updates_v2_pre
        else: updates_m = self.updates_m;updates_v2 = self.updates_v2
        GAIN = self.session.run(self.KL)
        if(verbose): print 'BEFORE',l,GAIN,self.session.run(self.like0)
        self.session.run(updates_v2[l])
        if(verbose): print 'V2',l,self.session.run(self.KL),self.session.run(self.like0)
        # FIRST LAYER CASE
	if(l==0): 
            self.session.run(updates_m[l])
            L = self.session.run(self.KL)
            if(verbose): print 'FIRST',l,L
            return L-GAIN
        #LAST LAYER CASE
        if(l==(self.L-1)):
            self.session.run(updates_m[l])
            if(verbose): print 'LAST M',self.session.run(self.KL)
            self.session.run(self.updates_p[l])
            if(verbose): print 'LAST P',self.session.run(self.KL)
            L = self.session.run(self.KL)
            return L-GAIN
        if(isinstance(self.layers[l],layers_.PoolLayer)):
            if(mp_opt==0):
                self.session.run(updates_m[l])
                if(verbose): print 'M',l,self.session.run(self.KL),self.session.run(self.like0)
                self.session.run(self.updates_p[l])
                if(verbose): print 'P',l,self.session.run(self.KL),self.session.run(self.like0)
            else:
                self.session.run(self.updates_p[l])
                if(verbose): print 'P',l,self.session.run(self.KL),self.session.run(self.like0)
                self.session.run(updates_m[l])
                if(verbose): print 'M',l,self.session.run(self.KL),self.session.run(self.like0)
            return self.session.run(self.KL)-GAIN
        if(random==0): iih = self.layers[l].p_indices
        else:          iih = self.layers[l].p_indices[permutation(len(self.layers[l].p_indices))]
	if(isinstance(self.layers[l],layers_.ConvLayer)):
            if(random==0): miih = self.layers[l].m_indices
            else:          miih = self.layers[l].m_indices[permutation(len(self.layers[l].m_indices))]
            if(mp_opt==0):
                for i in miih:
                    self.session.run(updates_m[l],feed_dict={self.layers[l].i_:int32(i[0]),
                                                                        self.layers[l].j_:int32(i[1])})
                    if(verbose==2): print 'M',l,self.session.run(self.KL),self.session.run(self.like0)
                for i in iih:
                    self.session.run(self.updates_p[l],feed_dict={self.layers[l].i_:int32(i[1]),
                                                                        self.layers[l].j_:int32(i[2]),
                                                                        self.layers[l].k_:int32(i[0])})
                    if(verbose==2): print 'P',l,self.session.run(self.KL),self.session.run(self.like0)
            else:
                for i in iih:
                    self.session.run(self.updates_p[l],feed_dict={self.layers[l].i_:int32(i[1]),
                                                                        self.layers[l].j_:int32(i[2]),
                                                                        self.layers[l].k_:int32(i[0])})
                    if(verbose==2): print 'P',l,self.session.run(self.KL),self.session.run(self.like0)
                for i in miih:
                    self.session.run(updates_m[l],feed_dict={self.layers[l].i_:int32(i[0]),
                                                                        self.layers[l].j_:int32(i[1])})
                    if(verbose==2): print 'M',l,self.session.run(self.KL),self.session.run(self.like0) 
            if(verbose==1): print 'MP',l,self.session.run(self.KL),self.session.run(self.like0)
	elif(isinstance(self.layers[l],layers_.DenseLayer)):
            if(mp_opt==0):
                self.session.run(updates_m[l])
                if(verbose): print 'M',l,self.session.run(self.KL),self.session.run(self.like0)
                for i in iih:
                    self.session.run(self.updates_p[l],feed_dict={self.layers[l].k_:int32(i)})
                if(verbose): print 'P',l,self.session.run(self.KL),self.session.run(self.like0)
            else:
                for i in iih:
                    self.session.run(self.updates_p[l],feed_dict={self.layers[l].k_:int32(i)})
                if(verbose): print 'P',l,self.session.run(self.KL),self.session.run(self.like0)
                self.session.run(updates_m[l])
                if(verbose): print 'M',l,self.session.run(self.KL),self.session.run(self.like0)
        L = self.session.run(self.KL)
        return L-GAIN
    def layer_M_step(self,l,random=0,fineloss=0,verbose=2,pretraining=False):
        #FIRST LAYER
        if(l==0):
            return 0
	GAIN = self.session.run(self.like1)
        self.session.run(self.updates_pi[l])
        if(verbose): print 'PI ',l,self.session.run(self.like1)
        # CATEGORICAL LAST LAYER
        if(isinstance(self.layers[l],layers_.CategoricalLastLayer)):
            self.session.run(self.updates_Wk[l])
            if(verbose): print 'LW',l,self.session.run(self.like1)
            self.session.run(self.updates_BV[l])
            if(verbose): print 'BV',l,self.session.run(self.like1)
            if(pretraining==False):
                self.session.run(self.updates_sigma)#)##############################[l])
                if(verbose): print 'SIGMA',l,self.session.run(self.like1)
            return self.session.run(self.like1)-GAIN
        # POOL LAYER
        if(isinstance(self.layers[l],layers_.PoolLayer)):
            if(pretraining==False):
                self.session.run(self.updates_sigma)###################################[l]
                if(verbose): print 'SIGMA',l,self.session.run(self.like1)
            return self.session.run(self.like1)-GAIN
        if(random==0): iih = self.layers[l].W_indices
        else:          iih = self.layers[l].W_indices[permutation(len(self.layers[l].W_indices))]
        if(isinstance(self.layers[l],layers_.DenseLayer) or isinstance(self.layers[l],layers_.ContinuousLastLayer)):
            for kk in iih:
                self.session.run(self.updates_Wk[l],feed_dict={self.layers[l].k_:int32(kk)})
                if(verbose==2): print 'W',l,self.session.run(self.like1)
	    if(verbose): print 'W',l,self.session.run(self.like1)
        elif(isinstance(self.layers[l],layers_.ConvLayer)):
            for kk in iih:
                self.session.run(self.updates_Wk[l],feed_dict={	self.layers[l].k_:int32(kk[0]),self.layers[l].r_:int32(kk[1])})#,
#								self.layers[l].i_:int32(kk[1]),
#								self.layers[l].j_:int32(kk[2])})
                if(verbose==2): print 'CW',l,self.session.run(self.like1)
	    if(verbose==1): print 'CW',l,self.session.run(self.like1)
        self.session.run(self.updates_BV[l])
        if(verbose): print 'BV',l,self.session.run(self.like1)
        if(pretraining==False):
            self.session.run(self.updates_sigma)########################################[l])
            if(verbose): print 'SIGMA',l,self.session.run(self.like1)
        L = self.session.run(self.like1)
        return L-GAIN
    def E_step(self,rcoeff,fineloss=0,random=0,verbose=0,mp_opt=0,per_layer=True):
        GAINS      = 0
        if(per_layer):
            LAYER_GAIN = rcoeff+1
            while(LAYER_GAIN>rcoeff):
                LAYER_GAIN = self.session.run(self.KL)
                for l in xrange(self.L):
                    g_ = rcoeff+1
                    self.session.run(self.updates_BN[l])
                    while(g_>rcoeff):
                        g_=self.layer_E_step(l,random=random,fineloss=fineloss,verbose=verbose,mp_opt=mp_opt)
                LAYER_GAIN = self.session.run(self.KL)-LAYER_GAIN
	        GAINS+= LAYER_GAIN
        else:
            g = rcoeff+1
            while(g>rcoeff):
                g=0
                for l in permutation(self.L):#xrange(self.L):
                    g_=self.layer_E_step(l,random=random,fineloss=fineloss,verbose=verbose,mp_opt=mp_opt)
                    g+=g_
#                print g
                GAINS+=g
	return GAINS
    def M_step(self,rcoeff,fineloss=0,random=0,verbose=0):
        GAINS = 0
        for l in xrange(self.L):
            g_ = rcoeff+1
            while(g_>rcoeff):
                g_=self.layer_M_step(l,random=random,fineloss=fineloss,verbose=verbose)
                GAINS+=g_
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
    def get_evidence(self):
        return self.session.run(self.evidence)
    def reconstruct(self):
        return self.session.run(self.samplet)
    def predict(self):
        return squeeze(self.session.run(self.layers[-1].p_))





def pretrain(model,OPT=False):
    for LAYERS in xrange(1,model.L):
        indices = generate_batch_indices(model.N,model.bs)
        if(len(model.layers[LAYERS].output_shape)==2):
            nn   = shape(model.layers_[model.layers[LAYERS-1]].m)
            what,b = PCA(model.layers_[model.layers[LAYERS-1]].m.reshape((nn[0],-1)),model.layers[LAYERS].K,OPT)
            model.session.run(model.layers[LAYERS].init_W(what,reshape(b,[-1])))
        elif(isinstance(model.layers[LAYERS],layers_.ConvLayer)):
            P = extract_patches(model.layers_[model.layers[LAYERS-1]].m,model.layers[LAYERS].Ic)
            what,_ = PCA(P,model.layers[LAYERS].K,OPT)
            what = what.reshape((model.layers[LAYERS].K,model.layers[LAYERS].Ic,model.layers[LAYERS].Ic,model.layers[LAYERS].C))
            model.session.run(model.layers[LAYERS].init_W(what))
        for kkk in [0]:#xrange(5):
#            model.set_batch(indices[0])
#            model.set_alpha(1)
#            model.session.run(model.updates_S)
#            model.session.run(model.updates_sigma[LAYERS])
            for i in range(len(indices)):
                model.set_batch(indices[i])
                for jjj in xrange(10):
                    g=model.layer_E_step(LAYERS,pretraining=True,verbose=0)
#                m = model.session.run(model.layers[LAYERS].m)[:,:10]
#                p = model.session.run(model.layers[LAYERS].p)[:,:10,0]
#                print "m",m.min(0),m.max(0),m.std(0)
#                print "p",p.min(0),p.max(0),p.std(0)
                model.save_batch(indices[i])
#            model.layers_[model.layers[LAYERS]].m-=model.layers_[model.layers[LAYERS]].m.mean(0,keepdims=True)
#            model.layers_[model.layers[LAYERS]].m/=model.layers_[model.layers[LAYERS]].m.std(0,keepdims=True)
#            for i in range(len(indices)):
#                model.set_batch(indices[i])
#                model.set_alpha(1.0/(1+i))
#                model.session.run(model.updates_S)
#            for iiii in xrange(10):
#                print LAYERS,iiii
#                for kk in model.layers[LAYERS].W_indices[permutation(len(model.layers[LAYERS].W_indices))]:
#                    model.session.run(model.updates_Wk[LAYERS],feed_dict={model.layers[LAYERS].k_:int32(kk)})
#                    W = model.session.run(model.layers[LAYERS].W_)[:,0,:]
#                    print dot(W,W.T)
#            model.save_batch(indices[i])




def train_layer_model(model,rcoeff_schedule,alpha_schedule,CPT,random=0,fineloss=1,return_time=0,verbose=0,per_layer=0,mp_opt=0,partial_E=False,G=False,PLOT=False):
    """ mp_opt : { 0,1,2,3}, m then p, p then m, mpmpmp, pmpmpm"""
    LIKE = []

    for epoch in xrange(CPT):# and GAIN>0):
        if(alpha_schedule.opt=='mean'): alpha_schedule.reset()
        indices = generate_batch_indices(model.N,model.bs)
        for batch in range(len(indices)):
            print 'Epoch...',epoch,'/',CPT,'  Batch...',batch,'/',len(indices)
            model.set_batch(indices[batch])
            if(epoch==0 and batch==0):# and False):
                model.set_alpha(float32(1))
                model.session.run(model.updates_S)
                model.session.run(model.updates_sigma)
                print "INIT SIGMA",
                print [(model.session.run(model.layers[l].sigmas2_).min(),model.session.run(model.layers[l].sigmas2_).max()) for l in xrange(1,model.L)]
            t=time.time()
	    g=model.E_step(rcoeff=rcoeff_schedule.get(),random=random,fineloss=fineloss,verbose=verbose,mp_opt=mp_opt,per_layer=per_layer)
            print "\tAFTER E",model.session.run(model.KL),model.session.run(model.like0),'   gain',g,'  time:',time.time()-t
            model.save_batch(indices[batch])
            if(alpha_schedule.opt=='mean'):  model.set_alpha(alpha_schedule.get())
            else:                            model.set_alpha(alpha_schedule.get())
            model.session.run(model.updates_S)
            print "AFTER UPDATE",model.session.run(model.like0),model.session.run(model.like1)
            print "Sp",model.session.run(model.layers[-1].p_).max()
            print "BIN",bincount(argmax(model.session.run(model.layers[-1].p_),1))
            print "m",model.session.run(model.layers[-2].m_).min(),model.session.run(model.layers[-2].m_).max()
            print "p",model.session.run(model.layers[-2].p)[:,:,0].min(),model.session.run(model.layers[-2].p)[:,:,0].max()
#            print "m",model.session.run(model.layers[-3].m_).min(),model.session.run(model.layers[-3].m_).max()
#            print "p",model.session.run(model.layers[-3].p)[:,:,0].min(),model.session.run(model.layers[-3].p)[:,:,0].max()
            if(partial_E):
                if(PLOT):
                    for ll in xrange(model.L-1): plot_layer(model,ll,10,1)
                    show()
                t=time.time()
                g = model.M_step(rcoeff=rcoeff_schedule.get(),random=random,fineloss=fineloss,verbose=verbose)
                LIKE.append(model.session.run(model.like1))
                print "\tAFTER M",model.session.run(model.KL),model.session.run(model.like0),LIKE[-3:],'   gain',g,'   time',time.time()-t
                print [(model.session.run(model.layers[l].sigmas2_).min(),model.session.run(model.layers[l].sigmas2_).max()) for l in xrange(1,model.L)]
        if(partial_E==0):
            if(PLOT):
                for ll in xrange(model.L-1): plot_layer(model,ll,10,1)
                show()
            t=time.time()
            g = model.M_step(rcoeff=rcoeff_schedule.get(),random=random,fineloss=fineloss,verbose=verbose)
            LIKE.append(model.session.run(model.like1))
            print "\tAFTER M",model.session.run(model.KL),model.session.run(model.like0),LIKE[-3:],'   gain',g,'   time',time.time()-t
            print [(model.session.run(model.layers[l].sigmas2_).min(),model.session.run(model.layers[l].sigmas2_).max()) for l in xrange(1,model.L)]
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



def load_data(DATASET,k=-1,unlabeled=False):
    if(DATASET=='MNIST'):
        mnist         = fetch_mldata('MNIST original')
        x             = mnist.data.reshape(70000,1,28,28).astype('float32')
        y             = mnist.target.astype('int32')
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=10000,stratify=y)
        Y_mask = zeros(len(x_train))
    elif(DATASET=='STL10'):
        x_train = read_all_images('../../DATASET/STL10/train_X.bin')
        y_train = read_labels('../../DATASET/STL10/train_y.bin')
        x_test  = read_all_images('../../DATASET/STL10/train_X.bin')
        y_test  = read_labels('../../DATASET/STL10/test_y.bin')
        if(unlabeled):
            x_unsup = read_all_images('../../DATASET/SST10/unlabeled.bin')
            x_train = concatenate([x_train,x_unsup],0)
            y_train = concatenate([my_onehot(y_train,10),ones((x_unsup.shape[0],10))/10])
            Y_mask  = concatenate([zeros(len(y_train)),ones(x_unsup.shape[0])])
        else: Y_mask = zeros(len(y_train))
    elif(DATASET=='FASHION'):
        from numpy import loadtxt
        ff = loadtxt('../../DATASET/fashion-mnist_train.csv',delimiter=',',skiprows=1)
        x_train = ff[:,1:].reshape((-1,1,28,28)).astype('float32')
        y_train = ff[:,0].astype('int32')
        ff = loadtxt('../../DATASET/fashion-mnist_test.csv',delimiter=',',skiprows=1)
        x_test = ff[:,1:].reshape((-1,1,28,28)).astype('float32')
        y_test = ff[:,0].astype('int32')
        Y_mask = zeros(len(x_train))
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
        Y_mask = zeros(len(x_train))
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
        Y_mask = zeros(len(x_train))
    ptr = permutation(len(x_train))
    pte = permutation(len(x_test))
    if(DATASET=='CIFAR' or DATASET=='STL10'):
        x_train          -= x_train.mean((1,2,3),keepdims=True)
        x_test           -= x_test.mean((1,2,3),keepdims=True)
    else:
        x_train          -= x_train.min((1,2,3),keepdims=True)
        x_test           -= x_test.min((1,2,3),keepdims=True)
    x_train          /= abs(x_train).max((1,2,3),keepdims=True)
    x_test           /= abs(x_test).max((1,2,3),keepdims=True)
    x_train           = x_train.astype('float32')
    x_test            = x_test.astype('float32')
    y_train           = array(y_train).astype('int32')
    y_test            = array(y_test).astype('int32')
    return x_train[ptr],y_train[ptr],x_test[pte],y_test[pte],Y_mask









#### SST 10
def read_labels(path_to_labels):
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_data):
    with open(path_to_data, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))
        return images.astype('float32')









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











