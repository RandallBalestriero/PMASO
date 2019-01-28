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
    XX = (x-x.min())/(x.max()-x.min())
    if(len(XX.shape)==2):
        return XX
    elif(XX.shape[2]==1):
        return XX[:,:,0]
    return XX


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


def sigmoid(x): return 1/(1+exp(-x))


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
    if(l==0):
        figure()
        R = model.reconstruct()
        subplot(141)
        imshow(mynormalize(model.layers_[model.layers[l]].m[n_]),interpolation='nearest',aspect='auto')
        subplot(142)
        imshow(mynormalize(R[n_]),interpolation='nearest',aspect='auto')
        subplot(143)
        imshow(mynormalize(R[0]),interpolation='nearest',aspect='auto')
        subplot(144)
        imshow(mynormalize(R[1]),interpolation='nearest',aspect='auto')
    if(isinstance(model.layers[l],layers_.ConvLayer) or isinstance(model.layers[l],layers_.AltConvLayer)):
        figure()
#        subplot(3,model.layers[l].K,1)
#        imshow(model.l4ayers_[model.layers[l]].m[n_,:,:,0],interpolation='nearest',aspect='auto')
        W = model.session.run(model.layers[l].W_)
        for k in xrange(model.layers[l].K):
            subplot(2+model.layers[l].C,model.layers[l].K,1+k)
            imshow(model.layers_[model.layers[l]].m[n_,k],interpolation='nearest',aspect='auto',vmin=model.layers_[model.layers[l]].m[n_].min(),vmax = model.layers_[model.layers[l]].m[n_].max())
            subplot(2+model.layers[l].C,model.layers[l].K,1+model.layers[l].K+k)
            imshow(model.layers_[model.layers[l]].p[n_,k],interpolation='nearest',aspect='auto',vmin=model.layers_[model.layers[l]].p[n_].min(),vmax = model.layers_[model.layers[l]].p[n_].max())
            for c in xrange(model.layers[l].C):
                subplot(2+model.layers[l].C,model.layers[l].K,(c+2)*model.layers[l].K+k+1)
                imshow(W[k,:,:,c],interpolation='nearest',aspect='auto',vmin=W.min(),vmax=W.max())
            suptitle('Convolutional input and m,p and filter variables')
    elif(isinstance(model.layers[l],layers_.PoolLayer)):
        figure()
        for k in xrange(model.layers[l].K):
            subplot(2,model.layers[l].K,1+k)
            imshow(model.layers_[model.layers[l]].m[n_,k],interpolation='nearest',aspect='auto',vmin=model.layers_[model.layers[l]].m[n_].min(),vmax = model.layers_[model.layers[l]].m[n_].max())
            subplot(2,model.layers[l].K,1+model.layers[l].K+k)
            imshow(model.layers_[model.layers[l]].p[n_,k,:,:,0,0],interpolation='nearest',aspect='auto',vmin=model.layers_[model.layers[l]].p[n_].min(),vmax = model.layers_[model.layers[l]].p[n_].max())
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




def collect_KL_likelihood(layers):
    """ gather all the per layer likelihoods
    and add them together as derived in the paper"""
    like_E = []
    like_M = []
    kl     = []
    for l in layers:
        like_E.append(l.likelihood(E_step=True))
	like_M.append(l.likelihood(E_step=False))
	kl.append(l.KL())
    return tf.add_n(like_E),tf.add_n(like_M),tf.add_n(kl)
	


class model:
    def __init__(self,layers,sigma='unconstrained'):
        self.layers    = layers
        self.L         = len(layers)
        # INIT SESSION
        session_config = tf.ConfigProto(allow_soft_placement=False,log_device_placement=True)
        session_config.gpu_options.allow_growth=True
        session        = tf.Session(config=session_config)
        self.session=session
	############## GATHER UPDATE OPS AND STATISTICS
        self.like_W,self.like_M,self.KL = collect_KL_likelihood(layers)
	self.meta_alpha      = tf.Variable(tf.ones(1))
	self.update_alpha    = tf.group([session.assign(l.alpha,self.meta_alpha) for l in layers[1:]])
        # STATISTICS UPDATES
        self.updates_S       = [l.update_S() for l in layers]
	# THETA PARAMETERS UPDATES
    	self.updates_b       = [l.update_BV() for l in layers]
        self.updates_Wk      = [l.update_Wk() for l in layers]
        self.updates_pi      = [l.update_pi() for l in layers]
        if(sigma=='universal'): self.updates_sigma = [collect_sigmas(layers)]*self.L
        else:                   self.updates_sigma = [l.update_sigma() for l in layers]
	# THETAQ PARAMETERS UPDATES
        self.updates_m       = [l.update_m() for l in layers]
        self.updates_p       = [l.update_p() for l in layers]
	self.updates_v2      = [l.update_v2() for l in layers]
        self.evidence        = sum([l.evidence() for l in layers])
	############## GATHER  SAMPLES
        if(not isinstance(layers[-1],layers_.ContinuousLastLayer)):
            self.samplesclass= [sampleclass(layers,k,sigma=self.sigma) for k in xrange(layers[-1].R)]
        self.samples         = sample(layers,sigma=self.sigma)
        self.reconstruction  = layers[1].backward()
	self.reconstructed_input = layers[0].m_data
        init                 = tf.global_variables_initializer()
        session.run(init)
    def set_alpha(self,alpha):
        self.session.run(self.update_alpha,feed_dict={self.meta_alpha:float32(alpha)})
    def get_params(self):
	return []
        params = []
        for l in self.layers[1:]:
            params.append([self.session.run(l.W),self.session.run(l.sigmas2),self.session.run(l.pi),self.session.run(l.b_),self.session.run(l.V_)])
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
                TT = time.time()
                for i in miih:
                    self.session.run(updates_m[l],feed_dict={self.layers[l].i_:int32(i[0]),self.layers[l].j_:int32(i[1])})
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
	elif(isinstance(self.layers[l],layers_.AltConvLayer)):
            if(mp_opt==0):
                self.session.run(updates_m[l])
                if(verbose==2): print 'M',l,self.session.run(self.KL),self.session.run(self.like0)
                for i in iih:
                    self.session.run(self.updates_p[l],feed_dict={self.layers[l].k_:int32(i[0])})
                    if(verbose==2): print 'P',l,self.session.run(self.KL),self.session.run(self.like0)
            else:
                for i in iih:
                    self.session.run(self.updates_p[l],feed_dict={self.layers[l].k_:int32(i[0])})
                    if(verbose==2): print 'P',l,self.session.run(self.KL),self.session.run(self.like0)
                self.session.run(updates_m[l])
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
        if(verbose): print "INIT M ",GAIN
        self.session.run(self.updates_pi[l])
        if(verbose): print 'PI ',l,self.session.run(self.like1)
        # CATEGORICAL LAST LAYER
        if(isinstance(self.layers[l],layers_.CategoricalLastLayer)):
            self.session.run(self.updates_Wk[l])
            if(verbose): print 'LW',l,self.session.run(self.like1)
            self.session.run(self.updates_BV[l])
            if(verbose): print 'BV',l,self.session.run(self.like1)
            if(pretraining==False):
                self.session.run(self.updates_sigma[l])#)##############################[l])
                if(verbose): print 'SIGMA',l,self.session.run(self.like1)
            return self.session.run(self.like1)-GAIN
        # POOL LAYER
        if(isinstance(self.layers[l],layers_.PoolLayer)):
            if(pretraining==False):
                self.session.run(self.updates_sigma[l])###################################[l]
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
                self.session.run(self.updates_Wk[l],feed_dict={	self.layers[l].k_:int32(kk[0]),self.layers[l].i_:int32(kk[1]),self.layers[l].j_:int32(kk[2])})
                if(verbose==2): print 'CW',l,self.session.run(self.like1)
	    if(verbose==1): print 'CW',l,self.session.run(self.like1)
        elif(isinstance(self.layers[l],layers_.AltConvLayer)):
            for kk in iih:
                self.session.run(self.updates_Wk[l],feed_dict={ self.layers[l].k_:int32(kk[0])})
                if(verbose==2): print 'CW',l,self.session.run(self.like1)
                if(verbose==1): print 'CW',l,self.session.run(self.like1)
        self.session.run(self.updates_BV[l])
        if(verbose): print 'BV',l,self.session.run(self.like1)
        if(pretraining==False):
            self.session.run(self.updates_sigma[l])########################################[l])
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
#                    self.session.run(self.updates_BN[l])
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
    def train(model,eps,alpha_schedule,EPOCH,random=0,fineloss=1,return_time=0,verbose=0,per_layer=0,mp_opt=0,partial_E=False,G=False,PLOT=False):
        LIKE = []
        for epoch in xrange(EPOCH):
            alpha_schedule.reset_epoch()
            indices = generate_batch_indices(self.layers[0].N,self.layers[0].input_shape[0])
            for batch in range(len(indices)):
                print 'Epoch...',epoch,'/',CPT,'  Batch...',batch,'/',len(indices)
		# we set the batch THETAQ variables for the current indices
                [l.set_batch(indices[batch]) for l in self.layers]
		# for the first batch of the first epoch  we do a hard initialization
		# of the statistics that have been init at 0, based only on the current batch
                if(epoch==0 and batch==0):
                    model.set_alpha(float32(1))
                    model.session.run(model.updates_S)
		#### E STEP
            	t=time.time()
            	g=self.E_step(rcoeff=eps,random=random,fineloss=fineloss,verbose=verbose,mp_opt=mp_opt,per_layer=per_layer)
            	print "\tAFTER E",model.session.run(model.KL),model.session.run(model.like0),'   gain',g,'  time:',time.time()-t
            	model.save_batch(indices[batch])
                model.session.run(model.updates_S)
                if(partial_E):
		    # PARTIAL M STEP
                    t=time.time()
                    g = self.M_step(eps=alpha_schedule.get(),random=random,fineloss=fineloss,verbose=verbose)
                    LIKE.append(model.session.run(model.like1))
            if(partial_E is False):
	        # GLOBAL M STEP
                t=time.time()
                g = model.M_step(rcoeff=rcoeff_schedule.get(),random=random,fineloss=fineloss,verbose=verbose)
                LIKE.append(model.session.run(model.like1))
                print "\tAFTER M",model.session.run(model.KL),model.session.run(model.like0),LIKE[-3:],'   gain',g,'   time',time.time()-t
    	return LIKE






def pretrain(model,OPT=False):
    for LAYERS in xrange(1,model.L):
        if(LAYERS<=model.L-1):
            if(isinstance(model.layers[LAYERS+1],layers_.PoolLayer)): continue
        indices = generate_batch_indices(model.N,model.bs)
#        if(len(model.layers[LAYERS].output_shape)==2):
#            nn   = shape(model.layers_[model.layers[LAYERS-1]].m)
#            what,b = PCA(model.layers_[model.layers[LAYERS-1]].m.reshape((nn[0],-1)),model.layers[LAYERS].K,OPT)
#            model.session.run(model.layers[LAYERS].init_W(what,reshape(b,[-1])))
#        elif(isinstance(model.layers[LAYERS],layers_.ConvLayer)):
#            P = extract_patches(model.layers_[model.layers[LAYERS-1]].m,model.layers[LAYERS].Ic)
#            what,_ = PCA(P,model.layers[LAYERS].K*2,OPT)
#            what = what.reshape((model.layers[LAYERS].K,2,model.layers[LAYERS].Ic,model.layers[LAYERS].Ic,model.layers[LAYERS].C))
#            model.session.run(model.layers[LAYERS].init_W(what))
        print 'PRETRAINING LAYER ->',LAYERS
        for e in xrange(10):
            for i in range(len(indices)):
                model.set_batch(indices[i])
                for jjj in xrange(10):
                    for l in xrange(LAYERS+1):
                        g=model.layer_E_step(l,pretraining=(l==LAYERS),verbose=0,random=True)
#                        print 'E',g
                model.save_batch(indices[i])
                if(LAYERS<model.L-1):
                    print "m",[(model.session.run(model.layers[l].m_).min(),model.session.run(model.layers[l].m_).max()) for l in xrange(1,LAYERS+1)]
                print "p",[(model.session.run(model.layers[l].p).reshape((-1,2))[:,0].min(),model.session.run(model.layers[l].p).reshape((-1,2))[:,0].max()) for l in xrange(1,LAYERS+1)]
                model.set_alpha(float32(1.0/(i+1)))
                model.session.run(model.updates_S)
            if((e+1)%3 ==0):
                for l in xrange(LAYERS+1):
                    plot_layer(model,l,0)
                show()
            for jjj in xrange(5):
                for l in xrange(LAYERS+1):
                    g=model.layer_M_step(l,verbose=0,random=True)
                    print 'M',g
            print 'SIGMA',[(model.session.run(model.layers[l].sigmas2_).min(),model.session.run(model.layers[l].sigmas2_).max()) for l in xrange(1,LAYERS+1)]


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











