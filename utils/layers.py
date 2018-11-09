import tensorflow as tf
from pylab import *
import utils
import itertools
from math import pi as PI_CONST


eps = float32(0.000000000001)

class BN:
    def __init__(self,center,scale):
	self.center = center
	self.scale  = scale



def bn(x,axis):
    m,v = tf.nn.moments(x,axes=axis,keep_dims=True)
    v_ = tf.sqrt(tf.maximum(v,0.0001))
    return (x-m)/v_,m,v_



def mynorm(W,axis=None,keepdims=False):
    return tf.reduce_sum(tf.square(W),axis=axis,keepdims=keepdims)

 
def mysoftmax(W,axis=-1,coeff=0.00):
    input = W-tf.reduce_max(W,axis=axis,keepdims=True)
    delta = tf.exp(input)/tf.reduce_sum(tf.exp(input),axis=axis,keepdims=True)
    if(coeff==0):
	return delta
    deltap = delta+coeff
    return deltap/tf.reduce_sum(deltap,axis=axis,keepdims=True)


def mynormalize(W,axis=-1):
    return W/tf.reduce_sum(W,axis=axis,keepdims=True)


def myexpand(x,axis):
    nx = tf.expand_dims(x,axis[0])
    if(len(axis)==1):
        return nx
    else:
        return myexpand(nx,axis[1:])


def mysumpool(x,p):
    return tf.nn.avg_pool(x,[1,p[0],p[1],1],[1,p[0],p[1],1],'VALID')*p[0]*p[1]




class Layer:
    def BN(self):
        return []
    def update_b(self):
        return []
    def update_Wk(self):
        return []
    def update_pi(self):
        return []
    def update_sigma(self):
        return []
    def update_BV(self):
        return []
    def update_v2(self,pretraining=1):
        return []
    def update_m(self,pretraining=1,b=0):
        return []
    def update_rho(self,pretraining=1):
        return []
    def update_p(self,pretraining=1):
        return []
    def update_S(self):
        return []

#########################################################################################################################
#
#
#                                       DENSE/CONV/POOL LAYERS
#
#
#########################################################################################################################



class DenseLayer(Layer):
    def __init__(self,input_layer,K,R,sparsity_prior = 0,leakiness=None,sigma='local',p_drop=0,residual=0,alpha=0.95,update_b=True,init_W=tf.orthogonal_initializer):
        self.update_b       = update_b
        self.leakiness      = leakiness
        self.alpha          = tf.Variable(alpha,trainable=False)
	self.sigma_opt      = sigma
	self.residual       = residual
        self.sparsity_prior = sparsity_prior
        self.input_layer    = input_layer
	self.p_drop         = float32(p_drop)
        input_layer.next_layer = self
        self.input_shape       = input_layer.output_shape
        self.bs,self.R,self.K  = self.input_shape[0],R,K
        self.D_in         = prod(self.input_shape[1:])
        self.input_shape_ = (self.bs,self.D_in)#potentially different if flattened
        self.output_shape = (self.bs,K)
        self.input        = input_layer.m
        # PARAMETERS
	self.drop_   = tf.Variable(tf.ones((self.K,2,self.bs))*tf.reshape(tf.one_hot(1,2),[1,2,1]),trainable=False)
        self.sigmas2_= tf.Variable(tf.ones(self.D_in))
        self.sigmas2 = tf.expand_dims(self.sigmas2_,0)
        init_W = init_W(float32(1))
        if(leakiness is None):
	    self.W_  = tf.Variable(init_W((K,R,self.D_in))/sqrt(K+prod(self.input_shape[1:])))
            self.W   = self.W_
        else:
            self.W_  = tf.Variable(init_W((K,1,self.D_in))/sqrt(K+prod(self.input_shape[1:])))
            self.W   = tf.concat([self.W_,self.W_*self.leakiness],axis=1)
	self.pi      = tf.Variable(mysoftmax(tf.ones((K,R))))#;tf.add_to_collection('PARAMS',self.pi)
	self.b_      = tf.Variable(tf.zeros(self.D_in));tf.add_to_collection('PARAMS',self.b_)
	self.b       = tf.expand_dims(self.b_,0)
        self.V_      = tf.Variable(tf.zeros(self.K));tf.add_to_collection('PARAMS',self.V_)
        self.V       = tf.expand_dims(self.V_,0)
        # VI PARAMETERS
	self.m_      = tf.Variable(tf.random_normal((K,self.bs))*1);tf.add_to_collection('LATENT',self.m_)
        self.m       = tf.transpose(self.m_)
        self.p_      = tf.Variable(mysoftmax(tf.random_normal((K,self.bs,R))*0.1,axis=2));tf.add_to_collection('LATENT',self.p_)
        self.p       = tf.transpose(self.p_,[1,0,2])    
        self.v2_     = tf.Variable(tf.ones((K,self.bs)));tf.add_to_collection('LATENT',self.v2_)
        self.v2      = tf.transpose(self.v2_,[1,0])
	# placeholder for update and indices
        self.k_      = tf.placeholder(tf.int32)                                 # placeholder that will indicate which neuron is being updated
	self.W_indices = asarray(range(self.K))
        self.p_indices = self.W_indices
        # STATISTICS
        self.S1 = tf.Variable(tf.zeros((self.K,self.R,self.D_in)))
        self.S2 = tf.Variable(tf.zeros((self.K,self.R)))
        self.S3 = tf.Variable(tf.zeros((self.K,self.R)))
        self.S4 = tf.Variable(tf.zeros(self.D_in))
        self.S5 = tf.Variable(tf.zeros((self.K,self.R,self.K,self.R)))
        self.S6 = tf.Variable(tf.zeros(self.D_in))
        self.S7 = tf.Variable(tf.zeros((self.K,self.R)))
        self.S8 = tf.Variable(tf.zeros(self.D_in))
	# RESHAPE IF NEEDED
        if(len(self.input_shape)>2):
            self.is_flat = False
            self.input_  = tf.reshape(self.input,(self.bs,self.D_in))
        else:
            self.is_flat = True
            self.input_  = self.input
#                        self.input_,self.bn_m,self.bn_v = bn(self.input,[0])
    def vector2tensor(self,u):
        return tf.reshape(u,self.input_shape)
    def update_S(self):
        S1 = tf.einsum('nd,knr->krd',self.input_,tf.expand_dims(self.m_,-1)*self.p_)/self.bs
        S2 = tf.einsum('kn,knr->kr',self.m_,self.p_)/self.bs
        S3 = tf.einsum('kn,knr->kr',tf.square(self.m_)+self.v2_,self.p_)/self.bs
        if(self.is_flat):
            S4 = tf.reduce_sum(self.input_layer.v2_,1)/self.bs  
            S6 = tf.reduce_sum(self.input_layer.m_,1)/self.bs
        else:
            S4 = tf.reshape(tf.reduce_sum(self.input_layer.v2,0),[self.D_in])/self.bs
            S6 = tf.reshape(tf.reduce_sum(self.input_layer.m,0),[self.D_in])/self.bs
        S5 = (1-tf.reshape(tf.eye(self.K),[self.K,1,self.K,1]))*tf.einsum('knr,unv->kruv',tf.expand_dims(self.m_,-1)*self.p_,tf.expand_dims(self.m_,-1)*self.p_)/self.bs
        S7 = tf.reduce_sum(self.p_,1)/self.bs
        S8 = tf.reduce_sum(tf.square(self.input_),0)/self.bs
        return tf.group(tf.assign(self.S1,self.alpha*S1+(1-self.alpha)*self.S1),tf.assign(self.S2,self.alpha*S2+(1-self.alpha)*self.S2),
                        tf.assign(self.S3,self.alpha*S3+(1-self.alpha)*self.S3),tf.assign(self.S4,self.alpha*S4+(1-self.alpha)*self.S4),
                        tf.assign(self.S5,self.alpha*S5+(1-self.alpha)*self.S5),tf.assign(self.S6,self.alpha*S6+(1-self.alpha)*self.S6),
                        tf.assign(self.S7,self.alpha*S7+(1-self.alpha)*self.S7),tf.assign(self.S8,self.alpha*S8+(1-self.alpha)*self.S8))
    def backward(self,flat=1,resi=None):
	if(resi is None): resi = self.residual
	else: resi = resi*self.residual
	back = tf.einsum('knr,krd->nd',self.p_*tf.expand_dims(self.m_*self.drop_[:,1],-1),self.W)
	if(flat):
	    if(resi): return back+self.V*self.m
	    else:     return back
	else:
	    if(resi): return tf.reshape(back+self.V*self.m,self.input_shape)
	    else:     return tf.reshape(back,self.input_shape)
    def backwardmk(self,k,with_m=1,m=None,p=None):
	if(m is None): m = self.m_
	if(p is None): p = self.p_
        AA = p*tf.expand_dims(m*self.drop_[:,1]*tf.expand_dims(1-tf.one_hot(k,self.K),-1),-1)
        tf.Tensor.set_shape(AA,[self.K,self.bs,self.R])
	back = tf.einsum('knr,krd->nd',AA,self.W)
	if(self.residual==0): return back
	if(with_m):           return back+self.V*tf.transpose(m)
	else:                 return back+self.V*tf.transpose(m)*tf.expand_dims(1-tf.one_hot(k,self.K),0)
    def sample(self,M,K=None,sigma=1,deterministic=False):
	if(isinstance(self.input_layer,InputLayer)):sigma=0
	if(deterministic):    return tf.reshape(self.backward(1)+self.b,self.input_shape)
	noise = sigma*tf.random_normal((self.bs,self.D_in))*tf.sqrt(self.sigmas2)
	if(self.p_drop>0): dropout = tf.distributions.Bernoulli(probs=self.p_drop,dtype=tf.float32).sample((self.bs,self.K))
	else: dropout = float32(1)
        if(K==None):K = tf.one_hot(tf.transpose(tf.multinomial(tf.log(self.pi),self.bs)),self.R)
	tf.Tensor.set_shape(M,self.output_shape)
	MdropK = tf.einsum('nkr,krd->nd',tf.expand_dims(M*dropout,-1)*K,self.W)
	if(self.residual): return tf.reshape(MdropK+noise+self.residual*self.V*M+self.b,self.input_shape)
	else:              return tf.reshape(MdropK+self.b+noise,self.input_shape)
    def evidence(self):
	if(self.residual): 
    	    currentv2 = -tf.einsum('kn,k->n',self.v2_,tf.square(self.V_)/self.sigmas2_)*float32(0.5)
	    extra_v2  = -tf.einsum('kn,knr,rk,k->n',self.v2_,self.p_,tf.map_fn(lambda r:tf.diag_part(self.W[:,r,:]),tf.range(self.R,dtype=tf.int32),dtype=tf.float32),self.V_/self.sigmas2_)
	else:   currentv2 = 0 ; extra_v2 = 0
	if(self.p_drop>0): k1  = tf.reduce_sum(self.drop_[:,0],0)*tf.log(self.p_drop)+tf.reduce_sum(self.drop_[:,1],0)*tf.log(float32(1)-self.p_drop)
	else:              k1 = 0
        k2  = -tf.reduce_sum(tf.log(self.sigmas2_))*float32(0.5)                            #(1)
        k3  = tf.einsum('knr,kr->n',self.p_,tf.log(self.pi))                                #(1)
	k4  = -float32(self.D_in)*tf.log(2*PI_CONST)*float32(0.5)
        reconstruction    = -tf.einsum('nd,d->n',tf.square(self.input_-self.b-self.backward(flat=1,resi=1)),1/self.sigmas2_)*float32(0.5)                    #(1)
	if(isinstance(self.input_layer,DenseLayer)): 
	    input_v2  = -tf.einsum('dn,d->n',self.input_layer.v2_,1/self.sigmas2_)*float32(0.5)  #(1)
	else:   input_v2  = -tf.reduce_sum(tf.reshape(self.input_layer.v2,[self.bs,self.D_in])/self.sigmas2,1)*float32(0.5) #(1)
        sum_norm_k      = tf.einsum('kdn,d->n',tf.square(tf.einsum('krd,knr,kn->kdn',self.W,self.p_,self.m_*self.drop_[:,1])),1/self.sigmas2_)*float32(0.5)
        m2v2   = -tf.reduce_sum(self.drop_[:,1]*(self.v2_+tf.square(self.m_))*tf.einsum('knr,krd,d->kn',self.p_,tf.square(self.W),1/self.sigmas2_),0)*float32(0.5)
        return k1+k2+k3+k4+(reconstruction+input_v2+sum_norm_k+m2v2+currentv2+extra_v2)
    def likelihood(self,batch=None,pretraining=False):
        if(batch==False):
            if(pretraining==False): extra_k = 0
            else:                   extra_k = -0.5*tf.reduce_sum(tf.square(self.m_)+self.v2_)/self.bs
            if(self.residual):
                currentv2 = -tf.einsum('kn,k->',self.v2_,tf.square(self.V_)/self.sigmas2_)
                extra_v2  = -tf.einsum('kn,knr,rk,k->',self.v2_,self.p_,tf.map_fn(lambda r:tf.diag_part(self.W[:,r,:]),tf.range(self.R,dtype=tf.int32),dtype=tf.float32),self.V_/self.sigmas2_)*2
            else:currentv2 = 0 ; extra_v2 = 0
            if(self.p_drop>0): k1 = tf.reduce_sum(self.drop_[:,0])*tf.log(self.p_drop)+tf.reduce_sum(self.drop_[:,1])*tf.log(float32(1)-self.p_drop)
            else:              k1 = 0
            k2  = -tf.log(2*PI_CONST)*float32(self.D_in*0.5)-tf.reduce_sum(tf.log(self.sigmas2_))*float32(0.5)                            #(1)
            k3  = tf.einsum('knr,kr->',self.p_,tf.log(self.pi))*2                                #(1)
            reconstruction    = -tf.einsum('nd,d->',tf.square(self.input_-self.b-self.backward(flat=1,resi=1)),1/self.sigmas2_)                   #(1)
            if(self.is_flat): input_v2 = -tf.einsum('dn,d->',self.input_layer.v2_,1/self.sigmas2_)  #(1)
            else:             input_v2 = -tf.reduce_sum(tf.reshape(tf.reduce_sum(self.input_layer.v2,0),[self.D_in])/self.sigmas2_) #(1)
            sum_norm_k      = tf.einsum('kdn,d->',tf.square(tf.einsum('krd,knr,kn->kdn',self.W,self.p_,self.m_*self.drop_[:,1])),1/self.sigmas2_)
            m2v2   = -tf.reduce_sum(self.drop_[:,1]*(self.v2_+tf.square(self.m_))*tf.einsum('knr,krd,d->kn',self.p_,tf.square(self.W),1/self.sigmas2_))
            return k2+(k1+k3+m2v2+reconstruction+input_v2+sum_norm_k+currentv2+extra_v2)*float32(0.5/self.bs)-self.sparsity_prior*tf.reduce_sum(tf.square(self.W))+extra_k
        else:
            k1 = tf.log(2*PI_CONST)*float32(self.D_in*0.5)+tf.reduce_sum(tf.log(self.sigmas2_)*float32(0.5))
            k2 = tf.reduce_sum((self.S8+self.S4-2*self.b_*self.S6+tf.square(self.b_))/(2*self.sigmas2_))
            k3 = tf.einsum('krd,d->',(self.S1-tf.einsum('kr,d->krd',self.S2,self.b_))*self.W,1/self.sigmas2_)
            k4 = tf.einsum('abcd,abu,cdu->',self.S5,self.W,self.W/tf.expand_dims(2*self.sigmas2,0))
            k5 = tf.reduce_sum(tf.log(self.pi)*self.S7-self.S3*tf.einsum('krd,d->kr',tf.square(self.W),0.5/self.sigmas2_))
            return -k1-k2+k3-k4+k5
    def KL(self,pretraining=False):
	if(self.p_drop>0):
            return self.likelihood(0,pretraining)+(-tf.reduce_sum(self.p_*tf.log(self.p_+eps))+tf.reduce_sum(tf.log(self.v2_+eps))*float32(0.5)\
			-tf.reduce_sum(self.drop_*tf.log(self.drop_+eps)))/float32(self.bs)+float32(self.D_in/2.0)*tf.log(2*PI_CONST)
	else:
            return self.likelihood(0,pretraining)+(-tf.reduce_sum(self.p_*tf.log(self.p_+eps))+tf.reduce_sum(tf.log(self.v2_+eps))*float32(0.5))/float32(self.bs)+float32(self.D_in/2.0)*tf.log(2*PI_CONST)
                                                            ################### E STEP VARIABLES
    def update_v2(self,pretraining=False):
        if(pretraining==False): next_sigmas = self.next_layer.sigmas2
        else:                   next_sigmas = tf.ones_like(self.next_layer.sigmas2)
        scalor = tf.reduce_min(next_sigmas)
	a40 = self.drop_[:,1]*tf.einsum('knr,krd,d->kn',self.p_,tf.square(self.W),scalor/self.sigmas2_)+scalor/tf.transpose(next_sigmas)
	if(self.residual==0):  a4 = a40
	else:                  a4 = a40+tf.expand_dims(tf.square(self.V_)*scalor/self.sigmas2_,-1)+2*tf.einsum('knr,rk,k->kn',self.p_,tf.map_fn(lambda r:tf.diag_part(self.W[:,r,:]),tf.range(self.R),dtype=tf.float32),self.V_*scalor/self.sigmas2_)
	return tf.assign(self.v2_,scalor/a4)
    def update_m(self,ii=0,pretraining=False):
        rescalor = tf.reduce_min(self.sigmas2_)
        if(pretraining==False):
            back   = (self.next_layer.backward()+self.next_layer.b)*rescalor/self.next_layer.sigmas2
            next_sigmas = self.next_layer.sigmas2
        else:
            back   = tf.zeros_like(self.next_layer.b)
            next_sigmas = tf.ones_like(self.next_layer.b)
        Wp = tf.einsum('krd,knr->nkd',self.W,self.p_)
        b  = tf.einsum('nd,nkd->nk',(self.input_-self.b)*rescalor/self.sigmas2,Wp)+back
        D  = tf.matrix_diag(tf.einsum('krd,d,knr->nk',tf.square(self.W),rescalor/self.sigmas2_,self.p_))
        A  = tf.einsum('nkd,nad->nka',Wp,Wp*rescalor/tf.expand_dims(self.sigmas2,0))*(1-tf.expand_dims(tf.eye(self.K),0))+tf.expand_dims(tf.diag(rescalor/next_sigmas[0]),0)+D
        with(tf.device('/device:CPU:0')):
            new_m = tf.matrix_solve(A,tf.expand_dims(b,-1))[:,:,0]#,l2_regularizer=0,fast=True)[:,:,0]
        return tf.assign(self.m_,tf.transpose(new_m))
    def update_p(self):
        k  = self.k_
        Wk = self.W[self.k_];tf.Tensor.set_shape(Wk,[self.R,self.D_in])
        mk      = self.m_[self.k_]
	reconstruction = self.input_-self.b-self.backwardmk(self.k_,with_m=1)
        tf.Tensor.set_shape(reconstruction,[self.bs,self.D_in])
        forward = tf.einsum('nd,rd,n->nr',reconstruction,Wk/self.sigmas2,mk*self.drop_[self.k_,1])                        # (N R)
        prior   = tf.expand_dims(tf.log(self.pi[self.k_]),0)                                                                       # (1 R)
        m2v2    = tf.einsum('n,rd,d->nr',(tf.square(mk)+self.v2_[self.k_])*self.drop_[self.k_,1],tf.square(Wk),1/(2*self.sigmas2_)) # (N R)
	if(self.residual): extra_v2 = tf.einsum('n,r->nr',self.v2_[self.k_],self.W[self.k_,:,self.k_]*self.V_[self.k_]/self.sigmas2_[self.k_])
	else:              extra_v2 = 0
        new_p   = tf.scatter_update(self.p_,[self.k_],[tf.nn.softmax(forward+prior-m2v2-extra_v2)])
	return new_p
	if(self.p_drop==0):
	    return tf.group(new_m,new_p)
	## UPDATE DROPOUT
        proj    = tf.einsum('nd,rd,nr->n',reconstruction,self.W[k]/self.sigmas2,new_p[self.k_])*new_m[self.k_] # (N)
        squared = tf.einsum('n,nr,rd,d->n',self.v2_[self.k_]+tf.square(new_m[self.k_]),new_p[self.k_],tf.square(self.W[k]),1/self.sigmas2_) #(N)
        filled0 = tf.fill([self.bs],tf.cast(tf.log(self.p_drop),tf.float32))
        filled1 = tf.fill([self.bs],tf.cast(tf.log(1-self.p_drop),tf.float32))
        new_drop = mysoftmax(tf.stack([filled0,filled1-squared*float32(0.5)+proj]),axis=0,coeff=0.)
        return tf.group(new_m,new_p,tf.scatter_update(self.drop_,[k],[new_drop]))
                                            ######################## M STEP VARIABLES
    def update_sigma(self):
        value = self.S8+tf.square(self.b_)+self.S4+tf.einsum('kr,krd->d',self.S3,tf.square(self.W))+tf.einsum('kruv,krd,uvd->d',self.S5,self.W,self.W)-2*tf.einsum('krd,krd->d',self.S1,self.W)+2*self.b_*(tf.einsum('kr,krd->d',self.S2,self.W)-self.S6)
        value_ = tf.clip_by_value(value,0.000001,1000)
        if(self.sigma_opt=='local'):     return tf.assign(self.sigmas2_,value_)
        elif(self.sigma_opt=='global'):  return tf.assign(self.sigmas2_,tf.fill([self.D_in],tf.reduce_sum(value_)/self.D_in))
        elif(self.sigma_opt=='channel'):
            v=tf.reduce_sum(tf.reshape(value_,self.input_shape[1:]),axis=[0,1],keepdims=True)/(self.input_shape[1]*self.input_shape[2])
            return tf.assign(self.sigmas2_,tf.reshape(tf.ones([self.input_shape[1],self.input_shape[2],1])*v,[self.D_in]))
    def update_pi(self):
        return tf.assign(self.pi,tf.clip_by_value(self.S7/tf.reduce_sum(self.S7,axis=1,keepdims=True),0.01,0.99))
    def update_Wk(self):
        k = self.k_
        if(self.leakiness is None):
            new_w = (self.S1[k]-tf.einsum('r,d->rd',self.S2[k],self.b_)-tf.einsum('krd,tkr->td',self.W,self.S5[k]))/tf.expand_dims(self.S3[k],-1)
        else:
            new_w = tf.expand_dims((self.S1[k,0]+self.leakiness*self.S1[k,1]-(self.S2[k,0]+self.leakiness*self.S2[k,1])*self.b_-tf.einsum('krd,kr->d',self.W,self.S5[k,0]+self.leakiness*self.S5[k,1]))/(self.S3[k,0]+tf.square(self.leakiness)*self.S3[k,1]),0)
        return tf.scatter_update(self.W_,[self.k_],[new_w])#tf.nn.l2_normalize(new_w,0)])
    def update_BV(self):
        if(self.update_b): newb = tf.assign(self.b_,self.S6-tf.einsum('krd,kr->d',self.W,self.S2))
        else: newb = []
	if(self.residual==0): return newb
	back = tf.einsum('knr,krd->nd',self.p_*tf.expand_dims(self.m_*self.drop_[:,1],-1),self.W)  # (N D)
	forward = tf.einsum('kn,nk->k',self.m_,self.input_-tf.expand_dims(newb,0)-back)-tf.diag_part(tf.einsum('kn,knr,krd->kd',self.v2_,self.p_,self.W)) # K
	b = tf.reduce_sum(self.v2_+tf.square(self.m_),1)
	return tf.group(newb,tf.assign(self.V_,forward/b))






class ConvLayer:
    def __init__(self,input_layer,K,Ic,Jc,R,sparsity_prior = 0,leakiness=None,sigma='local',alpha=0.5,init_W = tf.orthogonal_initializer(),update_b=True):
        self.alpha             = tf.Variable(alpha)
        self.update_b          = update_b
        self.leakiness         = leakiness
	self.sigma_opt         = sigma
        self.sparsity_prior    = sparsity_prior
        self.input_layer       = input_layer
        input_layer.next_layer = self
        self.bs,self.Iin,self.Jin,self.C  = input_layer.output_shape 
        self.Ic,self.Jc,self.K,self.R     = Ic,Jc,K,R
        self.input             = input_layer.m
        self.input_shape       = input_layer.output_shape
        self.output_shape      = (self.bs,self.input_shape[-3]-self.Ic+1,self.input_shape[-2]-self.Jc+1,K)
	self.D_in              = prod(self.input_shape[1:])
        self.I,self.J          = self.output_shape[1],self.output_shape[2]
        self.input_patch       = self.extract_patch(self.input,with_n=1)
        self.pi_               = PI_CONST
        self.W_                = tf.Variable(init_W((self.K,self.Ic,self.Jc,self.C)));tf.add_to_collection('PARAMS',self.W_)
        self.W                 = tf.stack([self.W_,tf.zeros_like(self.W_)],axis=1)
	# WE DEFINE THE PARAMETERS
        self.pi             = tf.Variable(mysoftmax(tf.random_normal((K,R)),axis=1));tf.add_to_collection('PARAMS',self.W_)
	self.sigmas2_       = tf.Variable(tf.ones((self.Iin,self.Jin,self.C)));tf.add_to_collection('PARAMS',self.sigmas2_)
	self.sigmas2        = tf.expand_dims(self.sigmas2_,0)
        self.sigmas2_patch_ = self.extract_patch(self.sigmas2,with_n=0)
	self.sigmas2_patch  = tf.expand_dims(self.sigmas2_patch_,0)
	# SOME OTHER VARIABLES
	self.b_      = tf.Variable(tf.zeros(self.input_shape[1:]));tf.add_to_collection('PARAMS',self.b_)
	self.b       = tf.expand_dims(self.b_,0)
        self.b_patch_= self.extract_patch(self.b,with_n=0)
        self.b_patch = tf.expand_dims(self.b_patch_,0)
	self.m_      = tf.Variable(tf.zeros((K,self.I,self.J,self.bs)));tf.add_to_collection('LATENT',self.m_)
        self.m       = tf.transpose(self.m_,[3,1,2,0])   # (N I J K)
	self.p_      = tf.Variable(mysoftmax(tf.random_normal((K,self.I,self.J,self.R,self.bs)),axis=3));tf.add_to_collection('LATENT',self.p_)# (K,I,J,R,N)
        self.p       = tf.transpose(self.p_,[4,1,2,0,3]) # (N I J K R)
        self.v2_     = tf.Variable(tf.ones((self.K,self.I,self.J,self.bs)));tf.add_to_collection('LATENT',self.v2_) # (K I J N)
        self.v2      = tf.transpose(self.v2_,[3,1,2,0])
	self.drop_   = tf.Variable(tf.ones((K,2,self.I,self.J,self.bs))*tf.reshape(tf.one_hot(1,2),(1,2,1,1,1))) # (K 2 I J N)
        self.drop    = tf.transpose(self.drop_,[1,4,2,3,0]) #(2 N I J K)
        # STATISTICS
        self.S1 = tf.Variable(tf.zeros((self.C,self.Iin,self.Jin)))
        self.S2 = tf.Variable(tf.zeros((self.C,self.Iin,self.Jin)))
        self.S3 = tf.Variable(tf.zeros((self.K,self.I,self.J,self.Ic,self.Jc,self.C)))
        self.S4 = tf.Variable(tf.zeros((self.K,self.I,self.J)))
        self.S5 = tf.Variable(tf.zeros((self.K,self.I,self.J,self.K,self.Ic*2-1,self.Jc*2-1)))
        self.S6 = tf.Variable(tf.zeros((self.K,self.I,self.J)))
        self.S7 = tf.Variable(tf.zeros((self.C,self.Iin,self.Jin)))
        self.S8 = tf.Variable(tf.zeros((self.K,self.R)))
	#
        input_layer.next_layer = self
        self.k_      = tf.placeholder(tf.int32)
        self.i_      = tf.placeholder(tf.int32)
        self.j_      = tf.placeholder(tf.int32)
        self.r_      = tf.placeholder(tf.int32)
	self.ratio   = Ic
        self.Ni      = tf.cast(tf.ceil((self.I-tf.cast(self.i_,tf.float32))/self.ratio),tf.int32) # NUMBER OF TERMS
        self.Nj      = tf.cast(tf.ceil((self.J-tf.cast(self.j_,tf.float32))/self.ratio),tf.int32) # NUMBER OF TERMS
        self.xi,self.yi = tf.meshgrid(tf.range(self.j_,self.J,self.ratio),tf.range(self.i_,self.I,self.ratio)) # THE SECOND IS CONSTANT (meshgrid)
        self.indices_   = tf.concat([tf.fill([self.Ni*self.Nj,1],self.k_),tf.reshape(self.yi,(self.Ni*self.Nj,1)),tf.reshape(self.xi,(self.Nj*self.Ni,1))],axis=1) # (V 3) indices where the 1 pops
        self.m_indices_ = tf.concat([tf.concat([tf.fill([self.Ni*self.Nj,1],KK),tf.reshape(self.yi,(self.Ni*self.Nj,1)),tf.reshape(self.xi,(self.Nj*self.Ni,1))],axis=1) for KK in range(self.K)],axis=0)
        self.W_indices = range(self.K)#[a for a in itertools.product(range(self.K),range(self.Ic),range(self.Jc))]
	self.W_indices = asarray(self.W_indices)#.reshape((-1,1))
        self.m_indices = asarray([a for a in itertools.product(range(self.ratio),range(self.ratio))])
	self.p_indices = asarray([a for a in itertools.product(range(self.K),range(self.ratio),range(self.ratio))])
	mask           = tf.reshape(tf.one_hot(self.k_,self.K),(self.K,1,1,1))*tf.reshape(tf.tile(tf.one_hot(self.i_,self.ratio),[(self.I/self.ratio+1)]),(1,(self.I/self.ratio+1)*self.ratio,1,1))*tf.reshape(tf.tile(tf.one_hot(self.j_,self.ratio),[self.J/self.ratio+1]),(1,1,(self.J/self.ratio+1)*self.ratio,1))
	self.mask      = mask[:,:self.I,:self.J] # (K I J 1)
        m_mask         = tf.ones((self.K,1,1,1))*tf.reshape(tf.tile(tf.one_hot(self.i_,self.ratio),[(self.I/self.ratio+1)]),(1,(self.I/self.ratio+1)*self.ratio,1,1))*tf.reshape(tf.tile(tf.one_hot(self.j_,self.ratio),[self.J/self.ratio+1]),(1,1,(self.J/self.ratio+1)*self.ratio,1))
        self.m_mask    = m_mask[:,:self.I,:self.J]
        self.E= tf.einsum('ij,kc,a,b->kijcab',tf.ones((self.I,self.J)),tf.eye(self.K),tf.one_hot(self.Ic-1,2*self.Ic-1),tf.one_hot(self.Ic-1,2*self.Ic-1))
    def BN(self): return []#return tf.assign(self.m_,bn(self.m_,[1,2,3])[0])
    def filter_corr(self,A,B):
        #takes as input filter banks A and B of same shape (K a b c)
        sigma_W = tf.pad(tf.transpose(tf.reshape(tf.einsum('ijabc,kabc->abckij',1/self.sigmas2_patch_,A),[self.Ic,self.Jc,self.C,self.K*self.I*self.J]),[3,0,1,2]),[[0,0],[self.Ic-1,self.Ic-1],[self.Jc-1,self.Jc-1],[0,0]])
        patches = tf.reshape(tf.extract_image_patches(sigma_W,(1,self.Ic,self.Jc,1),(1,1,1,1),(1,1,1,1),"VALID"),(self.K*self.I*self.J,2*self.Ic-1,2*self.Jc-1,self.Ic,self.Jc,self.C))
        return tf.transpose(tf.reshape(tf.einsum('nijabc,kabc->kijn',patches,B),[self.K,2*self.Ic-1,2*self.Jc-1,self.K,self.I,self.J]),[3,4,5,0,1,2])*(1-self.E)#(K I J K' I' J')
    def update_S(self):
        S1 = tf.reduce_sum(tf.square(self.input_layer.m_),3)/self.bs
        S2 = tf.reduce_sum(self.input_layer.v2_,3)/self.bs
        S3 = tf.einsum('nijabc,kijn->kijabc',self.input_patch,self.m_*self.p_[:,:,:,0])/self.bs
        S4 = tf.reduce_sum((tf.square(self.m_)+self.v2_)*self.p_[:,:,:,0],3)/self.bs
        mp_patches = tf.reshape(tf.extract_image_patches(tf.pad(self.m*self.p[:,:,:,:,0],[[0,0],[self.Ic-1,self.Ic-1],[self.Jc-1,self.Jc-1],[0,0]]),(1,2*self.Ic-1,2*self.Jc-1,1),(1,1,1,1),(1,1,1,1),"VALID"),(self.bs,self.I,self.J,2*self.Ic-1,2*self.Jc-1,self.K)) # (N I J I' J' K)
        S5 = tf.einsum('kijn,nijabc->kijcab',self.m_*self.p_[:,:,:,0],mp_patches)/self.bs
        S6 = tf.reduce_sum(self.m_*self.p_[:,:,:,0],3)/self.bs
        S7 = tf.reduce_sum(self.input_layer.m_,3)/self.bs
        S8 = tf.reduce_sum(self.p_,[1,2,4])/self.bs
        return tf.group(tf.assign(self.S1,self.alpha*S1+(1-self.alpha)*self.S1),tf.assign(self.S2,self.alpha*S2+(1-self.alpha)*self.S2),
                        tf.assign(self.S3,self.alpha*S3+(1-self.alpha)*self.S3),tf.assign(self.S4,self.alpha*S4+(1-self.alpha)*self.S4),
                        tf.assign(self.S5,self.alpha*S5+(1-self.alpha)*self.S5),tf.assign(self.S6,self.alpha*S6+(1-self.alpha)*self.S6),
                        tf.assign(self.S7,self.alpha*S7+(1-self.alpha)*self.S7),tf.assign(self.S8,self.alpha*S8+(1-self.alpha)*self.S8))
    def extract_patch(self,u,with_n=1,with_reshape=1):
	patches = tf.extract_image_patches(u,(1,self.Ic,self.Jc,1),(1,1,1,1),(1,1,1,1),"VALID")
	if(with_reshape):
	    if(with_n): return tf.reshape(patches,(self.bs,self.output_shape[1],self.output_shape[2],self.Ic,self.Jc,self.C))
	    else:       return tf.reshape(patches,(self.output_shape[1],self.output_shape[2],self.Ic,self.Jc,self.C))
	return patches
#                                           ---- BACKWARD OPERATOR ---- 
    def deconv(self,input=None,masked_m=0,masked_w=0,m=None,p=None):
	if(m is None):m=self.m_
	if(p is None):p=self.p_
	if(masked_m==1):    m_masked = m*p[:,:,:,0]*(1-self.mask)# (N I J Ir JR K)->(N I' J' K)
	elif(masked_m==-1): m_masked = p[:,:,:,0]*self.mask#
        else:               m_masked = m*p[:,:,:,0]
        return tf.gradients(self.input_patch,self.input,tf.einsum('kijn,kabc->nijabc',m_masked,self.W_))[0]
    def sample(self,M,K=None,sigma=1):
	#multinomial returns [K,n_samples] with integer value 0,...,R-1
	if(isinstance(self.input_layer,InputLayer)):sigma=0
	noise      = sigma*tf.random_normal(self.input_shape)*tf.sqrt(self.sigmas2)
        sigma_hot  = tf.one_hot(tf.reshape(tf.multinomial(tf.log(self.pi),self.bs*self.I*self.J),(self.K,self.I,self.J,self.bs)),self.R) # (K I J N R)
        return (self.deconv(m=tf.transpose(M,[3,1,2,0]),p=tf.transpose(sigma_hot,[0,1,2,4,3]))+noise+self.b)
    def evidence(self): return 0
    def likelihood(self,batch=0,pretraining=False):
        if(batch==0):
            if(pretraining==False): extra_k = 0
            else:                   extra_k = -0.5*tf.reduce_sum(tf.square(self.m_)+self.v2_)/self.bs
            a1  = -tf.einsum('nijc,ijc->',tf.square(self.input-self.deconv()-self.b)+self.input_layer.v2,1/(2*self.sigmas2_*self.bs))
            a2  = tf.einsum('kijn,kabc,ijabc->',tf.square(self.m_*self.p_[:,:,:,0])-(tf.square(self.m_)+self.v2_)*self.p_[:,:,:,0],tf.square(self.W_),1/self.sigmas2_patch_)*0.5/self.bs
            k1  = -float32(self.D_in/2.0)*tf.log(2*PI_CONST)-tf.reduce_sum(tf.log(self.sigmas2_))/2+ tf.einsum('kr,krn->',tf.log(self.pi),tf.reduce_sum(self.p_,[1,2]))/float32(self.bs)
            return k1+a1+a2-self.sparsity_prior*tf.reduce_sum(tf.square(self.W))/2+extra_k
        else:
            k0 = -float32(self.D_in/2.0)*tf.log(2*PI_CONST)-tf.reduce_sum(tf.log(self.sigmas2_))/2+tf.reduce_sum(self.S8*tf.log(self.pi))
            k1 = -tf.reduce_sum((self.S1+self.S2+tf.square(tf.transpose(self.b_,[2,0,1]))-2*self.S7*tf.transpose(self.b_,[2,0,1]))/tf.transpose(self.sigmas2_,[2,0,1]))*0.5
            k2 = tf.reduce_sum(self.S3*tf.einsum('kabc,ijabc->kijabc',self.W_,1/self.sigmas2_patch_))
            k3 = -tf.reduce_sum(self.S6*tf.einsum('kabc,ijabc->kij',self.W_,self.b_patch_/self.sigmas2_patch_)+self.S4*tf.einsum('kabc,ijabc->kij',tf.square(self.W_),0.5/self.sigmas2_patch_))
            filters = self.filter_corr(self.W_,self.W_) # (kij k'i'j')
            k4 = -tf.reduce_sum(self.S5*filters)*0.5
            return k0+k4+k1+k2+k3-self.sparsity_prior*tf.reduce_sum(tf.square(self.W))/2
    def KL(self,pretraining=False):
        return self.likelihood(0,pretraining)+(-tf.reduce_sum(self.p_*tf.log(self.p_+eps))+float32(0.5)*tf.reduce_sum(tf.log(self.v2_)))/float32(self.bs)+float32(self.D_in/2.0)*tf.log(2*PI_CONST)
    def update_v2(self,pretraining=False):# DONE
        if(isinstance(self.next_layer,ConvLayer) or isinstance(self.next_layer,PoolLayer)): v_value = self.next_layer.sigmas2 # (N I J K)
        else:                                          v_value = tf.expand_dims(tf.reshape(self.next_layer.sigmas2_,self.output_shape[1:]),0)
        if(pretraining==False):next_sigmas = v_value
        else:                  next_sigmas = tf.ones_like(v_value)
        rescalor = 1#tf.reduce_min(self.sigmas2)
        a4 = tf.einsum('kijn,kabc,ijabc->kijn',self.p_[:,:,:,0],tf.square(self.W_),rescalor/self.sigmas2_patch_) # (K I J N)
        return tf.assign(self.v2_,rescalor/(tf.transpose(rescalor/next_sigmas,[3,1,2,0])+a4))
    def update_m(self,mp_opt=0,pretraining=False):
        m_masked = self.m_*self.p_[:,:,:,0]*(1-self.m_mask)
        reconstruction = tf.gradients(self.input_patch,self.input,tf.einsum('kijn,kabc->nijabc',m_masked,self.W_))[0]
        Wp = tf.einsum('kabc,kijn->nijkabc',self.W_,self.p_[:,self.i_::self.ratio,self.j_::self.ratio,0])
	if(isinstance(self.next_layer,ConvLayer) or isinstance(self.next_layer,PoolLayer)): 
            back = (self.next_layer.deconv()[:,self.i_::self.ratio,self.j_::self.ratio]\
                    +self.next_layer.b[:,self.i_::self.ratio,self.j_::self.ratio])
            A2   = self.next_layer.sigmas2_[self.i_::self.ratio,self.j_::self.ratio]
	else:       
            back = self.next_layer.vector2tensor(self.next_layer.backward()+self.next_layer.b)[:,self.i_::self.ratio,self.j_::self.ratio]
            A2   = tf.reshape(self.next_layer.sigmas2_,self.output_shape[1:])[self.i_::self.ratio,self.j_::self.ratio]
        if(pretraining):
            next_back   = tf.zeros_like(back)
            next_sigmas = tf.ones_like(A2)
        else:
            next_back   = back
            next_sigmas = A2
        rescalor = tf.reduce_min(self.sigmas2_patch_[self.i_::self.ratio,self.j_::self.ratio])
        b2 = tf.einsum('nijabc,nijkabc->nijk',self.extract_patch(self.input-reconstruction-self.b)[:,self.i_::self.ratio,self.j_::self.ratio]*rescalor/self.sigmas2_patch[:,self.i_::self.ratio,self.j_::self.ratio],Wp)
        bias = next_back*rescalor/tf.expand_dims(next_sigmas,0)+b2
        A1 = tf.einsum('nijkabc,nijdabc->nijkd',Wp,Wp*rescalor/myexpand(self.sigmas2_patch_[self.i_::self.ratio,self.j_::self.ratio],[0,-4]))*(1-tf.reshape(tf.eye(self.K),[1,1,1,self.K,self.K]))
        A3 = tf.einsum('kabc,kijn,ijabc->nijk',tf.square(self.W_),self.p_[:,self.i_::self.ratio,self.j_::self.ratio,0],rescalor/self.sigmas2_patch_[self.i_::self.ratio,self.j_::self.ratio])
        A = A1+tf.matrix_diag(A3)+tf.expand_dims(tf.matrix_diag(rescalor/next_sigmas),0)
        with(tf.device('/device:CPU:0')):
            new_m = tf.transpose(tf.matrix_solve(A,tf.expand_dims(bias,-1))[:,:,:,:,0],[0,3,1,2]) # (N K I J)
	return tf.scatter_nd_update(self.m_,self.m_indices_,tf.transpose(tf.reshape(new_m,(self.bs,-1))))
    def update_p(self):
        m_masked       = self.m_*self.p_[:,:,:,0]*(1-self.mask)
        reconstruction = tf.gradients(self.input_patch,self.input,tf.einsum('kijn,kabc->nijabc',m_masked,self.W_))[0]
        forward  = tf.einsum('nijabc,ijn,abc->nij',self.extract_patch((self.input-reconstruction-self.b)/self.sigmas2)[:,self.i_::self.ratio,self.j_::self.ratio],self.m_[self.k_,self.i_::self.ratio,self.j_::self.ratio],self.W_[self.k_])  # (N I' J')
	m2v2     = tf.einsum('ijn,abc,ijabc->nij',tf.square(self.m_[self.k_,self.i_::self.ratio,self.j_::self.ratio])\
                +self.v2_[self.k_,self.i_::self.ratio,self.j_::self.ratio],tf.square(self.W_[self.k_]),0.5/self.sigmas2_patch_[self.i_::self.ratio,self.j_::self.ratio]) #(N I' J')
	value    = forward-m2v2+tf.log(self.pi[self.k_,0])# (N I' J')
        update_value = tf.nn.softmax(tf.stack([value,tf.fill([self.bs,self.Ni,self.Nj],tf.log(self.pi[self.k_,1]))],axis=0),axis=0)
	return tf.scatter_nd_update(self.p_,self.indices_,tf.transpose(tf.reshape(update_value,(self.R,self.bs,-1)),[2,0,1]))
    def update_Wk(self):
#        rescalor = 1#tf.reduce_min(self.sigmas2_)
#        filters  = self.filter_corr(self.W_,self.W_) # (kij k'i'j')
#        k4       = tf.reduce_sum(rescalor*self.S5*filters)*0.5
#        value    = tf.gradients(k4,self.W_)[0][self.k_,self.i_,self.j_]
#        numerator = tf.reduce_sum((self.S3[self.k_,:,:,self.i_,self.j_,:]-tf.expand_dims(self.S6[self.k_],-1)*self.b_[self.i_:self.I+self.i_,self.j_:self.J+self.j_])*rescalor/self.sigmas2_[self.i_:self.I+self.i_,self.j_:self.J+self.j_],[0,1])-value
#        denominator = tf.einsum('ij,ijc->c',self.S4[self.k_],rescalor/self.sigmas2_[self.i_:self.I+self.i_,self.j_:self.J+self.j_])+self.sparsity_prior
#        return tf.scatter_nd_update(self.W_,[[self.k_,self.i_,self.j_]],[(numerator)/denominator])
        # FOR THE BIAS
        rS5 = tf.transpose(tf.reshape(tf.reshape(tf.extract_image_patches(tf.reshape(tf.transpose(self.S5[self.k_],[2,3,4,0,1]),[self.K,self.Ic*2-1,self.Jc*2-1,self.I*self.J]),(1,self.Ic,self.Jc,1),(1,1,1,1),(1,1,1,1),"VALID"),(self.K,self.Ic,self.Jc,self.Ic,self.Jc,self.I*self.J)),[self.K,self.Ic,self.Jc,self.Ic,self.Jc,self.I,self.J]),[5,6,0,1,2,3,4])
#        k11 = tf.einsum('ij,ijkabcd,kcde,ijabe->eab',rS5[:,:,self.k_,self.Ic-1,self.Jc-1,self.Ic-1,self.Jc-1],rS5[:,:,:,:,:,::-1,::-1],self.W_*tf.reshape(1-tf.one_hot(self.k_,self.K),[-1,1,1,1]),0.5/self.sigmas2_patch_)
#        k12 = tf.einsum('ij,ijkabcd,kcde,ijcde->eab',rS5[:,:,self.k_,self.Ic-1,self.Jc-1,self.Ic-1,self.Jc-1],rS5[:,:,:,::-1,::-1],self.W_*tf.reshape(1-tf.one_hot(self.k_,self.K),[-1,1,1,1]),0.5/self.sigmas2_patch_)

        sigma_W = tf.pad(tf.transpose(tf.reshape(tf.einsum('ijabc,kabc->abckij',1/self.sigmas2_patch_,self.W_),[self.Ic,self.Jc,self.C,self.K*self.I*self.J]),[3,0,1,2]),[[0,0],[self.Ic-1,self.Ic-1],[self.Jc-1,self.Jc-1],[0,0]])
        patches = tf.reshape(tf.extract_image_patches(sigma_W,(1,self.Ic,self.Jc,1),(1,1,1,1),(1,1,1,1),"VALID"),(self.K*self.I*self.J,2*self.Ic-1,2*self.Jc-1,self.Ic,self.Jc,self.C))
        VV =  tf.transpose(tf.reshape(tf.einsum('nijabc,kabc->kijn',patches,self.W_),[self.K,2*self.Ic-1,2*self.Jc-1,self.K,self.I,self.J]),[3,4,5,0,1,2])*(1-tf.reshape(tf.eye(self.K),[self.K,1,1,self.K,1,1]))#(K I J K' I' J')

        k1      = -tf.transpose(tf.gradients(tf.reduce_sum(VV*self.S5)*0.5,self.W_)[0][self.k_],[2,0,1])#-k11-k12
        k2      = tf.einsum('ijabc,ijabc->cab',self.S3[self.k_],1/self.sigmas2_patch_)
        k3      = -tf.einsum('ij,ijabc->cab',self.S6[self.k_],self.b_patch_/self.sigmas2_patch_)

        # FOR THE MATRIX
        D       = tf.einsum('ij,ijabc->cab',self.S4[self.k_],1/self.sigmas2_patch_)
#        patches = tf.reshape(tf.extract_image_patches(tf.transpose(self.S5[self.k_,:,:,self.k_,:,:],[0,2,3,1]),(1,self.Ic,self.Jc,1),(1,1,1,1),(1,1,1,1),"VALID"),(self.I,self.Ic,self.Jc,self.Ic,self.Jc,self.J))
        a1     = tf.einsum('ijabcd,ijabk->kabcd',rS5[:,:,self.k_,:,:,::-1,::-1],0.5/self.sigmas2_patch_)
        a2     = tf.einsum('ijabcd,ijcdk->kabcd',rS5[:,:,self.k_,::-1,::-1],0.5/self.sigmas2_patch_)
        # CREATE THE SYSTEM OF EQUATIONS
        A      = tf.reshape(a1+a2,[self.C,self.Ic*self.Jc,self.Ic*self.Jc])*(1-tf.expand_dims(tf.eye(self.Ic*self.Jc),0))+tf.matrix_diag(tf.reshape(D,[self.C,-1])+self.sparsity_prior)
        B      = tf.reshape(k1+k2+k3,[self.C,-1])
        with(tf.device('/device:CPU:0')):
            W      = tf.reshape(tf.matrix_solve(A,tf.expand_dims(B,-1)),[self.C,self.Ic,self.Jc])
        return tf.scatter_nd_update(self.W_,[[self.k_]],[tf.transpose(W,[1,2,0])])

    def update_pi(self): return tf.assign(self.pi,self.S8/tf.reduce_sum(self.S8,axis=1,keepdims=True))
    def update_BV(self):
        if(self.update_b == False): return []
        value = tf.transpose(self.S7,[1,2,0])-tf.gradients(self.sigmas2_patch_,self.sigmas2_,tf.einsum('kij,kabc->ijabc',self.S6,self.W_))[0]
        if(self.update_b=='local'): return tf.assign(self.b_,value)
        rescalor = tf.reduce_min(self.sigmas2_)
        value1 = value*rescalor/self.sigmas2_
        value2 = rescalor/self.sigmas2_
        if(self.update_b=='channel'): return tf.assign(self.b_,tf.ones_like(self.b_)*tf.reduce_sum(value1,[0,1],keepdims=True)/tf.reduce_sum(value2,[0,1],keepdims=True))
        return tf.assign(self.b_,tf.ones_like(self.b_)*tf.reduce_sum(value1,keepdims=True)/tf.reduce_sum(value2,keepdims=True))
    def update_sigma(self):
        v1 = tf.transpose(self.S1+self.S2-2*tf.transpose(self.b_,[2,0,1])*self.S7,[1,2,0])+tf.square(self.b_)
        v2 = tf.gradients(self.sigmas2_patch_,self.sigmas2_,tf.einsum('kij,kabc->ijabc',self.S4,tf.square(self.W_))-2*tf.einsum('kijabc,kabc->ijabc',self.S3,self.W_))[0]
        u  = tf.zeros_like(self.sigmas2_)
        pu = self.extract_patch(tf.expand_dims(u,0),with_n=0)
        sigma_W = tf.pad(tf.transpose(tf.reshape(tf.einsum('ijabc,kabc->abckij',pu,self.W_),[self.Ic,self.Jc,self.C,self.K*self.I*self.J]),[3,0,1,2]),[[0,0],[self.Ic-1,self.Ic-1],[self.Jc-1,self.Jc-1],[0,0]])
        patches = tf.reshape(tf.extract_image_patches(sigma_W,(1,self.Ic,self.Jc,1),(1,1,1,1),(1,1,1,1),"VALID"),(self.K*self.I*self.J,2*self.Ic-1,2*self.Jc-1,self.Ic,self.Jc,self.C))
        scalar = tf.reduce_sum(self.S5* tf.transpose(tf.reshape(tf.einsum('nijabc,kabc->kijn',patches,self.W_),[self.K,2*self.Ic-1,2*self.Jc-1,self.K,self.I,self.J]),[3,4,5,0,1,2])*(1-self.E))#(K I J K' I' J')
        v3 = tf.gradients(scalar,u)[0]
        v4 = tf.gradients(self.sigmas2_patch_,self.sigmas2_,tf.einsum('kij,kabc->ijabc',self.S6,self.W_))[0]*self.b_*2
        value = v1+v2+v3+v4
        if(self.sigma_opt=='local'):     return tf.assign(self.sigmas2_,value)
        elif(self.sigma_opt=='channel'): return tf.assign(self.sigmas2_,tf.reduce_sum(value,[0,1],keepdims=True)*tf.ones([self.Iin,self.Jin,1])/(self.Iin*self.Jin))
	elif(self.sigma_opt=='global'):  return tf.assign(self.sigmas2_,tf.fill([self.Iin,self.Jin,self.C],tf.reduce_sum(value)/(self.Iin*self.Jin*self.C)))








class PoolLayer(Layer):
    def __init__(self,input_layer,Ic,Jc,sigma='local',alpha=0.5):
        self.alpha = tf.Variable(alpha)
	self.sigma_opt         = sigma
        self.input_layer       = input_layer
        input_layer.next_layer = self
        self.bs,self.Iin,self.Jin,self.C  = input_layer.output_shape 
        self.Ic,self.Jc,self.K = Ic,Jc,input_layer.output_shape[-1]
        K=self.K
        R,self.R = Ic*Jc,Ic*Jc
        self.input             = input_layer.m
        self.input_shape       = input_layer.output_shape
        self.output_shape      = (self.bs,self.input_shape[-3]/self.Ic,self.input_shape[-2]/self.Jc,K)
	self.D_in              = prod(self.input_shape[1:])
        self.I,self.J          = self.output_shape[1],self.output_shape[2]
        self.input_patch       = self.extract_patch(self.input,with_n=1)
        self.W_                = tf.Variable(tf.stack([tf.reshape(tf.one_hot(i,Ic*Jc),(Ic,Jc)) for i in xrange(Ic*Jc)]))# (R,Ic,Jc) always D_in last
        self.W                 = self.W_
	# WE DEFINE THE PARAMETERS
        self.pi      = tf.Variable(tf.ones((R,))/float32(R))
	self.sigmas2_= tf.Variable(tf.ones((self.Iin,self.Jin,self.C)))
	self.sigmas2 = tf.expand_dims(self.sigmas2_,0)
        self.sigmas2_patch_= self.extract_patch(self.sigmas2,with_n=0)
	self.sigmas2_patch = tf.expand_dims(self.sigmas2_patch_,0)
	# SOME OTHER VARIABLES
	self.b_      = tf.Variable(tf.zeros_like(self.sigmas2_))
	self.b       = tf.expand_dims(self.b_,0)
#        self.b_patch_= self.extract_patch(self.b,with_n=0)
#        self.b_patch = tf.expand_dims(self.b_patch_,0)
	self.m_      = tf.Variable(tf.zeros((K,self.I,self.J,self.bs)))
        self.m       = tf.transpose(self.m_,[3,1,2,0])   # (N I J K)
	self.p_      = tf.Variable(tf.zeros((K,self.I,self.J,self.R,self.bs)))# (K,I,J,R,N)
        self.p       = tf.transpose(self.p_,[4,1,2,0,3]) # (N I J K R)
        self.v2_     = tf.Variable(tf.zeros((self.K,self.I,self.J,self.bs))) # (K I J N)
        self.v2      = tf.transpose(self.v2_,[3,1,2,0])
        # STATISTICS
        self.Sm2    = tf.Variable(tf.zeros((self.K,self.Iin,self.Jin)))
        self.Sv2    = tf.Variable(tf.zeros((self.K,self.Iin,self.Jin)))
        self.SmMp   = tf.Variable(tf.zeros((self.K,self.I,self.J,self.R,self.Ic,self.Jc)))
        self.SM2v2p = tf.Variable(tf.zeros((self.K,self.I,self.J,self.R)))
        self.SMp    = tf.Variable(tf.zeros((self.K,self.I,self.J,self.R)))
        self.Sm     = tf.Variable(tf.zeros((self.K,self.Iin,self.Jin)))
        self.Sp     = tf.Variable(tf.zeros((self.R,)))
	#
        input_layer.next_layer = self
    def update_S(self):
        Sm2    = tf.reduce_sum(tf.square(self.input_layer.m_)/self.bs,3)
        Sv2    = tf.reduce_sum(self.input_layer.v2_,3)/self.bs
        SmMp   = tf.einsum('nijabc,cijrn->cijrab',self.input_patch,tf.expand_dims(self.m_,3)*self.p_)/self.bs
        SM2v2p = tf.reduce_sum(tf.expand_dims(tf.square(self.m_)+self.v2_,3)/self.bs*self.p_,4)
        SMp    = tf.reduce_sum(tf.expand_dims(self.m_,3)*self.p_,4)/self.bs
        Sm     = tf.reduce_sum(self.input_layer.m_,3)/self.bs
        Sp     = tf.reduce_sum(self.p_,[0,1,2,4])/self.bs
        return tf.group(tf.assign(self.Sm2,self.alpha*Sm2+(1-self.alpha)*self.Sm2),tf.assign(self.Sv2,self.alpha*Sv2+(1-self.alpha)*self.Sv2),
                        tf.assign(self.SmMp,self.alpha*SmMp+(1-self.alpha)*self.SmMp),tf.assign(self.SM2v2p,self.alpha*SM2v2p+(1-self.alpha)*self.SM2v2p),
                        tf.assign(self.SMp,self.alpha*SMp+(1-self.alpha)*self.SMp),tf.assign(self.Sm,self.alpha*Sm+(1-self.alpha)*self.Sm),
                        tf.assign(self.Sp,self.alpha*Sp+(1-self.alpha)*self.Sp))
    def extract_patch(self,u,with_n=1,with_reshape=1):
	patches = tf.extract_image_patches(u,(1,self.Ic,self.Jc,1),(1,self.Ic,self.Jc,1),(1,1,1,1),"VALID")
	if(with_reshape):
	    if(with_n): return tf.reshape(patches,(self.bs,self.output_shape[1],self.output_shape[2],self.Ic,self.Jc,self.C))
	    else:       return tf.reshape(patches,(self.output_shape[1],self.output_shape[2],self.Ic,self.Jc,self.C))
	return patches
#                                           ---- BACKWARD OPERATOR ---- 
    def deconv(self,input=None,masked_m=0,masked_w=0,m=None,p=None):
	if(m is None):m=self.m_
	if(p is None):p=self.p_
        return tf.gradients(self.input_patch,self.input,tf.einsum('kijn,kijrn,rab->nijabk',m,p,self.W_))[0]
    def sample(self,M,K=None,sigma=1):
	#multinomial returns [K,n_samples] with integer value 0,...,R-1
	if(isinstance(self.input_layer,InputLayer)):sigma=0
	noise      = sigma*tf.random_normal(self.input_shape)*tf.sqrt(self.sigmas2)
        sigma_hot  = tf.one_hot(tf.reshape(tf.multinomial(tf.expand_dims(tf.log(self.pi),0),self.bs*self.I*self.J*self.K),(self.K,self.I,self.J,self.bs)),self.R) # (K I J N R)
        return self.deconv(m=tf.transpose(M,[3,1,2,0]),p=tf.transpose(sigma_hot,[0,1,2,4,3]))+noise
    def evidence(self): return 0
    def likelihood(self,batch=0,pretraining=False):
        if(batch==0):
            if(pretraining==False):  extra_k = 0
            else:                    extra_k = -0.5*tf.reduce_sum((tf.square(self.m_)+self.v2_)/self.bs)
            a1  = -tf.einsum('nijc,ijc->',tf.square(self.input)+self.input_layer.v2,1/(2*self.sigmas2_*self.bs))
            a2  = tf.reduce_sum(self.input*self.deconv()/self.sigmas2)/self.bs
            a3  = -tf.reduce_sum((tf.square(self.m_)+self.v2_)*tf.einsum('ijabk,kijabn->kijn',1/(self.bs*self.sigmas2_patch_),tf.reshape(self.p_,[self.K,self.I,self.J,self.Ic,self.Jc,self.bs])))*0.5
            k1  = -float32(self.D_in/2.0)*tf.log(2*PI_CONST)-tf.reduce_sum(tf.log(self.sigmas2_+eps))/2
            k2  = tf.einsum('r,kijrn->',tf.log(self.pi),self.p_)/float32(self.bs)
            return k1+k2+(a1+a2+a3)+extra_k
        else:
            k1  = -float32(self.D_in/2.0)*tf.log(2*PI_CONST)-tf.reduce_sum(tf.log(self.sigmas2_+eps))/2
            k2  = tf.reduce_sum(self.Sp*tf.log(self.pi))
            a1  = tf.reduce_sum((-self.Sm2-self.Sv2)/tf.transpose(self.sigmas2_,[2,0,1]))*0.5
            a2  = tf.reduce_sum(self.SmMp*tf.einsum('rab,ijabc->cijrab',self.W_,1/self.sigmas2_patch_))
            a3 = -tf.reduce_sum(self.SM2v2p*tf.einsum('rab,ijabc->cijr',tf.square(self.W_),1/self.sigmas2_patch_))*0.5
#            a4 = -tf.einsum('cijr,rab,ijabc->',self.SMp,self.W_,self.b_patch_/self.sigmas2_patch_)*0.5
            return k1+k2+a1+a2+a3
    def KL(self,pretraining=False):
            return self.likelihood(0,pretraining)+(-tf.reduce_sum(self.p_*tf.log(self.p_+eps))+float32(0.5)*tf.reduce_sum(tf.log(self.v2_+eps)))/float32(self.bs)+float32(self.D_in/2.0)*tf.log(2*PI_CONST)
    def update_v2(self,pretraining=False):# DONE
        if(isinstance(self.next_layer,ConvLayer)): v_value = self.next_layer.sigmas2 # (N I J K)
        else:                                      v_value = tf.expand_dims(tf.reshape(self.next_layer.sigmas2_,self.output_shape[1:]),0) # (N I J K)
        if(pretraining):     next_sigmas = tf.ones_like(v_value)
        else:                next_sigmas = v_value
        rescalor = tf.reduce_min(self.sigmas2_)
        a4       = tf.transpose(rescalor/self.sigmas2,[3,1,2,0])#tf.einsum('kijrn,rab,ijabc->kijn',self.p_,tf.square(self.W_),rescalor/self.sigmas2_patch_) # (K I J N)
        return tf.assign(self.v2_,rescalor/(tf.einsum('kijabn,ijabk->kijn',tf.reshape(self.p_,[self.K,self.I,self.J,self.Ic,self.Jc,self.bs]),rescalor/self.sigmas2_patch_)+tf.transpose(rescalor/next_sigmas,[3,1,2,0])))
    def update_m(self,mp_opt=0,pretraining=False):
        forward  = tf.einsum('nijabc,cijrn,rab->nijc',self.extract_patch(self.input/self.sigmas2),self.p_,self.W_)# (N I J K)
	if(isinstance(self.next_layer,ConvLayer)): back = (self.next_layer.deconv()+self.next_layer.b)/self.next_layer.sigmas2
        elif(isinstance(self.next_layer,ContinuousLastLayer)): back = self.next_layer.vector2tensor((self.next_layer.backward()+self.next_layer.b)/self.next_layer.sigmas2)
	else:                                      back = self.next_layer.vector2tensor((self.next_layer.backward()+self.next_layer.b)/self.next_layer.sigmas2)
        if(pretraining):                           back = tf.zeros_like(back)
        update_value_m = (forward+back)*self.v2 # (N I J K)
	return tf.assign(self.m_,tf.transpose(update_value_m,[3,1,2,0]))
    def update_p(self):
        forward = tf.transpose(tf.reshape(tf.einsum('nijabc,cijn->cijnab',self.extract_patch(self.input/self.sigmas2),self.m_),[self.K,self.I,self.J,self.bs,self.R]),[0,1,2,4,3])
        update_value = tf.nn.softmax(forward,axis=3)
	return tf.assign(self.p_,update_value)
    def update_pi(self): return tf.assign(self.pi,self.Sp/tf.reduce_sum(self.Sp))
    def update_sigma(self):
        v1 = tf.transpose(self.Sm2+self.Sv2,[1,2,0])
        v2 = -2*tf.gradients(self.sigmas2_patch_,self.sigmas2_,tf.einsum('cijrab,rab->ijabc',self.SmMp,self.W_))[0]
        v3 = tf.gradients(self.sigmas2_patch_,self.sigmas2_,tf.einsum('cijr,rab->ijabc',self.SM2v2p,tf.square(self.W_)))[0]
        v4 = tf.gradients(self.sigmas2_patch_,self.sigmas2_,tf.einsum('cijr,rab->ijabc',self.SMp,self.W_))[0]*self.b_
        value = v1+v2+v3+v4
        if(self.sigma_opt=='local'):     return tf.assign(self.sigmas2_,value)
        elif(self.sigma_opt=='channel'): return tf.assign(self.sigmas2_,tf.reduce_sum(value,[0,1],keepdims=True)*tf.ones([self.Iin,self.Jin,1])/(self.Iin*self.Jin))
	elif(self.sigma_opt=='global'):  return tf.assign(self.sigmas2_,tf.fill([self.Iin,self.Jin,self.C],tf.reduce_sum(value)/(self.Iin*self.Jin*self.C)))





#########################################################################################################################
#
#
#                                       INPUT/OUTPUT LAYERS
#
#
#########################################################################################################################




class InputLayer(Layer):
        def __init__(self,input_shape,alpha=0.5):
            self.bs           = input_shape[0]
            self.input_shape,self.output_shape  = input_shape,input_shape
            self.alpha        = tf.Variable(alpha)
            self.mask         = tf.Variable(tf.zeros(input_shape))
	    self.v2           = tf.Variable(tf.zeros(input_shape))
            self.m            = tf.Variable(tf.zeros(self.input_shape))
            self.m_           = tf.transpose(self.m,[3,1,2,0])
	    self.v2_          = tf.transpose(self.v2,[3,1,2,0])
	def init_thetaq(self):return tf.assign(self.v2,self.mask)
        def likelihood(self,batch=0,pretraining=False): return float32(0)
        def KL(self,pretraining=False):	return float32(0.5)*tf.reduce_sum(tf.log(self.v2+eps)*self.mask)/float32(self.input_shape[0])
        def update_v2(self,pretraining=False):
		if(isinstance(self.next_layer,PoolLayer) or isinstance(self.next_layer,ConvLayer)):  a40 = self.next_layer.sigmas2
                else:                a40 = tf.expand_dims(tf.reshape(self.next_layer.sigmas2_,self.input_shape[1:]),0)
                return tf.assign(self.v2,self.mask*a40)
        def update_m(self,opt=0,pretraining=False):
                if(isinstance(self.next_layer,PoolLayer) or isinstance(self.next_layer,ConvLayer)):  priorm  = self.next_layer.deconv()+self.next_layer.b
		else:   priorm  = self.next_layer.vector2tensor(self.next_layer.backward()+self.next_layer.b)
                return tf.assign(self.m,priorm*self.mask+self.m*(1-self.mask))
        def evidence(self): return float32(0)



class CategoricalLastLayer(Layer):
    def __init__(self,input_layer,R,sparsity_prior=0,sigma='local',alpha=0.5,update_b=True,init_W = tf.orthogonal_initializer(float32(1))):
        self.bs                            = input_layer.bs
        self.D_in                          = prod(input_layer.output_shape[1:])
        self.input_shape,self.input_shape_,self.output_shape = input_layer.output_shape,(self.bs,self.D_in),(self.bs,1)
        self.update_b          = update_b
        self.mask              = tf.Variable(tf.zeros(input_layer.output_shape[0]))
        self.alpha             = tf.Variable(alpha)
        self.input_layer       = input_layer
        input_layer.next_layer = self
	self.sigma_opt         = sigma
	self.sparsity_prior    = sparsity_prior
        self.input             = input_layer.m
	self.R                 = R
	# PARAMETERS
        self.sigmas2_= tf.Variable(tf.ones(self.D_in))
        self.sigmas2 = tf.expand_dims(self.sigmas2_,0)
	self.W       = tf.Variable(init_W((R,self.D_in)))
	self.pi      = tf.Variable(tf.ones(R)/R)
        self.b_      = tf.Variable(tf.zeros(self.D_in))
        self.b       = tf.expand_dims(self.b_,0)
	# VI PARAMETERS
        self.p_      = tf.Variable(mysoftmax(tf.random_normal((self.bs,R)),axis=1));tf.add_to_collection('LATENT',self.p_) # (N R)
        # STATISTICS
        self.Spm = tf.Variable(tf.zeros((self.R,self.D_in)))
        self.Sv2 = tf.Variable(tf.zeros(self.D_in))
        self.Sm = tf.Variable(tf.zeros(self.D_in))
        self.Sp = tf.Variable(tf.zeros(self.R))
        self.Sm2 = tf.Variable(tf.zeros((self.D_in)))
        if(len(self.input_shape)>2):
            self.is_flat = False
            self.input_  = tf.reshape(self.input,(self.bs,self.D_in))
        else:
            self.is_flat = True
            self.input_  = self.input
    def vector2tensor(self,u):
        return tf.reshape(u,self.input_shape)
    def update_S(self):
        Spm = tf.einsum('nd,nr->rd',self.input_/self.bs,self.p_)
        if(self.is_flat): Sv2 = tf.reduce_sum(self.input_layer.v2_,1)/self.bs
        else:             Sv2 = tf.reshape(tf.reduce_sum(self.input_layer.v2,0),[self.D_in])/self.bs
        Sp   = tf.reduce_sum(self.p_,0)/self.bs
        Sm2  = tf.reduce_sum(tf.square(self.input_)/self.bs,0)
        Sm   = tf.reduce_sum(self.input_/self.bs,0)
        return tf.group(tf.assign(self.Spm,self.alpha*Spm+(1-self.alpha)*self.Spm),tf.assign(self.Sv2,self.alpha*Sv2+(1-self.alpha)*self.Sv2),
                        tf.assign(self.Sp,self.alpha*Sp+(1-self.alpha)*self.Sp),tf.assign(self.Sm2,self.alpha*Sm2+(1-self.alpha)*self.Sm2),
                        tf.assign(self.Sm,self.alpha*Sm+(1-self.alpha)*self.Sm))#,tf.assign(self.S6,self.alpha*S6+(1-self.alpha)*self.S6),tf.assign(self.S7,self.alpha*S7+(1-self.alpha)*self.S7))
    def backward(self,flat=1):
	if(flat):  return tf.tensordot(self.p_,self.W,[[1],[0]])
	else:      return self.vector2tensor(tf.tensordot(self.p_,self.W,[[1],[0]]))
#    def backwardk(self,k): return tf.tensordot(self.p_,self.W[:,k],[[1],[0]])
    def sample(self,samples,K=None,sigma=1,deterministic=0):
        """ K must be a pre imposed region used for generation
        if not given it is generated according to pi, its shape 
        must be (N K R) with a one hot vector on the last dimension
        sampels is a dummy variable not used in this layer   """
        noise = sigma*tf.random_normal(self.input_shape_)*tf.sqrt(self.sigmas2)
        if(K is None): K = tf.one_hot(tf.multinomial(tf.log(tf.expand_dims(self.pi,0)),self.bs)[0],self.R)
	if(self.is_flat): return tf.tensordot(K,self.W,[[1],[0]])+noise+self.b
	else:             return self.vector2tensor(tf.tensordot(K,self.W,[[1],[0]])+noise+self.b)
    def evidence(self):
        k1  = -tf.reduce_sum(tf.log(self.sigmas2_)/2)
        k2  = tf.einsum('nr,r->n',self.p_,tf.log(self.pi))
	k3  = -float32(self.D_in)*tf.log(2*PI_CONST)*float32(0.5)
        reconstruction  = -tf.einsum('nd,d->n',tf.square(self.input_-self.b-self.backward(1)),1/self.sigmas2_)*float32(0.5)
        if(isinstance(self.input_layer,DenseLayer)):
	    input_v2  = -tf.einsum('dn,d->n',self.input_layer.v2_,1/self.sigmas2_)*float32(0.5)
	else: input_v2= -tf.reduce_sum(tf.reshape(self.input_layer.v2,[self.bs,self.D_in])/self.sigmas2,1)*float32(0.5)
        m2v2        = -tf.einsum('nr,rd,d->n',self.p_,tf.square(self.W),1/self.sigmas2_)*float32(0.5) #(N D) -> (D)
        sum_norm_k  = tf.einsum('nd,d->n',tf.square(tf.tensordot(self.p_,self.W,[[1],[0]])),1/self.sigmas2_)*float32(0.5) #(D)
        return k1+k2+k3+(reconstruction+input_v2+sum_norm_k+m2v2)
    def likelihood(self,batch=None,pretraining=False):
        if(batch==0):
            k1  = -tf.reduce_sum(tf.log(self.sigmas2_))*float32(0.5)-tf.log(2*PI_CONST)*float32(self.D_in*0.5)
            k2  = tf.einsum('nr,r->',self.p_,tf.log(self.pi))/self.bs
            if(self.is_flat): v2 = self.input_layer.v2
            else:             v2 = tf.reshape(self.input_layer.v2,[self.bs,self.D_in])
            k3  = -tf.einsum('nrd,d,nr->',tf.square(tf.expand_dims(self.input_-self.b,1)-tf.expand_dims(self.W,0)),0.5/self.sigmas2_,self.p_)/self.bs
            k4  = -tf.einsum('nd,d->',v2,0.5/self.sigmas2_)/self.bs
            return k1+k2+k3+k4-self.sparsity_prior*tf.reduce_sum(tf.square(self.W))/2
        else:
            k1 = -tf.reduce_sum(tf.log(self.sigmas2_))*float32(0.5)-tf.log(2*PI_CONST)*float32(self.D_in*0.5)
            k2 = tf.reduce_sum(self.Sp*tf.log(self.pi))
            k3 = tf.reduce_sum(self.Spm*self.W/self.sigmas2)
            k4 = -0.5*tf.reduce_sum((tf.square(self.b_)+self.Sm2+self.Sv2-2*self.Sm*self.b_)/self.sigmas2_)
            k5 = -0.5*tf.reduce_sum(self.Sp*tf.einsum('rd,d->r',tf.square(self.W)+2*self.W*self.b,1/self.sigmas2_))
            return k1+k2+k3+k4+k5-self.sparsity_prior*tf.reduce_sum(tf.square(self.W))/2
    def KL(self,pretraining=False):
        return self.likelihood(0,pretraining)-tf.reduce_sum(self.p_*tf.log(self.p_+eps))/float32(self.bs)
                                        ################### E STEP UPDATES
    def update_p(self,ii=0,pretraining=False):
	proj    = tf.einsum('nd,rd->nr',self.input_-self.b,self.W/self.sigmas2) # ( N R)
        prior   = tf.expand_dims(self.pi,0)
        m2v2    = -tf.expand_dims(tf.einsum('rd,d->r',tf.square(self.W),float32(0.5)/self.sigmas2_),0) # ( 1 R )
        V       = mysoftmax(proj+m2v2+tf.log(prior),coeff=0.0)
        return tf.assign(self.p_,V*tf.expand_dims(self.mask,-1)+self.p_[0]*(1-tf.expand_dims(self.mask,-1)))
                                        ################### M STEP UPDATES
    def update_sigma(self):
        C1 = tf.einsum('r,rd->d',self.Sp,tf.square(self.W))+tf.square(self.b_)+self.Sm2+self.Sv2
        C2 = -2*tf.reduce_sum(self.Spm*self.W,0)-2*self.Sm*self.b_+2*tf.einsum('r,rd->d',self.Sp,self.W*self.b)
        value = C1+C2#self.S6+self.S2-tf.reduce_sum(self.Sp*self.W,0)
        value_ = tf.clip_by_value(value,0.00000001,1000)
        if(self.sigma_opt=='local'):    return tf.assign(self.sigmas2_,value_)
        elif(self.sigma_opt=='global'): return tf.assign(self.sigmas2_,tf.fill([self.D_in],tf.reduce_sum(value_)/self.D_in))
        elif(self.sigma_opt=='none'):   return []
    def update_pi(self): return []#tf.assign(self.pi,self.Sp/tf.reduce_sum(self.Sp))
    def update_Wk(self): return tf.assign(self.W,(self.Spm-tf.einsum('r,d->rd',self.Sp,self.b_))/(tf.expand_dims(self.Sp,-1)+self.sparsity_prior))
    def update_BV(self):
        if(self.update_b): return tf.assign(self.b_,self.Sm-tf.einsum('rd,r->d',self.W,self.Sp))
        else:              return []



class ContinuousLastLayer(Layer):
    def __init__(self,input_layer,K,sigma_opt='local',sparsity_prior=0,init_W=tf.orthogonal_initializer(1)):
        self.alpha             = tf.Variable(float32(0))
        self.sigma_opt        = sigma_opt
        self.input_shape       = input_layer.output_shape
        self.bs,self.D_in,self.K=input_layer.bs,prod(self.input_shape[1:]),K
        self.input             = input_layer.m
        self.input_layer       = input_layer
        input_layer.next_layer = self
        self.sparsity_prior    = sparsity_prior
        # PARAMETERS
        self.sigmas2_= tf.Variable(tf.ones(self.D_in))
        self.sigmas2 = tf.expand_dims(self.sigmas2_,0)
        W = init_W((K,self.D_in))
        self.W       = tf.Variable(W/sqrt(K+prod(self.input_shape[1:])))
        self.b_      = tf.Variable(tf.zeros(self.D_in))
        self.b       = tf.expand_dims(self.b_,0)
        input_layer.next_layer_sigmas2 = self.sigmas2_
        # VI PARAMETERS
        self.m_      = tf.Variable(tf.zeros((self.bs,K)))# (N K)
        self.v2_     = tf.Variable(tf.zeros((K,)))# (N K)
        # STATISTICS
        self.SmM = tf.Variable(tf.zeros((self.D_in,K)))
        self.Sv2 = tf.Variable(tf.zeros(self.D_in))
        self.Sm  = tf.Variable(tf.zeros(self.D_in))
        self.SV2 = tf.Variable(tf.zeros(self.K))
        self.SM  = tf.Variable(tf.zeros(self.K))
        self.Sm2 = tf.Variable(tf.zeros((self.D_in)))
        self.SMM = tf.Variable(tf.zeros((self.K,self.K)))
        self.SM2 = tf.Variable(tf.zeros(self.K))
        #
        self.k_ = tf.placeholder(tf.int32)
        if(len(self.input_shape)>2):
            self.is_flat = False
            self.input_  = tf.reshape(self.input,(self.bs,self.D_in))
        else:
            self.is_flat = True
            self.input_  = self.input
        self.W_indices = asarray(range(self.K))
    def update_S(self):
        SmM = tf.einsum('nd,nk->dk',self.input_/self.bs,self.m_)
        if(self.is_flat): Sv2 = tf.reduce_sum(self.input_layer.v2_,1)/self.bs;
        else:             Sv2 = tf.reshape(tf.reduce_sum(self.input_layer.v2,0),[self.D_in])/self.bs
        Sm   = tf.reduce_sum(self.input_,0)/self.bs
        SV2  = self.v2_#tf.reduce_sum(self.v2_,0)/self.bs 
        SM   = tf.reduce_sum(self.m_,0)/self.bs
        Sm2  = tf.reduce_sum(tf.square(self.input_)/self.bs,0)
        SMM  = tf.einsum('nk,nc->kc',self.m_,self.m_/self.bs)
        SM2  = tf.reduce_sum(tf.square(self.m_)/self.bs,0)
        return tf.group(tf.assign(self.SmM,self.alpha*SmM+(1-self.alpha)*self.SmM),tf.assign(self.Sv2,self.alpha*Sv2+(1-self.alpha)*self.Sv2),
                        tf.assign(self.Sm,self.alpha*Sm+(1-self.alpha)*self.Sm),tf.assign(self.SV2,self.alpha*SV2+(1-self.alpha)*self.SV2),
                        tf.assign(self.SM,self.alpha*SM+(1-self.alpha)*self.SM),tf.assign(self.Sm2,self.alpha*Sm2+(1-self.alpha)*self.Sm2),
                        tf.assign(self.SMM,self.alpha*SMM+(1-self.alpha)*self.SMM),tf.assign(self.SM2,self.alpha*SM2+(1-self.alpha)*self.SM2))
    def vector2tensor(self,u):
        return tf.reshape(u,self.input_shape)
    def backward(self,flat=1,resi=None): 
        value = tf.einsum('kd,nk->nd',self.W,self.m_)
        if(flat): return value
        else:     return tf.reshape(value,self.input_shape)
    def sample(self,M=None,sigma=1,deterministic=False): return self.backward(self.is_flat)+tf.reshape(self.b+tf.random_normal((self.bs,self.D_in))*tf.sqrt(self.sigmas2_),self.input_shape)
    def likelihood(self,batch=False,pretraining=False):
        if(batch==False):
            k1  = -tf.reduce_sum(tf.log(self.sigmas2_))*float32(0.5)-tf.log(2*PI_CONST)*float32(self.D_in*0.5)     #(1)
            k2  = -tf.reduce_sum((tf.reshape(self.input_layer.v2,[self.bs,-1])+tf.square(self.input_-self.b_-tf.einsum('kd,nk->nd',self.W,self.m_)))/self.sigmas2)*0.5/self.bs
            k3  = -tf.einsum('k,kd,d->',self.v2_,tf.square(self.W),1/self.sigmas2_)*float32(0.5)            #(1)
            EXTRA = -tf.reduce_sum(tf.square(self.m_))*0.5/self.bs-tf.reduce_sum(self.v2_)*0.5
            return k1+k2+k3+EXTRA+0.5*tf.reduce_sum(tf.square(self.W))*self.sparsity_prior
        else:
            k1  = -tf.reduce_sum(tf.log(self.sigmas2_))*float32(0.5)-tf.log(2*PI_CONST)*float32(self.D_in*0.5)     #(1)
            k2  = -tf.reduce_sum((self.Sv2+self.Sm2+tf.square(self.b_)-2*self.Sm*self.b_)/self.sigmas2_)
            k3  = -tf.reduce_sum(tf.einsum('kd,cd,kc->d',self.W,self.W/self.sigmas2,self.SMM))*0.5
            k4  = -tf.einsum('d,kd,k->',self.b_/self.sigmas2_,self.W,self.SM)+tf.einsum('dk,kd->',self.SmM,self.W/self.sigmas2)-tf.einsum('k,kd,d->',self.SV2,tf.square(self.W),1/self.sigmas2_)*float32(0.5)
#            k5  = -tf.einsum('k,kd,d->',self.SV2,tf.square(self.W),1/self.sigmas2_)*float32(0.5)            #(1)
            EXTRA = -0.5*tf.reduce_sum(self.SM2+self.SV2)
            return k1+k2+k3+k4+EXTRA+0.5*tf.reduce_sum(tf.square(self.W))*self.sparsity_prior
    def update_Wk(self):
        numerator   = -self.b_*self.SM[self.k_]+self.SmM[:,self.k_]-tf.reduce_sum(self.W*tf.expand_dims(self.SMM[self.k_]*(1-tf.one_hot(self.k_,self.K)),-1),0)
        denominator = self.SV2[self.k_]+self.SMM[self.k_,self.k_]
        return tf.scatter_update(self.W,[self.k_],[numerator/(denominator+self.sparsity_prior)])
    def update_b(self):
        return tf.assign(self.b_,self.m-tf.einsum('kd,k->',self.W,self.M))
    def update_sigma(self):
        k2 = self.Sv2+self.Sm2+tf.square(self.b_)+tf.einsum('kd,cd,kc->d',self.W,self.W,self.SMM)
        k3 = tf.einsum('d,kd,k->d',self.b_,self.W,self.SM)-tf.einsum('dk,kd->d',self.SmM,self.W)-self.Sm*self.b_
        k4 = tf.einsum('k,kd->d',self.SV2,tf.square(self.W))
        value = k2+k3*2+k4
        if(self.sigma_opt=='local'):
            return tf.assign(self.sigmas2_,value)
        elif(self.sigma_opt=='global'):
            return tf.assign(self.sigmas2_,tf.ones_like(self.sigmas2_)*tf.reduce_sum(value)/(prod(self.input_shape[1:])))
    def evidence(self,b=0):
        return 0.
    def update_v2(self,b=0):
        rescalor = tf.reduce_min(self.sigmas2_)
        return tf.assign(self.v2_,rescalor/(rescalor+tf.einsum('kd,d->k',tf.square(self.W),rescalor/self.sigmas2_)))
    def update_m(self,b=0,bb=0):
        b = tf.einsum('nd,kd->nk',(self.input_-self.b)/self.sigmas2,self.W)
        A = tf.einsum('kd,cd->kc',self.W,self.W/self.sigmas2)+tf.eye(self.K)
        with tf.device('/device:CPU:0'):
            m = tf.transpose(tf.matrix_solve(A,tf.transpose(b)))
        return tf.assign(self.m_,m)
    def KL(self,pretraining=False): return self.likelihood(0)+float32(0.5)*tf.reduce_sum(tf.log(self.v2_+eps))#*self.mask)/float32(self.input_shape[0])



