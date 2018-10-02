import tensorflow as tf
from pylab import *
import utils
import itertools
from math import pi as PI_CONST


eps = float32(0.0000000000000001)

class BN:
    def __init__(self,center,scale):
	self.center = center
	self.scale  = scale

def mynorm(W,axis=None,keepdims=False):
    return tf.reduce_sum(tf.square(W),axis=axis,keepdims=keepdims)

 
def mysoftmax(W,axis=-1,coeff=0.00):
    input = W-tf.reduce_max(W,axis=axis,keepdims=True)
    delta = tf.exp(input)/tf.reduce_sum(tf.exp(input),axis=axis,keep_dims=True)
    if(coeff==0):
	return delta
    deltap = delta+coeff
    return deltap/tf.reduce_sum(deltap,axis=axis,keepdims=True)


def mynormalize(W,axis=-1):
    return W/tf.reduce_sum(W,axis=axis,keep_dims=True)


def myexpand(x,axis):
    nx = tf.expand_dims(x,axis[0])
    if(len(axis)==1):
        return nx
    else:
        return myexpand(nx,axis[1:])


def mysumpool(x,p):
    return tf.nn.avg_pool(x,[1,p[0],p[1],1],[1,p[0],p[1],1],'VALID')*p[0]*p[1]

#########################################################################################################################
#
#
#                                       DENSE/CONV/POOL LAYERS
#
#
#########################################################################################################################



class DenseLayer:
	def __init__(self,input_layer,K,R,sparsity_prior = 0,leakiness=None,sigma='local',p_drop=0,residual=0):
                self.leakiness      = leakiness
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
		self.drop_   = tf.Variable(tf.ones((self.K,2,self.bs))*tf.reshape(tf.one_hot(1,2),[1,2,1]))
                self.sigmas2_= tf.Variable(tf.ones(self.D_in),trainable=False)
                self.sigmas2 = tf.expand_dims(self.sigmas2_,0)
                if(leakiness is None):
		    self.W_  = tf.Variable(tf.random_normal((K,R,self.D_in),float32(0),float32(0.1)))
                    self.W   = self.W_
                else:
                    self.W_  = tf.Variable(tf.random_normal((K,1,self.D_in),float32(0),float32(0.0001)))
                    self.W   = tf.concat([self.W_,self.W_*self.leakiness],axis=1)
		self.pi      = tf.Variable(mysoftmax(tf.ones((K,R))))
		self.b_      = tf.Variable(tf.zeros(self.D_in))
		self.b       = tf.expand_dims(self.b_,0)
                self.V_      = tf.Variable(tf.zeros(self.K))
                self.V       = tf.expand_dims(self.V_,0)
                # VI PARAMETERS
		self.m_      = tf.Variable(tf.random_normal((K,self.bs))*0.1)
                self.m       = tf.transpose(self.m_)
                self.p_      = tf.Variable(mysoftmax(tf.random_normal((K,self.bs,R))*0.1,axis=2)) # convenient dimension ordering for fast updates shape: (D^{(\ell)},N,R^{(\ell)})
                self.p       = tf.transpose(self.p_,[1,0,2])                            # variable for $[p^{(\ell)}_n]_{d,r} of shape (N,D^{(\ell)},R^{(\ell)})$
                self.v2_     = tf.Variable(tf.ones((K,self.bs)))        # variable holding $[v^{(\ell)}_n]^2_{d}, \forall n,d$
                self.v2      = tf.transpose(self.v2_,[1,0])
		# placeholder for update and indices
                self.k_      = tf.placeholder(tf.int32)                                 # placeholder that will indicate which neuron is being updated
		self.W_indices = asarray(range(self.K))
		self.m_indices = self.W_indices
		self.p_indices = self.W_indices
		# RESHAPE IF NEEDED
                if(len(self.input_shape)>2):
                        self.is_flat = False
                        self.input_  = tf.reshape(self.input,(self.bs,self.D_in))
                        self.reshape_sigmas2_ = tf.reshape(self.sigmas2_,self.input_shape[1:])
                        self.reshape_sigmas2  = tf.expand_dims(self.reshape_sigmas2_,0)
                        self.reshape_b_      = tf.reshape(self.b_,self.input_shape[1:])
                        self.reshape_b       = tf.expand_dims(self.reshape_b_,0)
                else:
                        self.is_flat = True
                        self.input_  = self.input
        def init_thetaq(self):
                new_p       = tf.assign(self.p_,mysoftmax(tf.ones((self.K,self.bs,self.R)),axis=2))
                new_v       = tf.assign(self.v2_,tf.ones((self.K,self.bs)))
                new_m       = tf.assign(self.m_,0*tf.truncated_normal((self.K,self.bs),0,float32(1/sqrt(self.K))))
                return [new_m,new_p,new_v]
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
        def backwardk(self,k,resi = None):
		if(resi is None): resi = self.residual
		else: resi = self.residual*resi
		back = tf.einsum('knr,kr->n',self.p_*tf.expand_dims(self.m_*self.drop_[:,1],-1),self.W[:,:,k])
		if(resi): return back+self.V_[k]*self.m_[k] # (N)
		else:     return back                       # (N)
	def backwardmk(self,k,with_m=1,m=None,p=None):
		if(m is None): m = self.m_
		if(p is None): p = self.p_
		back = tf.tensordot(p*tf.expand_dims(m*self.drop_[:,1]*tf.expand_dims(1-tf.one_hot(k,self.K),-1),-1),self.W,[[0,2],[0,1]])
		if(self.residual==0): return back
		if(with_m):           return back+self.V*tf.transpose(m)
		else:                 return back+self.V*tf.transpose(m)*tf.expand_dims(1-tf.one_hot(k,self.K),0)
	def sample(self,M,K=None,sigma=1,deterministic=False):
		if(isinstance(self.input_layer,InputLayer)):sigma=0
		if(deterministic):    return tf.reshape(self.backward(1)+self.b,self.input_shape)
		noise = sigma*tf.random_normal((self.bs,self.D_in))*tf.sqrt(self.sigmas2)
		if(self.p_drop>0): dropout = tf.distributions.Bernoulli(probs=self.p_drop,dtype=tf.float32).sample((self.bs,self.K))
#                    dropout = tf.cast(tf.reshape(tf.multinomial(tf.reshape(tf.stack([tf.log(self.p_drop),tf.log(float32(1)-self.p_drop)]),[1,2]),self.bs*self.K),(self.bs,self.K)),tf.float32)
		else: dropout = float32(1)
                if(K==None):
		    K = tf.one_hot(tf.transpose(tf.multinomial(tf.log(self.pi),self.bs)),self.R)
#		    tf.Tensor.set_shape(K,[self.bs,self.K,self.R])
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
	def likelihood(self):
                if(self.residual):
                        currentv2 = -tf.einsum('kn,k->',self.v2_,tf.square(self.V_)/self.sigmas2_)
                        extra_v2  = -tf.einsum('kn,knr,rk,k->',self.v2_,self.p_,tf.map_fn(lambda r:tf.diag_part(self.W[:,r,:]),tf.range(self.R,dtype=tf.int32),dtype=tf.float32),self.V_/self.sigmas2_)*2
                else:   currentv2 = 0 ; extra_v2 = 0
                if(self.p_drop>0): k1  = tf.reduce_sum(self.drop_[:,0])*tf.log(self.p_drop)+tf.reduce_sum(self.drop_[:,1])*tf.log(float32(1)-self.p_drop)
                else:              k1 = 0
                k2  = -tf.reduce_sum(tf.log(self.sigmas2_))*float32(0.5)                            #(1)
                k3  = tf.einsum('knr,kr->',self.p_,tf.log(self.pi))*2                                #(1)
                k4  = -tf.log(2*PI_CONST)*float32(self.D_in*0.5)
                reconstruction    = -tf.einsum('nd,d->',tf.square(self.input_-self.b-self.backward(flat=1,resi=1)),1/self.sigmas2_)                   #(1)
                if(isinstance(self.input_layer,DenseLayer)):
                        input_v2  = -tf.einsum('dn,d->',self.input_layer.v2_,1/self.sigmas2_)  #(1)
                else:   input_v2  = -tf.reduce_sum(tf.reshape(tf.reduce_sum(self.input_layer.v2,0),[self.D_in])/self.sigmas2_) #(1)
                sum_norm_k      = tf.einsum('kdn,d->',tf.square(tf.einsum('krd,knr,kn->kdn',self.W,self.p_,self.m_*self.drop_[:,1])),1/self.sigmas2_)
                m2v2   = -tf.reduce_sum(self.drop_[:,1]*(self.v2_+tf.square(self.m_))*tf.einsum('knr,krd,d->kn',self.p_,tf.square(self.W),1/self.sigmas2_))
                return k4+k2+(k1+k3+m2v2+reconstruction+input_v2+sum_norm_k+currentv2+extra_v2)*float32(0.5/self.bs)-self.sparsity_prior*tf.reduce_sum(tf.square(self.W))
        def KL(self):
		if(self.p_drop>0):
                    return self.likelihood()+(-tf.reduce_sum(self.p_*tf.log(self.p_+eps))+tf.reduce_sum(tf.log(self.v2_+eps))*float32(0.5)\
				-tf.reduce_sum(self.drop_*tf.log(self.drop_+eps)))/float32(self.bs)+float32(self.D_in/2.0)*tf.log(2*PI_CONST)
		else:
                    return self.likelihood()+(-tf.reduce_sum(self.p_*tf.log(self.p_+eps))+tf.reduce_sum(tf.log(self.v2_+eps))*float32(0.5))/float32(self.bs)+float32(self.D_in/2.0)*tf.log(2*PI_CONST)
#                                           ----      UPDATES      -----
        def update_v2(self):
		a40 = self.drop_[:,1]*tf.einsum('knr,krd,d,k->kn',self.p_,tf.square(self.W),1/self.sigmas2_,self.next_layer.sigmas2_)+1
		if(self.residual==0):  a4 = a40
		else:                  a4 = a40+tf.expand_dims(tf.square(self.V_)*self.next_layer.sigmas2_/self.sigmas2_,-1)+2*tf.einsum('knr,rk,k->kn',self.p_,tf.map_fn(lambda r:tf.diag_part(self.W[:,r,:]),tf.range(self.R),dtype=tf.float32),self.V_*self.next_layer.sigmas2_/self.sigmas2_)
		return tf.assign(self.v2_,tf.expand_dims(self.next_layer.sigmas2_,-1)/a4)
	def update_rho(self):
		return []
        def update_m(self,ii):
                k  = self.k_
		pk = self.p_[self.k_];tf.Tensor.set_shape(pk,[self.bs,self.R])
		Wk = self.W[self.k_];tf.Tensor.set_shape(Wk,[self.R,self.D_in])
		if(ii==0):
	                reconstruction = self.input_-self.b-self.backwardmk(self.k_,with_m=0);tf.Tensor.set_shape(reconstruction,[self.bs,self.D_in]) # (N D)
			## UPDATE M
			proj           = tf.einsum('nd,rd,nr->n',reconstruction,Wk/self.sigmas2,pk)*self.drop_[self.k_,1]
			if(self.residual): extra_proj = reconstruction[:,self.k_]*self.V_[self.k_]/self.sigmas2_[self.k_] #(N)
			else:              extra_proj = 0
	                priorm  = (self.next_layer.backwardk(self.k_)+self.next_layer.b_[self.k_])/self.next_layer.sigmas2_[k]#(N)
			denom  = self.drop_[self.k_,1]*tf.einsum('nr,rd,d->n',self.p_[self.k_],tf.square(Wk),1/self.sigmas2_)+1/self.next_layer.sigmas2_[self.k_]
			if(self.residual): extra_denom = (tf.square(self.V_[self.k_])+2*self.V_[self.k_]*tf.einsum('nr,r->n',self.p_[self.k_],self.W[self.k_,:,self.k_]))/self.sigmas2_[self.k_]
			else:              extra_denom = 0
	     		new_m = tf.scatter_update(self.m_,[self.k_],[(priorm+proj+extra_proj)/(denom+extra_denom)])
			return new_m
#		with tf.control_dependencies([new_m]):
		new_m = self.m_
		reconstruction = self.input_-self.b-self.backwardmk(self.k_,with_m=1);tf.Tensor.set_shape(reconstruction,[self.bs,self.D_in])
#			if(self.residual): reconstruction = self.input_-self.b-self.backwardmk(self.k_,with_m=1,m=new_m);tf.Tensor.set_shape(reconstruction,[self.bs,self.D_in]) # (N D)
                forward = tf.einsum('nd,rd,n->nr',reconstruction,Wk/self.sigmas2,new_m[self.k_]*self.drop_[self.k_,1])                        # (N R)
                prior   = tf.expand_dims(tf.log(self.pi[self.k_]),0)                                                                       # (1 R)
                m2v2    = tf.einsum('n,rd,d->nr',(tf.square(new_m[self.k_])+self.v2_[self.k_])*self.drop_[self.k_,1],tf.square(Wk),1/(2*self.sigmas2_)) # (N R)
		if(self.residual):
		    extra_v2 = tf.einsum('n,r->nr',self.v2_[self.k_],self.W[self.k_,:,self.k_]*self.V_[self.k_]/self.sigmas2_[self.k_])
		else: extra_v2 = 0
                new_p   = tf.scatter_update(self.p_,[self.k_],[mysoftmax((forward+prior-m2v2-extra_v2),coeff=eps)])
		return new_p#tf.group([new_m,new_p])
		if(self.p_drop==0):
		    return tf.group(new_m,new_p)
		## UPDATE DROPOUT
                proj    = tf.einsum('nd,rd,nr->n',reconstruction,self.W[k]/self.sigmas2,new_p[self.k_])*new_m[self.k_] # (N)
                squared = tf.einsum('n,nr,rd,d->n',self.v2_[self.k_]+tf.square(new_m[self.k_]),new_p[self.k_],tf.square(self.W[k]),1/self.sigmas2_) #(N)
                filled0 = tf.fill([self.bs],tf.cast(tf.log(self.p_drop),tf.float32))
                filled1 = tf.fill([self.bs],tf.cast(tf.log(1-self.p_drop),tf.float32))
                new_drop = mysoftmax(tf.stack([filled0,filled1-squared*float32(0.5)+proj]),axis=0,coeff=0.)
                return tf.group(new_m,new_p,tf.scatter_update(self.drop_,[k],[new_drop]))
        def update_sigma(self):              
                rec  = tf.reduce_sum(tf.square(self.input_-self.b-self.backward(flat=1,resi=1)),0)/self.bs
		if(self.residual): 
		    v20 = tf.reshape(tf.reduce_sum(self.input_layer.v2,0)/self.bs,[self.D_in])+tf.reduce_sum(self.v2_*tf.expand_dims(tf.square(self.V_),-1),1)/self.bs
		    v2  = v20+2*tf.diag_part(tf.einsum('kn,knr,krd,d->kd',self.v2_,self.p_,self.W,self.V_))/self.bs
		else:   v2  = tf.reshape(tf.reduce_sum(self.input_layer.v2,0)/self.bs,[self.D_in])
		m2v2       = tf.einsum('kr,krd->d',tf.einsum('kn,knr->kr',(self.v2_+tf.square(self.m_))*self.drop_[:,1],self.p_),tf.square(self.W))/self.bs # ( D)
		sum_norm_k = -tf.reduce_sum(tf.square(tf.einsum('krd,knr,kn->knd',self.W,self.p_,self.m_*self.drop_[:,1])),[0,1])/self.bs  #(D)
                value   = rec+sum_norm_k+v2+m2v2
                if(self.sigma_opt=='local'):
                    return tf.assign(self.sigmas2_,value)
		elif(self.sigma_opt=='global'):
		    return tf.assign(self.sigmas2_,tf.fill([self.D_in],tf.reduce_sum(value)/self.D_in))
		elif(self.sigma_opt=='channel'):
		    v=tf.reduce_sum(tf.reshape(value,self.input_shape[1:]),axis=[0,1],keepdims=True)/(self.input_shape[1]*self.input_shape[2])
		    return tf.assign(self.sigmas2_,tf.reshape(tf.ones([self.input_shape[1],self.input_shape[2],1])*v,[self.D_in]))
        def update_pi(self):
                a44     = tf.reduce_sum(self.p,axis=0)/self.bs
                return tf.assign(self.pi,a44/tf.reduce_sum(a44,axis=1,keepdims=True))
        def update_Wk(self):
                k = self.k_
		a = self.input_-self.b-self.backwardmk(k,with_m=1) # (N D)
                if(self.residual):ff = -tf.expand_dims(tf.einsum('n,nr->r',self.v2_[self.k_]*self.V_[self.k_],self.p_[self.k_]),-1)*tf.expand_dims(tf.one_hot(self.k_,self.K),0)
                else:             ff = 0
                if(self.leakiness is None):
                    numerator   = tf.tensordot(tf.expand_dims(self.m_[self.k_]*self.drop_[self.k_,1],-1)*self.p_[self.k_],a,[[0],[0]])+ff      # (R D)
                    denominator = tf.tensordot((tf.square(self.m_[self.k_])+self.v2_[self.k_])*self.drop_[self.k_,1],self.p_[self.k_],[[0],[0]]) # (R)
#                    new_w       = tf.scatter_update(self.W_,[k],[numerator[:1]/tf.sqrt(tf.nn.l2_loss(numerator[:1])*2),numerator[1:]/(tf.expand_dims(denominator[1:],-1))],axis=0)])
                    new_w       = tf.scatter_update(self.W_,[self.k_],[numerator/(self.sparsity_prior+tf.expand_dims(denominator,-1))])
                else:
                    numerator   = tf.tensordot( self.m_[self.k_]*self.drop_[self.k_,1]*(self.p_[self.k_,:,0]+self.p_[self.k_,:,1]*self.leakiness),a,[[0],[0]])+ff # (D)
                    denominator = tf.reduce_sum((tf.square( self.m_[self.k_])+self.v2_[self.k_])*self.drop_[self.k_,1]*(self.p_[self.k_,:,0]+(self.leakiness**2)*self.p_[k,:,1]),0) #(1)
#		    rescaling   = tf.maximum(tf.sqrt(tf.nn.l2_loss(numerator)*2),0)#self.sparsity_prior+denominator)
#                    new_w       = tf.scatter_update(self.W_,[k],[tf.expand_dims(numerator/(self.sparsity_prior+denominator),0)])
                    new_w       = tf.scatter_update(self.W_,[self.k_],[tf.expand_dims(numerator/(self.sparsity_prior+denominator),0)])
                return new_w
	def update_BV(self):
#		return []
		newb = tf.assign(self.b_,tf.reduce_sum(self.input_-self.backward(1),0)/self.bs)
		if(self.residual==0): return newb
		back = tf.einsum('knr,krd->nd',self.p_*tf.expand_dims(self.m_*self.drop_[:,1],-1),self.W)  # (N D)
		forward = tf.einsum('kn,nk->k',self.m_,self.input_-tf.expand_dims(newb,0)-back)-tf.diag_part(tf.einsum('kn,knr,krd->kd',self.v2_,self.p_,self.W)) # K
		b = tf.reduce_sum(self.v2_+tf.square(self.m_),1)
		return tf.group(newb,tf.assign(self.V_,forward/b))



class ConvPoolLayer:
	def __init__(self,input_layer,K,Ic,Jc,Ir,Jr,R,sparsity_prior = 0,leakiness=None,sigma='local',p_drop=0):
                self.leakiness         = leakiness
		self.sigma_opt         = sigma
		self.p_drop            = p_drop
                self.sparsity_prior    = sparsity_prior
                self.input_layer       = input_layer
                input_layer.next_layer = self
                self.bs,self.Iin,self.Jin,self.C  = input_layer.output_shape 
                self.Ic,self.Jc,self.Ir,self.Jr,self.K,self.R     = Ic,Jc,Ir,Jr,K,R
                self.input             = input_layer.m
                self.input_shape       = input_layer.output_shape
                self.conv_output_shape = (self.bs,self.input_shape[-3]-self.Ic+1,self.input_shape[-2]-self.Jc+1,K)
		self.output_shape      = (self.conv_output_shape[0],self.conv_output_shape[1]/Ir,self.conv_output_shape[2]/Jr,self.conv_output_shape[3])
		self.D_in              = prod(self.input_shape[1:])
                self.I,self.J          = self.output_shape[1],self.output_shape[2]
                self.input_small_patch = self.extract_small_patch(self.input,with_n=1)
                self.input_large_patch = self.extract_large_patch(self.input,with_n=1)
                self.pi_                 = PI_CONST
                if(leakiness is None):
		    init     = tf.random_normal_initializer(0,0.1)#tf.orthogonal_initializer()
		    self.W_  = tf.Variable(init([self.K,self.R,self.Ic,self.Jc,self.C]))
                    self.W   = self.W_
                elif(leakiness=='relu'):
                    self.W_  = tf.Variable(tf.truncated_normal((self.K,1,self.Ic,self.Jc,self.C),0,1))# (K,R,Ic,Jc,C) always D_in last
                    self.W   = tf.concat([self.W_,tf.zeros_like(self.W_)],axis=1)
                elif(leakiness=='abs'):
                    self.W_  = tf.Variable(tf.random_normal((self.K,1,self.Ic,self.Jc,self.C)))# (K,R,Ic,Jc,C) always D_in last
                    self.W   = tf.concat([self.W_,-self.W_],axis=1) 
		# WE DEFINE THE PARAMETERS
                self.pi      = tf.Variable(mysoftmax(tf.random_normal((K,R)),axis=1))
		self.sigmas2_= tf.Variable(tf.ones((self.Iin,self.Jin,self.C)))
		self.sigmas2 = tf.expand_dims(self.sigmas2_,0)
                self.sigmas2_large_patch_= self.extract_large_patch(self.sigmas2,with_n=0)
		self.sigmas2_large_patch = tf.expand_dims(self.sigmas2_large_patch_,0)
                self.sigmas2_small_patch_= self.extract_small_patch(self.sigmas2,with_n=0)
		self.sigmas2_small_patch = tf.expand_dims(self.sigmas2_small_patch_,0)
		###############################   WE DEFINE SOME HELPER VARIABLES FOR LATER FUNCTIONS ###############################################
		self.multiplied_x = tf.reshape(tf.extract_image_patches(tf.transpose(tf.reshape(tf.transpose(self.input_large_patch,[3,4,5,0,1,2]),[self.Ic+self.Ir-1,self.Jc+self.Jr-1,self.C,self.bs*self.I*self.J]),[3,0,1,2]),(1,self.Ic,self.Jc,1),(1,1,1,1),(1,1,1,1),"VALID"),[self.bs,self.I,self.J,self.Ir,self.Jr,self.Ic,self.Jc,self.C])
		#####
                self.dx       = tf.zeros((self.bs,self.conv_output_shape[1],self.conv_output_shape[2],1)) # (N I' J' 1)
                self.dxR     = tf.zeros((self.bs,self.conv_output_shape[1],self.conv_output_shape[2],self.R)) # (N I' J' R)
                self.dxK     = tf.zeros((self.bs,self.conv_output_shape[1],self.conv_output_shape[2],self.K)) # (N I' J' K)
		#
                self.dp      = tf.reshape(tf.extract_image_patches(self.dx,(1,Ir,Jr,1),(1,Ir,Jr,1),(1,1,1,1),"VALID"),(self.bs,self.I,self.J,Ir,Jr)) # (N I J Ir Jr)
                self.dpK     = tf.reshape(tf.extract_image_patches(self.dxK,(1,Ir,Jr,1),(1,Ir,Jr,1),(1,1,1,1),"VALID"),(self.bs,self.I,self.J,Ir,Jr,K)) # (N I J Ir Jr K)
                self.dpR     = tf.reshape(tf.extract_image_patches(self.dxR,(1,Ir,Jr,1),(1,Ir,Jr,1),(1,1,1,1),"VALID"),(self.bs,self.I,self.J,Ir,Jr,self.R)) # (N I J Ir Jr K)
		#
		self.dpK_pool= mysumpool(self.dxK,[self.Ir,self.Jr]) # (N I J K)
                self.dpR_pool= mysumpool(self.dxR,[self.Ir,self.Jr]) # (N I J K)
                self.dp_pool = mysumpool(self.dx,[self.Ir,self.Jr]) # (N I J 1)
		#
		self.r_large_patch=tf.transpose(tf.reshape(tf.transpose(self.input_large_patch,[3,4,5,0,1,2]),[self.Ic+self.Ir-1,self.Jc+self.Jr-1,self.C,self.bs*self.I*self.J]),[3,0,1,2]) #(NIJ,??,??,C)
		self.r_small_patch=tf.reshape(tf.extract_image_patches(self.r_large_patch,(1,Ic,Jc,1),(1,1,1,1),(1,1,1,1),"VALID"),(self.bs*self.I*self.J,Ir,Jr,self.Ic,self.Jc,self.C))
		# SOME OTHER VARIABLES
		self.b_      = tf.Variable(tf.zeros(self.input_shape[1:]))
		self.b       = tf.expand_dims(self.b_,0)
		self.m_      = tf.Variable(tf.zeros((K,self.I,self.J,self.bs)))
                self.m       = tf.transpose(self.m_,[3,1,2,0])   # (N I J K)
		self.p_      = tf.Variable(mysoftmax(tf.random_normal((K,self.I,self.J,self.R,self.bs)),axis=3))# (K,I,J,R,N)
                self.p       = tf.transpose(self.p_,[4,1,2,0,3]) # (N I J K R)
		self.p_expand= self.unpool_R(self.p)
		self.rho_    = tf.Variable(mysoftmax(tf.random_normal((K,self.I,self.J,self.Ir,self.Jr,self.bs)),axis=[3,4])) # (K I J Ir Jr N)
                self.rho_tensor = self.assemble_pool_patch(tf.transpose(self.rho_,[5,1,2,3,4,0]),self.K) # (N I' J' K)
                self.v2_     = tf.Variable(tf.ones((self.K,self.I,self.J,self.bs))) # (K I J N)
                self.v2      = tf.transpose(self.v2_,[3,1,2,0])
		self.drop_   = tf.Variable(tf.ones((K,2,self.I,self.J,self.bs))*tf.reshape(tf.one_hot(1,2),(1,2,1,1,1))) # (K 2 I J N)
                self.drop    = tf.transpose(self.drop_,[1,4,2,3,0]) #(2 N I J K)
		#
                input_layer.next_layer = self
                self.k_      = tf.placeholder(tf.int32)
                self.i_      = tf.placeholder(tf.int32)
                self.j_      = tf.placeholder(tf.int32)
                self.r_      = tf.placeholder(tf.int32)
		self.ratio   = (Ic-1)/Ir+1
                self.Ni      = tf.cast(tf.ceil((self.I-tf.cast(self.i_,tf.float32))/self.ratio),tf.int32) # NUMBER OF TERMS
                self.Nj      = tf.cast(tf.ceil((self.J-tf.cast(self.j_,tf.float32))/self.ratio),tf.int32) # NUMBER OF TERMS
                self.xi,self.yi= tf.meshgrid(tf.range(self.j_,self.J,self.ratio),tf.range(self.i_,self.I,self.ratio)) # THE SECOND IS CONSTANT (meshgrid)
                self.indices_= tf.concat([tf.fill([self.Ni*self.Nj,1],self.k_),tf.reshape(self.yi,(self.Ni*self.Nj,1)),tf.reshape(self.xi,(self.Nj*self.Ni,1))],axis=1) # (V 3) indices where the 1 pops
		if(self.leakiness is None):
			self.W_indices = [a for a in itertools.product(range(self.K),range(self.R),range(self.Ic),range(self.Jc))]
		else:
                        self.W_indices = [a for a in itertools.product(range(self.K),range(1),range(self.Ic),range(self.Jc))]
		self.W_indices = asarray(self.W_indices)
		self.indices   = asarray([a for a in itertools.product(range(self.ratio),range(self.ratio),range(self.K))])
                self.m_indices = self.indices
		self.tf_indices= tf.Variable(asarray(self.indices).astype('int32'),trainable=False)
		mask           = tf.reshape(tf.one_hot(self.k_,self.K),(self.K,1,1,1))*tf.reshape(tf.tile(tf.one_hot(self.i_,self.ratio),[(self.I/self.ratio+1)]),(1,(self.I/self.ratio+1)*self.ratio,1,1))*tf.reshape(tf.tile(tf.one_hot(self.j_,self.ratio),[self.J/self.ratio+1]),(1,1,(self.J/self.ratio+1)*self.ratio,1))
		self.mask      = mask[:,:self.I,:self.J] # (K I J 1)
         ###  first to go from the pooling patches to the conv output for (i) one filter k and (ii) all filters K
	def extract_small_patch(self,u,with_n=1,with_reshape=1):
	    patches = tf.extract_image_patches(u,(1,self.Ic,self.Jc,1),(1,1,1,1),(1,1,1,1),"VALID")
	    if(with_reshape):
		if(with_n): return tf.reshape(patches,(self.bs,self.conv_output_shape[1],self.conv_output_shape[2],self.Ic,self.Jc,self.C))
		else:       return tf.reshape(patches,(self.conv_output_shape[1],self.conv_output_shape[2],self.Ic,self.Jc,self.C))
	    return patches
        def extract_large_patch(self,u,with_n=1,with_reshape=1):
	    patches = tf.extract_image_patches(u,(1,self.Ir+self.Ic-1,self.Jr+self.Jc-1,1),(1,self.Ir,self.Jr,1),(1,1,1,1),"VALID")
	    if(with_reshape):
		if(with_n): return tf.reshape(patches,(self.bs,self.I,self.J,self.Ir+self.Ic-1,self.Jr+self.Jc-1,self.C))
		else:       return tf.reshape(patches,(self.I,self.J,self.Ir+self.Ic-1,self.Jr+self.Jc-1,self.C))
	    return patches
	def assemble_small_patch(self,u):
	    return tf.gradients(self.input_small_patch,self.input,u)[0]
        def assemble_large_patch(self,u):
            return tf.gradients(self.input_large_patch,self.input,u)[0]
	def assemble_pool_patch(self,u,k=1):
            # the following takes as input  (N I J Ir Jr)  returns (N I' J' 1) or (N I J Ir Jr K) returns (N I' J' K)
	    if(k==1):      return tf.gradients(self.dp,self.dx,u)[0]
            if(k==self.R): return tf.gradients(self.dpR,self.dxR,u)[0]
            else :         return tf.gradients(self.dpK,self.dxK,u)[0]
	def unpool(self,u,k=1):
	    # takes (N I J 1) returns (N I' J' 1) or takes (N I J R) returns (N I' J' R) or takes (N I J K) returns (N I' J' K)
            if(k==1) :       return tf.gradients(self.dp_pool,self.dx,tf.expand_dims(u,-1))[0] 
            elif(k==self.R) :return tf.gradients(self.dpR_pool,self.dxR,u)[0]
            elif(k==self.K) :return tf.gradients(self.dpK_pool,self.dxK,u)[0]
	def unpool_R(self,u):# takes (N I J K R) returns (N I' J' K R)
	    return tf.transpose(tf.map_fn(lambda i:self.unpool(i,self.K),tf.transpose(u,[4,0,1,2,3]),back_prop=False),[1,2,3,4,0])
        def normwithsigma_bigpatch_k(self,k): #helper
            Wflat   = self.get_bigpatch_k(k) # (N I J ?? ?? C)
            return tf.einsum('nijabc,ijabc->',tf.square(Wflat),1/(2*self.bs*self.sigmas2_large_patch_))
        def get_bigpatch_k(self,k): #helper_
	    mrhop = tf.einsum('ijn,ijpqn,ijrn,rabc->nijpqabc',self.m_[k]*self.drop_[k,1],self.rho_[k],self.p_[k],self.W[k])# (N I J Ir Jr Ic Jc C)
            return tf.gradients(self.multiplied_x,self.input_large_patch,mrhop)[0] #(N I J ?? ?? C)     # (N I J ?? ?? C)
#                                           ----  INITIALIZER    ----
        def update_BV(self):
            return tf.assign(self.b_,tf.reduce_sum(tf.reduce_sum(self.input-self.deconv(),0)/(self.bs*self.sigmas2_),[0,1],keepdims=True)*tf.ones((self.Iin,self.Jin,1))/tf.reduce_sum(1/self.sigmas2_,[0,1],keepdims=True))
        def init_thetaq(self):
            new_p       = tf.assign(self.p_,mysoftmax(tf.random_uniform((self.K,self.I,self.J,self.R,self.bs)),axis=3))
	    new_rho     = tf.assign(self.rho_,mysoftmax(tf.random_uniform((self.K,self.I,self.J,self.Ir,self.Jr,self.bs)),axis=[3,4]))
            new_v       = tf.assign(self.v2_,tf.ones((self.K,self.I,self.J,self.bs)))
            new_m       = tf.assign(self.m_,tf.random_normal((self.K,self.I,self.J,self.bs)))
            return [new_m,new_p,new_v,new_rho]
#                                           ---- BACKWARD OPERATOR ---- 
        def deconv(self,input=None,masked_m=0,masked_w=0,m=None,p=None):
		if(m is None):m=self.m_
		if(p is None):p=self.p_
		if(input is not None):
                    return self.small_patch2tensor(input)
                if(masked_w):
		    if(self.leakiness is None):
                        mask  = tf.reshape(tf.one_hot(self.i_,self.Ic),[1,1,self.Ic,1,1])*tf.reshape(tf.one_hot(self.j_,self.Jc),[1,1,1,self.Jc,1])*tf.reshape(tf.one_hot(self.k_,self.K),[self.K,1,1,1,1])*tf.reshape(tf.one_hot(self.r_,self.R),[1,self.R,1,1,1])
		    else:
			mask  = tf.reshape(tf.one_hot(self.i_,self.Ic),[1,1,self.Ic,1,1])*tf.reshape(tf.one_hot(self.j_,self.Jc),[1,1,1,self.Jc,1])*tf.reshape(tf.one_hot(self.k_,self.K),[self.K,1,1,1,1])
		    W=self.W*(1-mask)
		else:
		    W=self.W
                p_w   = tf.einsum('nijkr,krabc->nijkabc',self.unpool_R(tf.transpose(p,[4,1,2,0,3])),W) # (N I' J' K Ic Jc C)
		if(masked_m==1):
                    m_rho = self.assemble_pool_patch(tf.transpose(self.rho_*myexpand(m*self.drop_[:,1]*(1-self.mask),[-2,-2]),[5,1,2,3,4,0]),self.K)# (N I J Ir JR K)->(N I' J' K)
		elif(masked_m==-1):
                    m_rho = self.assemble_pool_patch(tf.transpose(self.rho_[self.k_]*myexpand(self.drop_[self.k_,1]*self.mask[self.k_],[2,3]),[4,0,1,2,3]),1)# (N I' J' 1)
		    return self.assemble_small_patch(myexpand(m_rho,[-1,-1])*p_w[:,:,:,self.k_])
                else:
                    m_rho = self.assemble_pool_patch(tf.transpose(self.rho_*myexpand(m*self.drop_[:,1],[-2,-2]),[5,1,2,3,4,0]),self.K)# (N I' J' K)
		if(masked_w):
		    return self.assemble_small_patch(tf.einsum('nijk,nijkabc->nijabc',m_rho,p_w)),W
                return self.assemble_small_patch(tf.einsum('nijk,nijkabc->nijabc',m_rho,p_w))
	def sample(self,M,K=None,sigma=1):
		#multinomial returns [K,n_samples] with integer value 0,...,R-1
		if(isinstance(self.input_layer,InputLayer)):sigma=0
		noise            = sigma*tf.random_normal(self.input_shape)*tf.sqrt(self.sigmas2)
                sigma_hot        = tf.one_hot(tf.reshape(tf.multinomial(tf.log(self.pi),self.bs*self.I*self.J),(self.K,self.I,self.J,self.bs)),self.R) # (K I J N R)
		sigma_hot_reshape= self.unpool_R(tf.transpose(sigma_hot,[3,1,2,0,4])) # (R N I' J' K) -> (N I' J' K R)
		sigma_hot_w      = tf.einsum('nijkr,krabc->nijkabc',sigma_hot_reshape,self.W) # (N I' J' K Ic Jc C)
                pool_hot         = tf.one_hot(tf.reshape(tf.multinomial(tf.zeros((1,self.Ir*self.Jr)),self.K*self.bs*self.I*self.J),(self.K,self.I,self.J,self.bs)),self.Ir*self.Jr) # (K I J N IrxJr)
		pool_hot_reshape = tf.transpose(tf.reshape(pool_hot,[self.K,self.I,self.J,self.bs,self.Ir,self.Jr]),[3,1,2,4,5,0]) # (N I J Ir Jr K)
		reverted         = self.assemble_pool_patch(myexpand(M,[-2,-2])*pool_hot_reshape,self.K) # (N I' J' K)
                return self.assemble_small_patch(tf.einsum('nijkabc,nijk->nijabc',sigma_hot_w,reverted))+noise+self.b
        def likelihood(self):
                a1  = -tf.einsum('nijc,ijc->',tf.square(self.input-self.deconv()-self.b),1/(2*self.sigmas2_*self.bs))
                a2  = -tf.einsum('cijn,ijc->',self.input_layer.v2_,1/(2*self.sigmas2_*self.bs))
                a3  = tf.reduce_sum(tf.map_fn(self.normwithsigma_bigpatch_k,tf.range(self.K),dtype=tf.float32,back_prop=False))
		a40 = tf.einsum('krijc,abijc->krab',tf.square(self.W),1/(2*self.sigmas2_small_patch_*self.bs)) #(K R I' J')
		a41 = tf.einsum('nijkr,krij->nijk',self.p_expand,a40)*self.rho_tensor # (N I' J' K)
                a4  = -tf.einsum('nijk,kijn->',mysumpool(a41,[self.Ir,self.Jr]),(tf.square(self.m_)+self.v2_)*self.drop_[:,1])
                k1  = -tf.reduce_sum(tf.log(self.sigmas2_))/2
                k2  = tf.einsum('kr,krn->',tf.log(self.pi),tf.reduce_sum(self.p_,[1,2]))/float32(self.bs)
		k3  = -float32(self.D_in/2.0)*tf.log(2*PI_CONST)/float32(self.bs)
                return k1+k2+k3+(a1+a2+a3+a4)-self.sparsity_prior*tf.reduce_sum(tf.square(self.W))
        def KL(self):
                return self.likelihood()+(-tf.reduce_sum(self.p_*tf.log(self.p_))-tf.reduce_sum(self.rho_*tf.log(self.rho_))\
				+float32(self.D_in/2.0)*tf.log(2*PI_CONST)+float32(0.5)*tf.reduce_sum(tf.log(self.v2_)))/float32(self.bs)
	def update_dropout(self):
		return []
        def update_v2(self):# DONE
                if(isinstance(self.next_layer,ConvPoolLayer)): v_value = self.next_layer.sigmas2 # (N I J K)
                else:                                          v_value = self.next_layer.reshape_sigmas2 # (N I J K)
                a40 = tf.einsum('nijkr,krabc,ijabc,nijk->nijk',self.unpool_R(self.p),tf.square(self.W),1/self.sigmas2_small_patch_,self.rho_tensor) # (N I' J' K)
                a4  = mysumpool(a40,[self.Ir,self.Jr])*self.drop[1] #( N I J K)
                return tf.assign(self.v2_,tf.transpose(v_value/(1+a4*v_value),[3,1,2,0]))
	def update_m(self):
		reconstruction = (self.input-self.b-self.deconv(None,1,0))/self.sigmas2
		Wk    = self.W[self.k_]
		### UPDATE M
		forward_m   = tf.reduce_sum(self.extract_large_patch(reconstruction*self.deconv(None,-1,0),with_reshape=0),3) #  (N I J ??) -> (N I J)
                if(isinstance(self.next_layer,ConvPoolLayer)): back = (self.next_layer.deconv()[:,:,:,self.k_]+self.next_layer.b[:,:,:,self.k_])/self.next_layer.sigmas2[:,:,:,self.k_]
		else:       back = (self.next_layer.backward(0)[:,:,:,self.k_]+self.next_layer.reshape_b[:,:,:,self.k_])/self.next_layer.reshape_sigmas2[:,:,:,self.k_]
		update_value_m = (forward_m+back)*self.v2[:,:,:,self.k_] # (N I'' J'')
                new_m = tf.scatter_nd_update(self.m_,self.indices_,tf.transpose(tf.reshape(update_value_m[:,self.i_::self.ratio,self.j_::self.ratio],(self.bs,-1))))
		new_m_k = new_m[self.k_]
		### UPDATE P
#               PROJECTION
		m_rho = self.unpool(tf.transpose(new_m_k*self.drop_[self.k_,1]*self.mask[self.k_],[2,0,1]),1)[:,:,:,0]*self.rho_tensor[:,:,:,self.k_]# (N I' J')
		deconv_with2   = tf.map_fn(lambda w:tf.reduce_sum(self.extract_large_patch(self.assemble_small_patch(tf.einsum('nij,abc->nijabc',m_rho,w))*reconstruction,with_reshape=0)[:,self.i_::self.ratio,self.j_::self.ratio],axis=3),Wk,dtype=tf.float32,back_prop=False) # (R N I'' J'')
#		M2V2
                a40  = tf.einsum('rabc,ijabc,nij->nijr',tf.square(Wk),0.5/self.sigmas2_small_patch_,self.rho_tensor[:,:,:,self.k_]) #(N I' J' R)
		a4   = tf.einsum('nijr,ijn->rnij',mysumpool(a40,[self.Ir,self.Jr]),tf.square(new_m_k)+self.v2_[self.k_])[:,:,self.i_::self.ratio,self.j_::self.ratio] # (R N I'' J'')
                update_value = mysoftmax(deconv_with2-a4+myexpand(tf.log(self.pi[self.k_]),[-1,-1,-1]),axis=0,coeff=eps)# (R N I'' J'')
                new_p        = tf.scatter_nd_update(self.p_,self.indices_,tf.transpose(tf.reshape(update_value,(self.R,self.bs,-1)),[2,0,1]))
		new_p_k = new_p[self.k_]
		### UPDATE RHO
#               PROJECTION
                mp_reverted = self.unpool(tf.einsum('ijn,ijrn->nijr',self.drop_[self.k_,1]*new_m_k,new_p_k),self.R) # (N I' J' R)
		proj2       = tf.einsum('nijr,rabc,nijabc->nij',mp_reverted,Wk,self.extract_small_patch(reconstruction)) # (N I' J')
#		M2V2
		a43 = tf.einsum('ijn,ijrn->nijr',self.drop_[self.k_,1]*(tf.square(new_m_k)+self.v2_[self.k_]),new_p_k) # (N I J R)
                a44 = tf.einsum('nijr,rabc,ijabc->nij',self.unpool(a43,self.R),tf.square(self.W[self.k_]),0.5/self.sigmas2_small_patch_) # (N I' J')
                V   = tf.reshape(tf.extract_image_patches(tf.expand_dims(proj2-a44,-1),(1,self.Ir,self.Jr,1),(1,self.Ir,self.Jr,1),(1,1,1,1),"VALID"),
                                                                (self.bs,self.I,self.J,self.Ir,self.Jr)) # (N I J Ir Jr)
                update_value_rho = mysoftmax(V[:,self.i_::self.ratio,self.j_::self.ratio],axis=[3,4],coeff=eps)# (N I'' J'' Ir Jr)
                new_rho = tf.scatter_nd_update(self.rho_,self.indices_,tf.transpose(tf.reshape(tf.transpose(update_value_rho,[3,4,0,1,2]),(self.Ir,self.Jr,self.bs,-1)),[3,0,1,2]))
		return tf.group(new_m,new_p,new_rho)
        def update_Wk(self):
#		return []
		i = self.i_
		j = self.j_
                k = self.k_
		r = self.r_
		## FIRST COMPUTE DENOMINATOR
		cropped_sigma = self.sigmas2_[self.i_:self.input_shape[1]-self.Ic+1+self.i_,self.j_:self.input_shape[2]-self.Jc+1+self.j_]
		## NOW COMPUTE NUMERATOR
		deconv,masked_W = self.deconv(None,0,1)
                reconstruction  = self.input-self.b-deconv # (N Iin Jin C)
                cropped_reconstruction = reconstruction[:,self.i_:self.input_shape[1]-self.Ic+1+self.i_,self.j_:self.input_shape[2]-self.Jc+1+self.j_] # (N I'' J'' C)
                if(self.leakiness is None):
                    m2v2_augmented   = self.unpool(tf.transpose((tf.square(self.m_[self.k_])+self.v2_[self.k_])*self.p_[self.k_,:,:,self.r_]*self.drop_[self.k_,1],[2,0,1]),1)[:,:,:,0]*self.rho_tensor[:,:,:,self.k_] # ( N I' J')
                    denominator      = tf.einsum('nij,ijc->c',m2v2_augmented,1/cropped_sigma)# (C) 
                    m_reverted = self.unpool(tf.transpose(self.m_[self.k_]*self.p_[self.k_,:,:,self.r_],[2,0,1]),1)[:,:,:,0]*self.rho_tensor[:,:,:,self.k_] # ( N I' J')
                    proj       = tf.einsum('nijc,nij->c',cropped_reconstruction/tf.expand_dims(cropped_sigma,0),m_reverted) # (C)
		    t_m_rho_p  = tf.einsum('ijn,ijpqn,ijrn->nijpqr',self.m_[self.k_],self.rho_[k],self.p_[k]) # (N I J Ir Jr R)
                    large_patch= tf.gradients(self.multiplied_x,self.input_large_patch,tf.einsum('nijpqr,rabc->nijpqabc',t_m_rho_p,masked_W[self.k_]))[0]/self.sigmas2_large_patch # (N I J ?? ?? C)
		    selected_W = large_patch[:,:,:,self.i_:self.i_+self.Ir,self.j_:self.j_+self.Jr]       # (N,I,J,Ir,Jr,C)
		    proj2      = tf.einsum('nijabc,nijab->c',selected_W,t_m_rho_p[:,:,:,:,:,self.r_])     # ( C)
                    new_w      = tf.scatter_nd_update(self.W_,[[self.k_,self.r_,self.i_,self.j_]],[(proj+proj2)/(denominator+self.sparsity_prior)])
                    return [new_w]
                else:# TO DOOOO
                    m2v2_augmented   = self.unpool(tf.transpose((tf.square(self.m_[self.k_])+self.v2_[self.k_])\
			*(self.p_[self.k_,:,:,0]+(self.leakiness**2)*self.p_[self.k_,:,:,1])*self.drop_[self.k_,1],[2,0,1]),1)[:,:,:,0]*self.rho_tensor[:,:,:,self.k_] # ( N I' J')
                    denominator      = tf.einsum('nij,ijc->c',m2v2_augmented,1/cropped_sigma)# (C) 
                    m_reverted = self.unpool(tf.transpose(self.m_[self.k_]*(self.p_[self.k_,:,:,0]+self.p_[self.k_,:,:,1]*self.leakiness),[2,0,1]),1)[:,:,:,0]*self.rho_tensor[:,:,:,self.k_]#( N I' J')
                    proj       = tf.einsum('nijc,nij->c',cropped_reconstruction/tf.expand_dims(cropped_sigma,0),m_reverted) # (C)
                    t_m_rho_p  = tf.einsum('ijn,ijpqn,ijrn->nijpqr',self.m_[self.k_],self.rho_[k],self.p_[k]) # (N I J Ir Jr R)
                    large_patch= tf.gradients(self.multiplied_x,self.input_large_patch,tf.einsum('nijpqr,rabc->nijpqabc',t_m_rho_p,masked_W[self.k_]))[0]/self.sigmas2_large_patch # (N I J ?? ?? C)
                    selected_W = large_patch[:,:,:,self.i_:self.i_+self.Ir,self.j_:self.j_+self.Jr]       # (N,I,J,Ir,Jr,C)
                    proj2      = tf.einsum('nijabc,nijab->c',selected_W,(t_m_rho_p[:,:,:,:,:,0]+self.leakiness*t_m_rho_p[:,:,:,:,:,1]))     # ( C)
                    new_w      = tf.scatter_nd_update(self.W_,[[self.k_,self.r_,self.i_,self.j_]],[(proj+proj2)/(denominator+self.sparsity_prior)])
                    return new_w
        def update_pi(self):
                a44      = tf.reduce_sum(self.p_,axis=[1,2,4])/self.bs
                return tf.assign(self.pi,a44/tf.reduce_sum(a44,axis=1,keepdims=True))
        def update_sigma(self):
                a1 = tf.reduce_sum(tf.square(self.input-self.b-self.deconv()),0)/self.bs
                a2 = tf.reduce_sum(self.input_layer.v2,0)/self.bs
                # HERE THE ADDITIONAL NORM
		patches = tf.reduce_sum(tf.square(tf.map_fn(self.get_bigpatch_k,tf.range(self.K),dtype=tf.float32,back_prop=False)),0) # (N,I,J Ic+Ir-1 Jc+Jr-1 C)
		a3  = tf.reduce_sum(self.assemble_large_patch(patches),0)/self.bs
                a40 = self.unpool_R(tf.einsum('kijrn,kijn->nijkr',self.p_,tf.square(self.m_)+self.v2_))*tf.expand_dims(self.rho_tensor,-1) # (N I' J' K R)
                a4  = tf.reduce_sum(self.assemble_small_patch(tf.einsum('nijkr,krabc->nijabc',a40,tf.square(self.W))),0)/self.bs
                value = a1+a2-a3+a4
                if(self.sigma_opt=='local'):
                    return tf.assign(self.sigmas2_,value)
                elif(self.sigma_opt=='channel'):
                    return tf.assign(self.sigmas2_,tf.reduce_sum(value,[0,1],keepdims=True)*tf.ones([self.Iin,self.Jin,1])/(self.Iin*self.Jin))
		elif(self.sigma_opt=='global'):
                    return tf.assign(self.sigmas2_,tf.fill([self.Iin,self.Jin,self.C],tf.reduce_sum(value)/(self.Iin*self.Jin*self.C)))








#########################################################################################################################
#
#
#                                       INPUT LAYER
#
#
#########################################################################################################################




class InputLayer:
        def __init__(self,input_shape,mask=None):
		if(mask is None):
			self.mask = tf.Variable(tf.zeros(input_shape))
			self.v2   = tf.Variable(tf.zeros(input_shape))
		else:
			self.mask = tf.Variable(mask.astype('float32'))
			self.v2   = tf.Variable(mask.astype('float32'))
		self.input_shape  = input_shape
		self.output_shape = input_shape
                self.m            = tf.Variable(tf.zeros(self.input_shape))
		self.v2_          = tf.transpose(self.v2,[3,1,2,0])
	def init_thetaq(self):
		return tf.assign(self.v2,self.mask)
        def likelihood(self):
                return 0
        def KL(self):
		return float32(0.5)*tf.reduce_sum(tf.log(self.v2+eps)*self.mask)/float32(self.input_shape[0])
        def update_v2(self):
		if(isinstance(self.next_layer,ConvPoolLayer)):  a40 = self.next_layer.sigmas2
		else:                a40 = self.next_layer.reshape_sigmas2
                return tf.assign(self.v2,self.mask*a40)
        def update_rho(self):
                return []
        def update_m(self):
                if(isinstance(self.next_layer,ConvPoolLayer)):    priorm  = self.next_layer.deconv()+self.next_layer.b
		else:   priorm  = self.next_layer.backward(0)+self.next_layer.reshape_b
                new_m   = tf.assign(self.m,priorm*self.mask+self.m*(1-self.mask))
                return new_m
        def update_sigma(self):
                return []
        def update_pi(self):
                return []
        def update_W(self):
                return []




#########################################################################################################################
#
#
#                                       FINAL LAYERS
#
#
#########################################################################################################################



class FinalLayer:
        def __init__(self,input_layer,R,sparsity_prior=0,sigma='local',mask = None):
		if(mask is None): self.mask = tf.Variable(tf.zeros(input_layer.output_shape[0]))
		else:		self.mask  = tf.Variable(mask.astype('float32'))
                self.input_layer       = input_layer
		self.sigma_opt         = sigma
		self.sparsity_prior    = sparsity_prior
                input_layer.next_layer = self
		self.input_shape       = input_layer.output_shape
                self.bs                = self.input_shape[0]
                self.output_shape      = (self.bs,1)
                self.D_in              = prod(self.input_shape[1:])
                self.input_shape_      = (self.bs,self.D_in)#potentially different if flattened
                self.input             = input_layer.m
		self.bs      = int32(self.input_shape[0])
		self.R       = R
                self.m       = float32(1)
                self.K       = 1
                self.k_      = tf.placeholder(tf.int32)
		# PARAMETERS
                self.sigmas2_= tf.Variable(tf.ones(self.D_in),trainable=False)
                self.sigmas2 = tf.expand_dims(self.sigmas2_,0)
		self.W       = tf.Variable(tf.random_normal((1,R,self.D_in),float32(0),0.1))
		self.pi      = tf.Variable(tf.ones((1,R))/R) # (1 R)
                self.b_     = tf.Variable(tf.zeros(self.D_in))
                self.b      = tf.expand_dims(self.b_,0)
		# VI PARAMETERS
                self.p_      = tf.Variable(mysoftmax(tf.random_normal((1,self.bs,R)),axis=2)) # (1 N R)
		self.p       = tf.transpose(self.p_,[1,0,2]) #  ( N 1 R ) 
		# HELPER
		self.m_indices = ones(1)
		self.W_indices = ones(1)
                input_layer.next_layer_sigmas2 = self.sigmas2
                if(len(self.input_shape)>2):
                        self.is_flat = False
                        self.input_  = tf.reshape(self.input,(self.bs,self.D_in))
                        self.reshape_sigmas2_ = tf.reshape(self.sigmas2_,self.input_shape[1:])
                        self.reshape_sigmas2  = tf.expand_dims(self.reshape_sigmas2_,0)
                        self.reshape_b_      = tf.reshape(self.b_,self.input_shape[1:])
                        self.reshape_b       = tf.expand_dims(self.reshape_b_,0)
                else:
                        self.is_flat = True
                        self.input_  = self.input
        def init_thetaq(self):
                new_p = tf.assign(self.p_,tf.fill([1,self.bs,self.R],float32(1.0/self.R)))
                return [new_p]
        def backward(self,flat=1):
		if(flat):  return tf.tensordot(self.p,self.W,[[1,2],[0,1]])
		else:      return tf.reshape(tf.tensordot(self.p,self.W,[[1,2],[0,1]]),self.input_shape)
        def backwardk(self,k):
                return tf.tensordot(self.p_[0],self.W[0,:,k],[[1],[0]]) #(N)
        def sample(self,samples,K=None,sigma=1,deterministic=0):
                """ K must be a pre imposed region used for generation
                if not given it is generated according to pi, its shape 
                must be (N K R) with a one hot vector on the last dimension
                sampels is a dummy variable not used in this layer   """
                noise = sigma*tf.random_normal(self.input_shape_)*tf.sqrt(self.sigmas2)
                if(K is None):
                    K = tf.one_hot(tf.transpose(tf.multinomial(self.pi,self.bs)),self.R)
		if(self.is_flat): return tf.tensordot(K,self.W,[[1,2],[0,1]])+noise+self.b
		else:             return tf.reshape(tf.tensordot(K,self.W,[[1,2],[0,1]])+noise+self.b,self.input_shape)
	def evidence(self):
                k1  = -tf.reduce_sum(tf.log(self.sigmas2_)/2)
                k2  = tf.einsum('unr,ur->n',self.p_,tf.log(self.pi))
		k3  = -float32(self.D_in)*tf.log(2*PI_CONST)*float32(0.5)
                reconstruction  = -tf.einsum('nd,d->n',tf.square(self.input_-self.b-self.backward(1)),1/self.sigmas2_)*float32(0.5)
                if(isinstance(self.input_layer,DenseLayer)):
		    input_v2  = -tf.einsum('kn,k->n',self.input_layer.v2_,1/self.sigmas2_)*float32(0.5)
		else: input_v2= -tf.reduce_sum(tf.reshape(self.input_layer.v2,[selef.bs,self.D_in])/self.sigmas2,1)*float32(0.5)
                m2v2        = -tf.einsum('nr,rd,d->n',self.p_[0],tf.square(self.W[0]),1/self.sigmas2_)*float32(0.5) #(N D) -> (D)
                sum_norm_k  = tf.einsum('nd,d->n',tf.square(tf.tensordot(self.p_[0],self.W[0],[[1],[0]])),1/self.sigmas2_)*float32(0.5) #(D)
                return k1+k2+k3+(reconstruction+input_v2+sum_norm_k+m2v2)
	def likelihood(self):
                k1  = -tf.reduce_sum(tf.log(self.sigmas2_))*float32(0.5)  
                k2  = tf.einsum('unr,ur->',self.p_,tf.log(self.pi))*2
                k3  = -tf.log(2*PI_CONST)*float32(self.D_in*0.5)
                reconstruction  = -tf.einsum('nd,d->',tf.square(self.input_-self.b-self.backward(1)),1/self.sigmas2_)
                if(isinstance(self.input_layer,DenseLayer)):
                    input_v2  = -tf.einsum('kn,k->',self.input_layer.v2_,1/self.sigmas2_)
                else: input_v2= -tf.reduce_sum(tf.reshape(tf.reduce_sum(self.input_layer.v2,0),[self.D_in])/self.sigmas2_)
                m2v2        = -tf.einsum('nr,rd,d->',self.p_[0],tf.square(self.W[0]),1/self.sigmas2_) #(N D) -> (D)
                sum_norm_k  = tf.einsum('nd,d->',tf.square(tf.tensordot(self.p_[0],self.W[0],[[1],[0]])),1/self.sigmas2_) #(D)
                return k1+k3+(k2+reconstruction+input_v2+sum_norm_k+m2v2)*float32(0.5/self.bs)-self.sparsity_prior*tf.reduce_sum(tf.square(self.W))
        def KL(self):
                return self.likelihood()-tf.reduce_sum(self.p*tf.log(self.p+eps))/float32(self.bs)
	def update_v2(self):
		return None
        def update_p(self):
		proj    = tf.einsum('nd,rd->nr',self.input_-self.b,self.W[0]/self.sigmas2) # ( N R)
                prior   = tf.expand_dims(self.pi[0],0)
                m2v2    = -tf.expand_dims(tf.einsum('rd,d->r',tf.square(self.W[0]),float32(0.5)/self.sigmas2_),0) # ( 1 R )
                V       = mysoftmax(proj+m2v2+tf.log(prior),coeff=0.0)
                return tf.assign(self.p_,tf.expand_dims(V*tf.expand_dims(self.mask,-1)+self.p_[0]*(1-tf.expand_dims(self.mask,-1)),0))
        def update_sigma(self):
#		return []
                rec = tf.reduce_sum(tf.square(self.input_-self.b-self.backward(1)),0)# (D)
                a1  = tf.reshape(tf.reduce_sum(self.input_layer.v2,0),[self.D_in]) #(D)
                a3  = tf.reduce_sum(tf.tensordot(self.p_[0],tf.square(self.W[0]),[[1],[0]]),0) #(D)
                a2  = -tf.reduce_sum(tf.square(tf.tensordot(self.p_[0],self.W[0],[[1],[0]])),0) #(D)
                value =  (rec+a1+a2+a3)/self.bs
                if(self.sigma_opt=='local'):
                    return tf.assign(self.sigmas2_,value)
		elif(self.sigma_opt=='global'):
                    return tf.assign(self.sigmas2_,tf.fill([self.D_in],tf.reduce_sum(value)/self.D_in))
		elif(self.sigma_opt=='none'):
		    return []
        def update_pi(self):
#		return []
                a44         = tf.reduce_sum(self.p,axis=0)/self.bs
                return tf.assign(self.pi,a44/tf.reduce_sum(a44,axis=1,keepdims=True))
        def update_Wk(self):
#		return []
                rec    = tf.einsum('nd,nr->rd',self.input_-self.b,self.p_[0])
                KK     = rec/(tf.expand_dims(tf.reduce_sum(self.p_[0],0),-1)+self.sparsity_prior)
                return tf.assign(self.W,[KK])
	def update_BV(self):
#		return []
		return tf.assign(self.b_,tf.reduce_sum(self.input_-self.backward(1),0)/self.bs)




class ContinuousLastLayer:
	def __init__(self,input_layer,sigma='local'):
		self.sigma_opt      = sigma
                self.input_layer    = input_layer
                input_layer.next_layer = self
                self.input_shape       = input_layer.output_shape
                self.bs,self.K  = self.input_shape[0],self.input_shape[1]
                self.D_in         = self.K
                self.input_shape_ = (self.bs,self.D_in)#potentially different if flattened
                self.input        = input_layer.m
                # PARAMETERS
                self.sigmas2_= tf.Variable(tf.ones(self.D_in),trainable=False)
                self.sigmas2 = tf.expand_dims(self.sigmas2_,0)
                # VI PARAMETERS
		self.m_      = tf.Variable(tf.random_normal((K,self.bs))*0.1)
                self.m       = tf.transpose(self.m_)
                self.p_      = tf.Variable(mysoftmax(tf.random_normal((K,self.bs,R))*0.1,axis=2)) # convenient dimension ordering for fast updates shape: (D^{(\ell)},N,R^{(\ell)})
                self.p       = tf.transpose(self.p_,[1,0,2])                            # variable for $[p^{(\ell)}_n]_{d,r} of shape (N,D^{(\ell)},R^{(\ell)})$
                self.v2_     = tf.Variable(tf.ones((K,self.bs)))        # variable holding $[v^{(\ell)}_n]^2_{d}, \forall n,d$
                self.v2      = tf.transpose(self.v2_,[1,0])
		# placeholder for update and indices
                self.k_      = tf.placeholder(tf.int32)                                 # placeholder that will indicate which neuron is being updated
        def init_thetaq(self):
                new_p       = tf.assign(self.p_,mysoftmax(tf.ones((self.K,self.bs,self.R)),axis=2))
                new_v       = tf.assign(self.v2_,tf.ones((self.K,self.bs)))
                new_m       = tf.assign(self.m_,0*tf.truncated_normal((self.K,self.bs),0,float32(1/sqrt(self.K))))
                return [new_m,new_p,new_v]
        def backward(self,flat=1,resi=None):
		return self.m
        def backwardk(self,k,resi = None):
		return self.m[:,k]
	def sample(self,M=None,sigma=1,deterministic=False):
		if(M is None): M = tf.random_normal((self.bs,self.D_in))*tf.sqrt(self.sigmas2)
		return M
	def likelihood(self):
                k1  = -tf.reduce_sum(tf.log(self.sigmas2_))*float32(0.5)                            #(1)
                k2  = -tf.log(2*PI_CONST)*float32(self.D_in*0.5)
                reconstruction = -tf.einsum('nd,d->',tf.square(self.input-self.m),1/self.sigmas2_)                   #(1)
                input_v2       = -tf.einsum('dn,d->',self.input_layer.v2_,1/self.sigmas2_)  #(1)
                m2v2           = -tf.einsum('dn,d->',self.v2_,1/self.sigmas2_)
                return k1+k2+(m2v2+reconstruction+input_v2)*float32(0.5/self.bs)
        def KL(self):
                return self.likelihood()+tf.reduce_sum(tf.log(self.v2_+eps))*float32(0.5/self.bs)+float32(self.D_in/2.0)*tf.log(2*PI_CONST)
        def update_v2(self):
		return tf.assign(self.v2_,tf.expand_dims(self.sigmas2_,1)*tf.ones((1,self.bs)))
	def update_rho(self):
		return []
        def update_m(self,ii):
		return tf.assign(self.m_,tf.transpose(self.input/self.sigmas2))
        def update_sigma(self):
		V = tf.reduce_sum(tf.square(self.input-self.m)+self.v2,0)/self.bs
                if(self.sigma_opt=='local'):
                    return tf.assign(self.sigmas2_,value)
		elif(self.sigma_opt=='global'):
		    return tf.assign(self.sigmas2_,tf.fill([self.D_in],tf.reduce_sum(value)/self.D_in))
        def update_pi(self):
		return []
        def update_Wk(self):
		return []
	def update_BV(self):
		return []




