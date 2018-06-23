import tensorflow as tf
from pylab import *
import utils

eps = 0.

#tf.get_collection('latent')
#tf.get_collection('params')



#def compute_WmpWmp(layer):
#	Wmp=tf.reduce_sum(tf.expand_dims(layer.W,0)*tf.expand_dims(layer.m*layer.p,-1),axis=2)# (N,K,D)
#	masked = tf.expand_dims(Wmp,1)*(1-tf.reshape(tf.eye(layer.K),(1,layer.K,layer.K,1)))#(N,K,K,D)
#	return tf.reduce_sum(tf.expand_dims(Wmp,1)*masked)


def mynorm(W,axis=None):
    return tf.reduce_sum(tf.square(W),axis=axis)

def mysoftmax(W,axis=-1,coeff=0.00):
    input = W-tf.reduce_max(W,axis=axis,keepdims=True)
    delta = tf.nn.softmax(input,axis=axis)
    deltap = delta+coeff
    return deltap/tf.reduce_sum(deltap,axis=axis,keepdims=True)

#########################################################################################################################
#
#
#                                       DENSE/CONV/POOL LAYERS
#
#
#########################################################################################################################



class DenseLayer:
	def __init__(self,input_layer,K,R,sparsity_prior = 0,nonlinearity='relu'):
                self.nonlinearity = nonlinearity
                self.sparsity_prior = sparsity_prior
                self.input_layer  = input_layer
                input_layer.next_layer = self
                self.input_shape  = input_layer.output_shape
                self.bs           = self.input_shape[0]
                self.output_shape = (self.bs,K)
                self.D_in         = prod(self.input_shape[1:])
                self.D_out        = K
                self.input_shape_ = (self.bs,self.D_in)#potentially different if flattened
                self.input        = input_layer.m
		if(len(self.input_shape)>2):
                        self.is_flat = False
                        self.input_  = tf.reshape(self.input,(self.bs,self.D_in))
		else:
                        self.is_flat = True
                        self.input_  = self.input
		self.R       = R
                self.K       = K
                # PARAMETERS
                if(nonlinearity is None):
		    self.W_  = tf.Variable(tf.random_normal((K,R,self.D_in))/(self.D_in))
                    self.W = self.W_
                elif(nonlinearity == 'relu'):
                    self.W_  = tf.Variable(tf.random_normal((K,1,self.D_in))/(self.D_in))
                    self.W   = tf.concat([self.W_,tf.zeros_like(self.W_)],axis=1)
                elif(nonlinearity=='abs'):
                    self.W_  = tf.Variable(tf.random_normal((K,1,self.D_in))/(self.D_in))
                    self.W   = tf.concat([self.W_,-self.W_],axis=1)
                self.pi_     = 3.14159
		self.pi      = tf.Variable(mysoftmax(tf.random_normal((K,R))))
		self.sigmas2 = tf.Variable(tf.ones(1))
                # VI PARAMETERS
		self.m_      = tf.Variable(tf.ones((K,self.bs)))
                self.m       = tf.transpose(self.m_)
                self.p_      = tf.Variable(mysoftmax(tf.random_normal((K,self.bs,R)),axis=2)) # convenient dimension ordering for fast updates shape: (D^{(\ell)},N,R^{(\ell)})
                self.p       = tf.transpose(self.p_,[1,0,2])                            # variable for $[p^{(\ell)}_n]_{d,r} of shape (N,D^{(\ell)},R^{(\ell)})$
                self.v2_     = tf.Variable(tf.ones((K,self.bs)))              # variable holding $[v^{(\ell)}_n]^2_{d}, \forall n,d$
                self.v2      = tf.transpose(self.v2_,[1,0])
                self.k_      = tf.placeholder(tf.int32)                                 # placeholder that will indicate which neuron is being updated
#                                           ----  INITIALIZER    ----
        def init_thetaq(self,random):
            if(random):
                new_p       = tf.assign(self.p_,mysoftmax(tf.random_normal((self.K,self.bs,self.R)),axis=2))
                new_v       = tf.assign(self.v2_,tf.ones((self.K,self.bs)))
                new_m       = tf.assign(self.m_,tf.ones((self.K,self.bs)))
            else:
                proj  = tf.tensordot(self.input_,self.W,[[1],[2]]) # (N K R)
                norms = tf.reshape(mynorm(self.W,axis=2),[1,self.K,self.R])
                new_p = tf.assign(self.p_,tf.transpose(mysoftmax(proj-0.5*norms),[1,0,2]))
                new_m = tf.assign(self.m_,tf.reduce_sum(tf.transpose(proj,[1,0,2])*new_p,2)/(1+tf.reduce_sum(tf.transpose(norms,[1,0,2])*new_p,2)))
                new_v       = tf.assign(self.v2_,tf.ones((self.K,self.bs)))
                return [new_m,new_p,new_v]
        def init_thetaW(self,random):
            if(random):
                if(self.nonlinearity=='abs' or self.nonlinearty=='relu'):
                    new_W  = tf.assign(self.W_,tf.random_normal((self.K,1,self.D_in))/(self.D_in))
                else:
                    new_W  = tf.assign(self.W_,tf.random_normal((self.K,self.R,self.D_in))/(self.D_in))
            else:
                p      = permutation(self.bs)[:self.K]
                Xs     = tf.gather(self.input_,p)
                if(self.nonlinearity=='abs' or self.nonlinearty=='relu'):
                    new_W  = tf.assign(self.W_,tf.expand_dims(Xs,1)+tf.random_normal((self.K,1,self.D_in))/float32(self.D_in))
                else:
                    new_W  = tf.assign(self.W_,tf.concat([tf.expand_dims(Xs,1)+tf.random_normal((self.K,1,self.D_in))/float32(self.D_in) for r in xrange(self.R)],axis=1))
            return [new_W]
#                                           ---- BACKWARD OPERATOR ---- 
        def backward(self,flat=1):
		if(flat):
                        return tf.tensordot(self.p*tf.expand_dims(self.m,-1),self.W,[[1,2],[0,1]])
		else:
			return tf.reshape(tf.tensordot(self.p*tf.expand_dims(self.m,-1),self.W,[[1,2],[0,1]]),self.input_shape)
	def backwardmk(self,k):
                return tf.tensordot(self.p*tf.expand_dims(self.m*tf.expand_dims(1-tf.one_hot(k,self.K),0),-1),self.W,[[1,2],[0,1]])
	def sample(self,M,K=None,sigma=1):
                if(isinstance(self.input_layer,InputLayer)):
                    sigma=0
		noise = sigma*tf.random_normal((self.bs,self.D_in))*tf.sqrt(self.sigmas2[0])
                if(K==None):
		    K = tf.transpose(tf.reshape(tf.one_hot(tf.multinomial(tf.log(self.pi),self.bs),self.R),(self.K,self.bs,self.R)),[1,0,2])
		if(self.is_flat):
			return tf.tensordot(tf.expand_dims(M,-1)*K,self.W,[[1,2],[0,1]])+noise
		else:
                        return tf.reshape(tf.tensordot(tf.expand_dims(M,-1)*K,self.W,[[1,2],[0,1]])+noise,self.input_shape)
        def likelihood(self):
                rec = -utils.SSE(self.input,self.backward(0))
                a1  = -self.bs*self.D_in*(tf.log(self.sigmas2[0]*2*self.pi_)/2)
                a2  = tf.reduce_sum(tf.reduce_sum(self.p,axis=0)*tf.log(self.pi+eps))
                a3  = -tf.reduce_sum(self.input_layer.v2)
                a4  = -tf.reduce_sum((self.v2+tf.pow(self.m,2))*tf.reduce_sum(self.p*tf.expand_dims(mynorm(self.W,2),0),2))
                a5  = tf.reduce_sum(tf.pow(self.m,2)*mynorm(tf.reduce_sum(tf.expand_dims(self.W,0)*tf.expand_dims(self.p,-1),2),2))
                return a1+a2+(rec+a3+a4+a5)/(2*self.sigmas2[0])
        def KL(self):
                return self.likelihood()-tf.reduce_sum(self.p_*tf.log(self.p_+eps))+tf.reduce_sum(tf.log(2*self.pi_*self.v2_+eps)/2)
#                                           ----      UPDATES      -----
        def update_v2(self):
                value   = 1/(self.next_layer.sigmas2[0]+eps)+tf.reduce_sum(self.p_*tf.expand_dims(mynorm(self.W,2),1),2)/(eps+self.sigmas2[0])
                new_v2  = tf.assign(self.v2_,1/value)
                return [new_v2]
        def update_vmpk(self):
                k       = self.k_
                proj    = tf.tensordot(self.input_-self.backwardmk(k),self.W[k],[[1],[1]]) #(N R)
                # UPDATE P
                prior   = tf.expand_dims(self.pi[k],0)
                m2v2    = -tf.expand_dims(tf.square(self.m_[k])+self.v2_[k],-1)*tf.expand_dims(mynorm(self.W[k],1),0) # ( N R )
                V       = mysoftmax(proj*tf.expand_dims(self.m_[k],-1)/self.sigmas2[0]+m2v2/(2*self.sigmas2[0])+tf.log(prior),coeff=0.01)
                new_p   = tf.scatter_update(self.p_,[k],[V])
                # UPDATE V
                value   = tf.reduce_sum(new_p[k]*tf.expand_dims(mynorm(self.W[k],1),0),1)
                value_  = 1/(self.next_layer.sigmas2[0]+eps)+value/(eps+self.sigmas2[0])
                new_v2  = tf.scatter_update(self.v2_,[k],[tf.clip_by_value(1/value_,0.01,10000)])
                # UPDATE M 
                priorm  = self.next_layer.backward()[:,k] # ( N )
                coeff   = self.next_layer.sigmas2[0]/self.sigmas2[0]
                new_m   = tf.scatter_update(self.m_,[k],[(priorm+coeff*tf.reduce_sum(proj*new_p[k],axis=1))/(1+value)])
                return [new_v2,new_m,new_p]
        def update_sigma(self,local=1):              
                rec     = -utils.SSE(self.input_,self.backward())
                a1      = -tf.reduce_sum(self.input_layer.v2)
                a3      = -tf.reduce_sum((self.v2+tf.square(self.m))*tf.reduce_sum(self.p*tf.expand_dims(mynorm(self.W,2),0),2))
                a2      = tf.reduce_sum(tf.square(self.m)*mynorm(tf.reduce_sum(tf.expand_dims(self.W,0)*tf.expand_dims(self.p,-1),2),2))
                value   = -(rec+a1+a3+a2)
                if(local):
                    new_sigmas2 = tf.assign(self.sigmas2,[tf.clip_by_value(value/float32(prod(self.input_shape)),0.1,10)])
                    return [new_sigmas2]
                else:
                    return value
        def update_pi(self):
                a44     = tf.reduce_mean(self.p,axis=0)
                pi_ = a44/tf.reduce_sum(eps+a44,axis=1,keepdims=True)
                new_pi  = tf.assign(self.pi,pi_)
                return [new_pi]
        def update_Wk(self):
                k       = self.k_
                if(self.nonlinearity is None):
                    numerator   = tf.tensordot(tf.expand_dims(self.m_[k],-1)*self.p_[k],(self.input_-self.backwardmk(k))/self.bs,[[0],[0]]) # ( R Din )
                    denominator = tf.reduce_mean(tf.expand_dims(tf.square(self.m_[k])+self.v2_[k],-1)*self.p_[k],0)
                    new_w       = tf.scatter_update(self.W_,[k],[numerator/(self.sparsity_prior+tf.expand_dims(denominator,-1))])
                elif(self.nonlinearity=='relu'):
                    numerator   = tf.tensordot(self.m_[k]*self.p_[k,:,0],(self.input_-self.backwardmk(k))/self.bs,[[0],[0]]) # (Din)
                    denominator = tf.reduce_sum((tf.square(self.m_[k])+self.v2_[k])*self.p_[k,:,0])/self.bs
                    new_w       = tf.scatter_nd_update(self.W_,[[k,0]],[numerator/(self.sparsity_prior+denominator)])
                elif(self.nonlinearity=='abs'):
                    numerator   = tf.tensordot(self.m_[k]*(self.p_[k,:,0]-self.p_[k,:,1]),(self.input_-self.backwardmk(k)),[[0],[0]]) # (Din)
                    denominator = tf.reduce_sum((tf.square(self.m_[k])+self.v2_[k]))
                    new_w       = tf.scatter_nd_update(self.W_,[[k,0]],[numerator/(self.sparsity_prior+denominator)])
                return [new_w]









class ConvLayer:
	def __init__(self,input_layer,Ic,Jc,K,R,stride=1,sparsity_prior = 0,nonlinearity='relu'):
                self.nonlinearity = nonlinearity
                self.sparsity_prior    = sparsity_prior
                self.input_layer       = input_layer
                input_layer.next_layer = self
                self.bs,self.Iin,self.Jin,self.C  = input_layer.output_shape 
                self.Ic,self.Jc,self.K,self.R     = Ic,Jc,K,R
                self.input        = input_layer.m
                self.input_shape = input_layer.output_shape
		self.output_shape = (self.bs,(self.input_shape[-3]-self.Ic+1)/stride,(self.input_shape[-2]-self.Jc+1)/stride,K)
		self.D_in     = prod(self.input_shape[1:])
		self.s        = stride
                self.I,self.J = self.output_shape[1],self.output_shape[2]
                self.input_patches = tf.reshape(tf.extract_image_patches(self.input,(1,self.Ic,self.Jc,1),(1,stride,stride,1),(1,1,1,1),"VALID"),(self.bs,self.I,self.J,self.Ic,self.Jc,self.C))
                self.pi_     = 3.14159
                if(nonlinearity is None):
		    self.W_  = tf.Variable(tf.random_normal((self.K,self.R,self.Ic,self.Jc,self.C))/float32(self.Ic*self.Jc*self.C))# (K,R,Ic,Jc,C) always D_in last
                    self.W   = self.W_
                elif(nonlinearity=='relu'):
                    self.W_  = tf.Variable(tf.random_normal((self.K,1,self.Ic,self.Jc,self.C))/float32(self.Ic*self.Jc*self.C))# (K,R,Ic,Jc,C) always D_in last
                    self.W   = tf.concat([self.W_,tf.zeros_like(self.W_)],axis=1)
                elif(nonlinearity=='abs'):
                    self.W_  = tf.Variable(tf.random_normal((self.K,1,self.Ic,self.Jc,self.C))/float32(self.Ic*self.Jc*self.C))# (K,R,Ic,Jc,C) always D_in last
                    self.W   = tf.concat([self.W_,-self.W_],axis=1) 
		self.pi      = tf.Variable(mysoftmax(tf.random_uniform((K,R)),axis=1))
		self.sigmas2 = tf.Variable(tf.ones(1))
		self.m_      = tf.Variable(tf.ones((K,self.I,self.J,self.bs)))#tf.concat(self.mk,axis=3) # (bs,I,J,K)
                self.m       = tf.transpose(self.m_,[3,1,2,0])
		self.p_      = tf.Variable(mysoftmax(tf.random_uniform((K,self.I,self.J,self.R,self.bs)),axis=3))# (K,I,J,R,N)
                self.p       = tf.transpose(self.p_,[4,1,2,0,3])
                self.v2_     = tf.Variable(tf.ones((self.K,self.I,self.J,self.bs)))
                self.v2      = tf.transpose(self.v2_,[3,1,2,0])
                input_layer.next_layer = self
                # PREPARE THE DECONV OPERATION
                self.deconv_ = lambda v:tf.add_n([tf.nn.conv2d_backprop_input(self.input_shape,tf.transpose(self.W[:,r],[1,2,3,0]),v[r],[1,self.s,self.s,1],"VALID") for r in xrange(self.R)])
                self.minideconv_ = lambda v:tf.add_n([tf.nn.conv2d_backprop_input([self.input_shape[0],self.Ic+2*(self.Ic-1),self.Jc+2*(self.Jc-1),self.input_shape[3]],tf.transpose(self.W[:,r],[1,2,3,0]),v[r],[1,1,1,1],"VALID") for r in xrange(self.R)])
                self.k_ = tf.placeholder(tf.int32)
                self.i_ = tf.placeholder(tf.int32)
                self.j_ = tf.placeholder(tf.int32)
                self.posis_ = tf.Variable(stack([repeat(range(self.I),self.J),tile(range(self.J),self.I)]).T,trainable=False,dtype=tf.int32)
                self.posis  = tf.concat([self.posis_,self.k_*tf.ones([self.I*self.J,1],dtype=tf.int32)],axis=1)
#                                           ----  INITIALIZER    ----
        def init_thetaq(self,random):
            if(random):
                new_p       = tf.assign(self.p_,mysoftmax(tf.random_uniform((self.K,self.I,self.J,self.R,self.bs)),axis=3))
                new_v       = tf.assign(self.v2_,tf.ones((self.K,self.I,self.J,self.bs)))
                new_m       = tf.assign(self.m_,tf.ones((K,self.I,self.J,self.bs)))
            else:
                proj= tf.tensordot(self.input_patches,self.W,[[3,4,4],[2,3,4]]) # ( N I J K R )
                norms = tf.reshape(mynorm(self.W,axis=[2,3,4]),[1,1,1,self.K,self.R]) 
                value  = mysoftmax(proj-0.5*norms,axis=4) # (N I J K R) 
                new_p = tf.assign(self.p_,tf.transpose(value,[3,1,2,4,0]))
                new_v       = tf.assign(self.v2_,tf.ones((self.K,self.I,self.J,self.bs)))
                new_m       = tf.assign(self.m_,tf.reduce_sum(new_p*tf.transpose(proj,[3,1,2,4,0]),3)/(1+tf.reduce_sum(new_p*tf.transpose(norms,[3,0,1,4,2]),3)))#tf.ones((self.K,self.I,self.J,self.bs)))
            return [new_m,new_p,new_v]
        def init_thetaW(self,random):
            if(random):
                new_W  = tf.assign(self.W,)
            else:
                p      = permutation(self.bs)[:self.K]
                i      = randint(5,self.I-5,self.K)
                j      = randint(5,self.J-5,self.K)
                Xs     = tf.gather_nd(self.input_patches,stack([p,i,j]))
                if(self.nonlinearity=='relu' or nonlinearity=='abs'):
                    new_W  = tf.assign(self.W_,tf.expand_dims(Xs,1)+tf.random_normal((self.K,1,self.Ic,self.Jc,self.C))/float32(self.Ic*self.Jc*self.C))
                else:
                    w=tf.concat([tf.expand_dims(Xs,1)+tf.random_normal((self.K,1,self.Ic,self.Jc,self.C))/float32(self.Ic*self.Jc*self.C) for r in xrange(self.R)],axis=1)
                    new_W  = tf.assign(self.W_,w)
                return [new_W]
#                                           ---- BACKWARD OPERATOR ---- 
        def deconv(self,v=None):
                if(v==None):
                        v = tf.expand_dims(self.m,-1)*self.p # (N I J K R)
                return self.deconv_(tf.transpose(v,[4,0,1,2,3]))
        def deconvmijk(self,i,j,k):
                mask     = 1-tf.reshape(tf.one_hot(k,self.K),(1,1,1,self.K))*tf.reshape(tf.one_hot(i,self.I),(1,self.I,1,1))*tf.reshape(tf.one_hot(j,self.J),(1,1,self.J,1))
                v = tf.expand_dims(self.m*mask,-1)*self.p # (N I J K R)
                return self.deconv_(tf.transpose(v,[4,0,1,2,3]))
        def minideconvmijk(self,i,j,k):
                #valid when using with patch (i,j)
                mask = 1-tf.reshape(tf.one_hot(self.Ic-1,2*self.Ic-1),(1,2*self.Ic-1,1,1,1))*tf.reshape(tf.one_hot(self.Jc-1,self.Jc*2-1),(1,1,self.Jc*2-1,1,1))*tf.reshape(tf.one_hot(k,self.K),(1,1,1,self.K,1))
                v         = tf.pad(tf.expand_dims(self.m,-1)*self.p,[[0,0],[self.Ic-1,self.Ic-1],[self.Jc-1,self.Jc-1],[0,0],[0,0]],'CONSTANT')[:,i:i+self.Ic*2-1,j:j+2*self.Jc-1] # (N I J K R)
                return self.minideconv_(tf.transpose(v*mask,[4,0,1,2,3]))[:,self.Ic-1:2*self.Ic-1,self.Jc-1:2*self.Jc-1]
	def backward(self):
                return tf.tensordot(tf.expand_dims(self.m,-1)*self.p,self.W,[[3,4],[0,1]])#(N,I,J,Ic,Jc,C)
        def backwardmk(self,k):
                mmk  = self.m*(1-tf.reshape(tf.one_hot(k,self.K),(1,1,1,self.K))) # (bs,I,J,K)
                return tf.tensordot(tf.expand_dims(mmk,-1)*self.p,self.W,[[3,4],[0,1]])#(bs,I,J,Ic,Jc,C)
        def backwardmijk(self,i,j,k):
                mask     = tf.reshape(tf.one_hot(k,self.K),(1,1,1,self.K))*tf.reshape(tf.one_hot(i,self.I),(1,self.I,1,1))*tf.reshape(tf.one_hot(j,self.J),(1,1,self.J,1))
                mmasked  = self.m*(1-mask) # (bs,I,J,K)
                return tf.tensordot(tf.expand_dims(mmasked,-1)*self.p,self.W,[[3,4],[0,1]])#(bs,I,J,Ic,Jc,C)
        def backwardk(self,k):#tf.tensordot(tf.expand_dims(self.mk[k][:,:,:,0],-1)*self.pk[k][:,:,:,0,:],self.W[k],[[3],[0]])
            return tf.tensordot(tf.expand_dims(self.m[:,:,:,k],-1)*self.p[:,:,:,k],self.W[k],[[3],[0]])#(N,I,J,Ic,Jc,C)
	def sample(self,M,K=None,sigma=1):
		#multinomial returns [K,n_samples] with integer value 0,...,R-1
                if(isinstance(self.input_layer,InputLayer)):
                    sigma=0
		noise = sigma*tf.random_normal(self.input_shape)*tf.sqrt(self.sigmas2[0])
                if(K==None):
                    K    = tf.reshape(tf.one_hot(tf.multinomial(tf.log(self.pi),self.bs*self.I*self.J),self.R),(self.K,self.bs,self.I,self.J,self.R))
                return self.deconv(tf.expand_dims(M,-1)*tf.transpose(K,[1,2,3,0,4]))+noise
        def likelihood(self):
                back = self.backward()
                a1 = tf.reduce_sum(tf.square(self.input-self.deconv()))
                a2 = tf.reduce_sum(self.input_layer.v2_)
                a3 = -tf.add_n([tf.reduce_sum(tf.square(self.backwardk(k))) for k in xrange(self.K)])
                a4 = tf.reduce_sum(tf.reduce_sum(tf.reshape(mynorm(self.W,[2,3,4]),(self.K,1,1,self.R,1))*self.p_,axis=3)*(tf.square(self.m_)+self.v2_))
                k1 = -self.bs*self.D_in*tf.log(self.sigmas2[0]*2*self.pi_)/2
                k2 = tf.reduce_sum(tf.log(self.pi+eps)*tf.reduce_sum(self.p,(0,1,2)))
                return k1+k2-(a1+a2+a3+a4)/(2*self.sigmas2[0])-self.sparsity_prior*tf.reduce_sum(tf.square(self.W))
        def KL(self):
                return self.likelihood()-tf.reduce_sum(self.p*(tf.log(self.p)))+tf.reduce_sum(tf.log(2*self.pi_*self.v2)/2)
        def update_v2(self):# DONE
                value  = tf.reduce_sum(tf.reshape(mynorm(self.W,[2,3,4]),(self.K,1,1,self.R,1))*self.p_,axis=3)# (K,I,J,bs)
                v2     = 1/self.next_layer.sigmas2[0]+value/self.sigmas2[0]
                new_v2 = tf.assign(self.v2_,1/v2)
                return [new_v2]
        def update_vmpk(self):# DONE
                i      = self.i_
                j      = self.j_
                k      = self.k_
                deconv = self.minideconvmijk(i,j,k)
                ######## P
                a       = self.input_patches[:,i,j,:]-deconv#[:,i*self.s:i*self.s+self.Ic,j*self.s:j*self.s+self.Jc,:]#(N Ic Jc C)
                prior   = tf.reshape(tf.log(self.pi[k]),(1,self.R))
                m2v2    = -tf.expand_dims(tf.square(self.m_[k,i,j])+self.v2_[k,i,j],-1)*tf.reshape(mynorm(self.W[k] ,[1,2,3]),(1,self.R))/(2*self.sigmas2[0]) #(N R) ### PLUSSSS
                a1      = tf.tensordot(a,self.W[k],[[1,2,3],[1,2,3]])*tf.expand_dims(self.m_[k,i,j],-1)/self.sigmas2[0]# (N R)
                K       = mysoftmax(m2v2+a1+prior,axis=1,coeff=0.001) # ( N R )
                new_p   = tf.scatter_nd_update(self.p_,[[k,i,j]],[tf.transpose(K)])
                ######## V
                value  = tf.tensordot(mynorm(self.W[k],[1,2,3]),new_p[k,i,j],[[0],[0]])# (bs)
                v2     = 1/self.next_layer.sigmas2[0]+value/self.sigmas2[0]
                new_v2 = tf.scatter_nd_update(self.v2_,[[k,i,j]],[tf.clip_by_value(1/v2,0.0001,10000)])
                ######## M
                if(isinstance(self.next_layer,ConvLayer) or isinstance(self.next_layer,PoolLayer)):
                        backward = self.next_layer.deconv()[:,i,j,k]
                else:
                        backward = self.next_layer.backward(0)[:,i,j,k]
                b       = tf.tensordot(new_p[k,i,j],self.W[k],[[0],[0]])# (N Ic Jc C)
                forward = tf.reduce_sum(a*b,[1,2,3])  # (N)
                mcoeff  = self.next_layer.sigmas2[0]/self.sigmas2[0]
                new_m   = tf.scatter_nd_update(self.m_,[[k,i,j]],[(forward*mcoeff+backward)/(1+value)])
                return [new_v2,new_m,new_p]
        def update_Wk(self):
                k = self.k_
                def helper(v):
                    ii,jj,kk=v[0],v[1],v[2]
                    deconv  = self.minideconvmijk(ii,jj,kk)
                    a       = self.input_patches[:,ii,jj,:]-deconv
                    #self.input[:,ii*self.s:ii*self.s+self.Ic,jj*self.s:jj*self.s+self.Jc,:]-deconv[:,ii*self.s:ii*self.s+self.Ic,jj*self.s:jj*self.s+self.Jc,:] # (N Ic Jc C)
                    weights = tf.expand_dims(self.m_[kk,ii,jj],0)*self.p_[kk,ii,jj] # (R N)
                    return tf.tensordot(weights/self.bs,a,[[1],[0]]) # (R Ic Jc C)
                def helperrelu(v):
                    ii,jj,kk=v[0],v[1],v[2]
                    deconv  = self.minideconvmijk(ii,jj,kk)
                    a       = self.input_patches[:,ii,jj,:]-deconv#[:,ii*self.s:ii*self.s+self.Ic,jj*self.s:jj*self.s+self.Jc,:] # (N Ic Jc C)
                    weights = self.m_[kk,ii,jj]*self.p_[kk,ii,jj,0]  # (N)
                    return tf.tensordot(weights/self.bs,a,[[0],[0]]) # (Ic Jc C)
                def helperabs(v):
                    ii,jj,kk=v[0],v[1],v[2]
                    deconv  = self.minideconvmijk(ii,jj,kk)
                    a       = self.input_patches[:,ii,jj,:]-deconv # (N Ic Jc C)
                    weights = self.m_[kk,ii,jj]*(self.p_[kk,ii,jj,0]-self.p_[kk,ii,jj,1]) # (N)
                    return tf.tensordot(weights/self.bs,a,[[0],[0]]) # (Ic Jc C)
                if(self.nonlinearity is None):
                    numerator   = tf.add_n([helper(self.posis[i]) for i in xrange(self.I*self.J)])#tf.reduce_sum(tf.map_fn(helper,self.posis,dtype=tf.float32,back_prop=False,parallel_iterations=5,swap_memory=True),0) # R Ic Jc C
                    denominator = tf.reshape(tf.reduce_sum(self.p_[k]*tf.expand_dims(tf.square(self.m_[k])+self.v2_[k],2),axis=(0,1,3)),(self.R,1,1,1))/self.bs
                    new_w       = tf.scatter_update(self.W_,[k],[numerator/(denominator+self.sparsity_prior)])
                elif(self.nonlinearity=='relu'):
                    numerator   = tf.add_n([helperrelu(self.posis[i]) for i in xrange(self.I*self.J)])#tf.reduce_sum(tf.map_fn(helperrelu,self.posis,dtype=tf.float32,back_prop=False,parallel_iterations=10),0) # Ic Jc C
                    denominator = tf.reduce_sum(self.p_[k,:,:,0]*(tf.square(self.m_[k])+self.v2_[k]))/self.bs
                    new_w       = tf.scatter_update(self.W_,[[k,0]],[numerator/(denominator+self.sparsity_prior)])
                elif(self.nonlinearity=='abs'):
                    numerator   = tf.add_n([helperabs(self.posis[i]) for i in xrange(self.I*self.J)])#tf.reduce_sum(tf.map_fn(helperabs,self.posis,dtype=tf.float32,back_prop=False,parallel_iterations=10),0) # R Ic Jc C
                    denominator = tf.reduce_sum(tf.square(self.m_[k])+self.v2_[k])/self.bs
                    new_w       = tf.scatter_update(self.W_,[[k,0]],[numerator/(denominator+self.sparsity_prior)])
                return [new_w]
        def update_pi(self):
                a44      = tf.reduce_mean(self.p,axis=[0,1,2])
                new_pi   = tf.assign(self.pi,a44/tf.reduce_sum(a44,axis=1,keepdims=True))
                return [new_pi]
        def update_sigma(self,local=1):
                rec_patch = self.backward()
                a1 = tf.reduce_sum(tf.square(self.input-self.deconv()))
                a2 = tf.reduce_sum(self.input_layer.v2_)
                a3 = -tf.add_n([tf.reduce_sum(tf.square(self.backwardk(k))) for k in xrange(self.K)])
                a4 = tf.reduce_sum(tf.reduce_sum(tf.reshape(mynorm(self.W,[2,3,4]),(self.K,1,1,self.R,1))*self.p_,axis=3)*(tf.square(self.m_)+self.v2_))
                value = (a1+a2+a3+a4)
                if(local):
                    new_sigmas2 = tf.assign(self.sigmas2,[tf.clip_by_value(value/float32(prod(self.input_shape)),0.1,2)])
                    return [new_sigmas2]
                else:
                    return value







class PoolLayer:
	def __init__(self,input_layer,I):
		#INPUT_SHAPE : (batch_size,input_filter,M,N)
                self.stride            = I
                self.input_layer       = input_layer
                input_layer.next_layer = self
                self.bs,self.Iin,self.Jin,self.C  = input_layer.output_shape 
                self.Ic,self.Jc,self.K,self.R     = I,I,self.C,I**2
                self.I,self.J = self.Iin/self.stride,self.Jin/self.stride
                self.input             = input_layer.m
                self.input_shape       = input_layer.output_shape
		self.output_shape      = (self.bs,self.I,self.J,self.K)
		self.D_in              = prod(self.input_shape[1:])
                self.input_            = tf.reshape(tf.extract_image_patches(self.input,(1,self.Ic,self.Jc,1),(1,self.stride,self.stride,1),(1,1,1,1),"VALID"),(self.bs,self.I,self.J,self.Ic,self.Jc,self.C))#N I J Ic Jc C
                self.pi_     = 3.14159
		self.W       = tf.Variable(tf.reshape(tf.eye(self.R),(self.R,self.Ic,self.Jc)))# (R,Ic,Jc) always D_in last
		self.pi      = tf.Variable(tf.fill([self.R],float32(1.0/self.R)))
		self.sigmas2 = tf.Variable(tf.ones(1))
		self.m_      = tf.Variable(tf.ones((self.C,self.bs,self.I,self.J))) # (C,bs,I,J)
                self.m       = tf.transpose(self.m_,[1,2,3,0])          # (bs,I,J,C)
		self.p_      = tf.Variable(mysoftmax(tf.random_uniform((self.C,self.bs,self.I,self.J,self.R)),axis=4))#tf.concat(self.pk,axis=3) # (C,bs,I,J,R)
                self.p       = tf.transpose(self.p_,[1,2,3,0,4])        # (bs,I,J,C,R)
                self.v2_     = tf.Variable(tf.ones((self.C,self.bs,self.I,self.J))) # (C,bs,I,J)
                self.v2      = tf.transpose(self.v2_,[1,2,3,0])         # (bs,I,J,C)
                # PREPARE THE DECONV OPERATION
#                x      = tf.random_uniform(self.input_shape)
                w      = tf.expand_dims(tf.transpose(self.W,[1,2,0]),2)
                self.deconv_ = lambda z: tf.concat([tf.nn.conv2d_backprop_input([self.bs,self.Iin,self.Jin,1],w,z[:,:,:,c,:],[1,self.stride,self.stride,1],"VALID") for c in xrange(self.C)],axis=3)
#                                           ----  INITIALIZER    ----
        def init_thetaq(self):
                new_p       = tf.assign(self.p_,mysoftmax(tf.random_uniform((self.C,self.bs,self.I,self.J,self.R)),axis=4))
                new_v       = tf.assign(self.v2_,tf.ones((self.C,self.bs,self.I,self.J)))
                new_m       = tf.assign(self.m_,tf.ones((self.C,self.bs,self.I,self.J)))
                return [new_m,new_p,new_v]
        def init_thetaW(self):
                return [None]
#                                           ---- BACKWARD OPERATOR ---- 
        def deconv(self,v=None):
                if(v==None):
                        v = tf.expand_dims(self.m,-1)*self.p
#                v = tf.transpose(v,[0,4,1,2,3])#TO GET IT (K,1,bs,I,J,R)
                return self.deconv_(v)
	def backward(self):
                return tf.transpose(tf.tensordot(tf.expand_dims(self.m,-1)*self.p,self.W,[[4],[0]]),[0,1,2,4,5,3])#(N,I,J,Ic,Jc,C)
	def sample(self,M,K=None,sigma=1):
		#multinomial returns [K,n_samples] with integer value 0,...,R-1
		noise = sigma*tf.random_normal(self.input_shape)*tf.sqrt(self.sigmas2[0])
                if(K==None):
                    K    = tf.reshape(tf.one_hot(tf.multinomial([self.pi],self.bs*self.K*self.I*self.J)[0],self.R),(self.bs,self.I,self.J,self.C,self.R))#(self.batch_size,self.K,self.M,self.N,self.R)
                return self.deconv(tf.expand_dims(M,-1)*K)+noise
        def likelihood(self):
                rec_patch = self.backward()
                a2 = -tf.reduce_sum(self.input_layer.v2_)
                a1 = -tf.reduce_sum(tf.square(self.input-self.deconv()))
                a3 = -tf.reduce_sum(tf.square(self.m_)*(1-mynorm(self.p_,4))+self.v2_)
#                a4 = tf.reduce_sum(tf.square(rec_patch))
                k1 = -self.bs*self.D_in*tf.log(self.sigmas2[0]*2*self.pi_)/2
                k2 = tf.reduce_sum(tf.log(self.pi+eps)*tf.reduce_sum(self.p,(0,1,2)))
                return k1+k2+(a1+a2+a3)/(2*self.sigmas2[0])            
        def KL(self):
                return self.likelihood()-tf.reduce_sum(self.p*(tf.log(self.p+eps)))+tf.reduce_sum(tf.log(2*self.pi_*self.v2+eps)/2)
        def update_v2(self):# DONE
                v2     = 1/self.next_layer.sigmas2[0]+1/self.sigmas2[0]
                new_v2 = tf.assign(self.v2_,tf.clip_by_value(tf.ones_like(self.v2_)/v2,0.001,10000))
                return [new_v2]
        def update_vmpk(self):# DONE
#                back_w  = self.backward()
                proj    = tf.transpose(tf.tensordot(self.input_,self.W,[[3,4],[1,2]]),[3,0,1,2,4])#/self.sigmas2[0] # (N I J C R) -> (C N I J R)
                ####### P
                a1      = proj*tf.expand_dims(self.m_,-1)/self.sigmas2[0]# (C N I J R)
                a2      = -tf.expand_dims(tf.square(self.m_)+self.v2_,-1)/(2*self.sigmas2[0]) # (C N I J)
                K       = mysoftmax(a1+a2+tf.log(tf.reshape(self.pi,(1,1,1,1,self.R))),axis=4,coeff=0.001) # (C N I J R)
                new_p   = tf.assign(self.p_,K)
                ####### V
                new_v2 = self.update_v2()[0]
                ####### M
                if(isinstance(self.next_layer,ConvLayer)):
                        priorm = self.next_layer.deconv()#/self.next_layer.sigmas2[0]
                else:
                        priorm = self.next_layer.backward(0)#/self.next_layer.sigmas2[0] 
                value_ = self.next_layer.sigmas2[0]/self.sigmas2[0] 
                forward = tf.reduce_sum(proj*new_p,4) # (C N I J)
                new_m   = tf.assign(self.m_,(tf.transpose(priorm,[3,0,1,2])+value_*forward)/2)
                return [new_v2,new_m,new_p]
        def update_Wk(self):
                return [None]
        def update_pi(self):
                a44 = tf.reduce_sum(self.p_,axis=[0,1,2,3])
                new_pi   = tf.assign(self.pi,a44/tf.reduce_sum(eps+a44))
                return []
        def update_sigma(self,local=1):
                rec_patch = self.backward()
                a2 = tf.reduce_sum(self.input_layer.v2_)
                a1 = tf.reduce_sum(tf.square(self.input-self.deconv()))
                a3 = tf.reduce_sum(tf.square(self.m_)*(1-mynorm(self.p_,4))+self.v2_)
#                a4 = -tf.reduce_sum(tf.square(rec_patch))
                value = (a1+a2+a3)
                if(local):
                    new_sigmas2 = tf.assign(self.sigmas2,[tf.clip_by_value(value/float32(prod(self.input_shape)),0.000001,10)])
                    return [new_sigmas2]
                else:
                    return value






#########################################################################################################################
#
#
#                                       INPUT LAYER
#
#
#########################################################################################################################




class InputLayer:
        def __init__(self,input_shape):
		self.input_shape = input_shape
		self.output_shape = input_shape
                self.m       = tf.Variable(tf.random_normal(self.input_shape))
#		self.M       = self.m
                self.v2      = tf.zeros(self.output_shape)
                self.v2_     = tf.transpose(self.v2,[3,0,1,2])#tf.zeros(self.output_shape)
        def update_v2(self):
                return None
        def likelihood(self):
                return 0
        def KL(self):
                return 0
        def update_v2(self):
                return []
        def update_m(self):
                return []
        def update_p(self):
                return []
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
        def __init__(self,input_layer,R):
                self.input_layer       = input_layer
                input_layer.next_layer = self
		self.input_shape       = input_layer.output_shape
                self.bs                = self.input_shape[0]
                self.output_shape      = (self.bs,1)
                self.D_in              = prod(self.input_shape[1:])
                self.input_shape_      = (self.bs,self.D_in)#potentially different if flattened
                self.input             = input_layer.m
		if(len(self.input_shape)>2):
                        self.is_flat = False
                        self.input_  = tf.reshape(self.input,(self.bs,self.D_in))
		else:
                        self.is_flat = True
                        self.input_  = self.input
		self.bs      = int32(self.input_shape[0])
		self.R       = R
                self.m       = float32(1)
                self.K       = 1
                self.k_      = tf.placeholder(tf.int32)
		self.W       = tf.Variable(tf.random_normal((1,R,self.D_in))/(self.D_in))
		self.pi      = tf.Variable(mysoftmax(tf.random_normal((1,R))*0.1))
		self.sigmas2 = tf.Variable(tf.ones(1))
                self.p_      = tf.Variable(mysoftmax(tf.random_normal((1,self.bs,R)),axis=2))
		self.p       = tf.transpose(self.p_,[1,0,2])
                input_layer.next_layer_sigmas2 = self.sigmas2
#                                           ----  INITIALIZER    ----
        def init_thetaq(self,random):
                if(random):
                    new_p = tf.assign(self.p_,tf.fill([1,self.bs,self.R],float32(1.0/self.R)))
                else:
                    proj  = tf.tensordot(self.input_,self.W,[[1],[2]]) # (N K R)
                    norms = tf.reshape(mynorm(self.W,axis=2),[1,self.K,self.R])
                    new_p = tf.assign(self.p_,tf.transpose(mysoftmax(proj-0.5*norms),[1,0,2]))
                return [new_p]
        def init_thetaW(self,random):
                if(random):
                    new_W  = tf.assign(self.W,tf.random_normal((1,R,self.D_in))/(self.D_in))
                else:
                    p      = permutation(self.bs)[:self.R]
                    Xs     = tf.gather(self.input_,p)
                    new_W  = tf.assign(self.W,Xs)
                return [new_W]
        def backward(self,flat=1):
		if(flat):
                        return tf.tensordot(self.p,self.W,[[1,2],[0,1]])
		else:
			return tf.reshape(tf.tensordot(self.p,self.W,[[1,2],[0,1]]),self.input_shape)
        def sample(self,samples,K=None,sigma=1):
                """ K must be a pre imposed region used for generation
                if not given it is generated according to pi, its shape 
                must be (N K R) with a one hot vector on the last dimension
                sampels is a dummy variable not used in this layer   """
                noise = sigma*tf.random_normal(self.input_shape)*tf.sqrt(self.sigmas2[0])
                if(K is None):
                    K = tf.one_hot(tf.transpose(tf.multinomial(self.pi,self.bs)),self.R)
		if(self.is_flat):
	                return tf.tensordot(K,self.W,[[1,2],[0,1]])+noise
		else:
                        return tf.reshape(tf.tensordot(K,self.W,[[1,2],[0,1]]),self.input_shape)+noise
        def likelihood(self):
                rec = -utils.SSE(self.input,self.backward(0))
                k1  = -self.bs*self.D_in*(tf.log(self.sigmas2[0]*2*3.14159)/2)
                k2  = tf.reduce_sum(tf.reduce_sum(self.p,axis=0)*tf.log(self.pi+eps))
                a3  = -tf.reduce_sum(self.input_layer.v2)
                a4  = -tf.reduce_sum(self.p*tf.expand_dims(mynorm(self.W,2),0))
                a5  = tf.reduce_sum(mynorm(tf.reduce_sum(tf.expand_dims(self.W,0)*tf.expand_dims(self.p,-1),2),2))
                return k1+k2+(rec+a3+a4+a5)/(2*self.sigmas2[0])
        def KL(self):
                return self.likelihood()-tf.reduce_sum(self.p*tf.log(self.p+0.00000000000000000000001))#we had a small constant for holding the case where p is fixed and thus a one hot
        def update_vmpk(self):
                a2    = tf.tensordot(self.input_,self.W,[[1],[2]])-tf.expand_dims(mynorm(self.W,2)/2,0) # (N K R)
                V     = mysoftmax(a2/self.sigmas2[0]+tf.log(tf.expand_dims(self.pi,0)),axis=2)
                new_p = tf.assign(self.p_,tf.transpose(V,[1,0,2]))
                return [new_p]
        def update_sigma(self,local=1):
                rec = -utils.SSE(self.input_,self.backward())
                a1  = -tf.reduce_sum(self.input_layer.v2)
                a3  = -tf.reduce_sum(tf.reduce_sum(self.p*tf.expand_dims(mynorm(self.W,2),0),2))
                a2  = tf.reduce_sum(mynorm(tf.reduce_sum(tf.expand_dims(self.W,0)*tf.expand_dims(self.p,-1),2),2))
                value =-(rec+a1+a2+a3)
                if(local):
                    new_sigmas2 = tf.assign(self.sigmas2,[tf.clip_by_value(value/float32(prod(self.input_shape)),0.000001,10)])
                    return [new_sigmas2]
                else:
                    return value
        def update_pi(self):
                a44         = tf.reduce_sum(self.p,axis=0)
                new_pi      = tf.assign(self.pi,a44/tf.reduce_sum(eps+a44,axis=1,keepdims=True))
                return [new_pi]
        def update_Wk(self):
                rec    = tf.reduce_mean(tf.reshape(self.input_,(self.bs,1,1,self.D_in))*tf.expand_dims(self.p,-1),0)
                K      = tf.reduce_mean(self.p,0)
                KK     = rec/tf.expand_dims(K,-1)
                new_W  = tf.assign(self.W,tf.nn.relu(KK))
                return [new_W]






