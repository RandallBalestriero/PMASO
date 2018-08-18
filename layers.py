import tensorflow as tf
from pylab import *
import utils
import itertools
from math import pi as PI_CONST


eps = 0.0#0000000000000000001



def mystd(W,axis=None,keepdims=False):
    return tf.sqrt(tf.reduce_mean(tf.square(W-tf.reduce_mean(W,axis=axis,keepdims=True)),axis=axis,keepdims=keepdims))+0.01


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


class BatchNorm:
        def __init__(self,scale=1,center=1):
                self.scale  = scale
                self.center = center

#########################################################################################################################
#
#
#                                       DENSE/CONV/POOL LAYERS
#
#
#########################################################################################################################



class DenseLayer:
	def __init__(self,input_layer,K,R,sparsity_prior = 0,nonlinearity='relu',batch_norm=BatchNorm(1,1),sigma='local'):
                self.nonlinearity   = nonlinearity
		self.sigma_opt      = sigma
		self.batch_norm     = batch_norm
                self.sparsity_prior = sparsity_prior
                self.input_layer    = input_layer
                input_layer.next_layer = self
                self.input_shape       = input_layer.output_shape
                self.bs,self.R,self.K  = self.input_shape[0],R,K
                self.output_shape = (self.bs,K)
                self.D_in         = prod(self.input_shape[1:])
                self.input_shape_ = (self.bs,self.D_in)#potentially different if flattened
                self.input        = input_layer.m
                self.sigmas2_= tf.Variable(tf.ones(self.D_in),trainable=False)
                self.sigmas2 = tf.expand_dims(self.sigmas2_,0)
                self.b_      = tf.Variable(tf.zeros(self.D_in))
                self.b       = tf.expand_dims(self.b_,0)
                # BATCH NORMALIZATION VARIABLES
                self.DD_     = tf.Variable(tf.ones(self.D_in))#mystd(self.input_,0)
                self.DD      = tf.expand_dims(self.DD_,0)
                self.RDD     = 1/self.DD
                self.emean_  = tf.Variable(tf.zeros(self.D_in))
                self.emean   = tf.expand_dims(self.emean_,0)
                self.U_      = tf.Variable(tf.ones(self.D_in))
                self.U       = tf.expand_dims(self.U_,0)
		if(len(self.input_shape)>2):
                        self.is_flat = False
                        self.input_  = tf.reshape(self.input,(self.bs,self.D_in))
			self.reshape_sigmas2_ = tf.reshape(self.sigmas2_,self.input_shape[1:])
                        self.reshape_sigmas2  = tf.expand_dims(self.reshape_sigmas2_,0)
			self.reshape_emean_   = tf.reshape(self.emean_,self.input_shape[1:])
                        self.reshape_emean    = tf.expand_dims(self.reshape_emean_,0)
			self.reshape_RDD      = tf.expand_dims(tf.reshape(1/self.DD_,self.input_shape[1:]),0)
			self.reshape_U        = tf.expand_dims(tf.reshape(self.U_,self.input_shape[1:]),0)
	                self.reshape_b        = tf.reshape(self.b_,self.input_shape[1:])
		else:
                        self.is_flat = True
                        self.input_  = self.input
                # PARAMETERS
                if(nonlinearity is None):
		    self.W_  = tf.Variable(tf.random_normal((K,R,self.D_in),float32(0),float32(0.1)))
                    self.W = self.W_
                elif(nonlinearity == 'relu'):
                    self.W_  = tf.Variable(tf.random_normal((K,1,self.D_in),0,0.1))
                    self.W   = tf.concat([self.W_,tf.zeros_like(self.W_)],axis=1)
                elif(nonlinearity=='abs'):
                    self.W_  = tf.Variable(tf.truncated_normal((K,1,self.D_in),0,0.1))
                    self.W   = tf.concat([self.W_,-self.W_],axis=1)
                self.pi_     = PI_CONST
		self.pi      = tf.Variable(mysoftmax(tf.ones((K,R))/R))
                # VI PARAMETERS
		self.m_      = tf.Variable(tf.truncated_normal((K,self.bs),0,float32(1/sqrt(self.K))))
                self.m       = tf.transpose(self.m_)
                self.p_      = tf.Variable(mysoftmax(tf.random_normal((K,self.bs,R)),axis=2)) # convenient dimension ordering for fast updates shape: (D^{(\ell)},N,R^{(\ell)})
                self.p       = tf.transpose(self.p_,[1,0,2])                            # variable for $[p^{(\ell)}_n]_{d,r} of shape (N,D^{(\ell)},R^{(\ell)})$
                self.v2_     = tf.Variable(tf.ones((K,self.bs)))        # variable holding $[v^{(\ell)}_n]^2_{d}, \forall n,d$
                self.v2      = tf.transpose(self.v2_,[1,0])
		# placeholder for update
                self.k_      = tf.placeholder(tf.int32)                                 # placeholder that will indicate which neuron is being updated
		self.W_indices = asarray(range(self.K))
		self.m_indices = self.W_indices
		self.p_indices = self.W_indices
        def init_thetaq(self):
                new_p       = tf.assign(self.p_,mysoftmax(tf.random_normal((self.K,self.bs,self.R)),axis=2))
                new_v       = tf.assign(self.v2_,tf.ones((self.K,self.bs)))
                new_m       = tf.assign(self.m_,tf.truncated_normal((self.K,self.bs),0,float32(1/sqrt(self.K))))
                return [new_m,new_p,new_v]
#                                           ---- BACKWARD OPERATOR ---- 
        def backward(self,flat=1):
		if(flat):
                        return tf.tensordot(self.p*tf.expand_dims(self.m,-1),self.W,[[1,2],[0,1]])
		else:
			return tf.reshape(tf.tensordot(self.p*tf.expand_dims(self.m,-1),self.W,[[1,2],[0,1]]),self.input_shape)
        def backwardk(self,k):
                return tf.tensordot(self.p*tf.expand_dims(self.m,-1),self.W[:,:,k],[[1,2],[0,1]]) # (N)
	def backwardmk(self,k):
                return tf.tensordot(self.p*tf.expand_dims(self.m*tf.expand_dims(1-tf.one_hot(k,self.K),0),-1),self.W,[[1,2],[0,1]])
	def sample(self,M,K=None,sigma=1,deterministic=False):
		if(deterministic):
                    return tf.reshape(tf.tensordot(tf.expand_dims(self.m,-1)*self.p,self.W,[[1,2],[0,1]])*self.DD*self.U+self.emean+self.b*self.DD,self.input_shape)
		noise = sigma*tf.random_normal((self.bs,self.D_in))*tf.sqrt(self.sigmas2)*self.DD
                if(K==None):
		    K = tf.transpose(tf.reshape(tf.one_hot(tf.multinomial(tf.log(self.pi),self.bs),self.R),(self.K,self.bs,self.R)),[1,0,2])
                return tf.reshape(tf.tensordot(tf.expand_dims(M,-1)*K,self.W,[[1,2],[0,1]])*self.DD*self.U+noise+self.emean+self.b*self.DD,self.input_shape)
        def likelihood(self):
                k1  = -tf.reduce_sum(tf.log(self.sigmas2_+eps))/2
                k2  = tf.reduce_sum(tf.reduce_mean(self.p,axis=0)*tf.log(self.pi))
                a1  = -tf.reduce_mean(tf.square((self.input_-self.emean)*self.RDD-self.b-self.U*self.backward(1)),0)
                a2  = -tf.reshape(tf.reduce_mean(self.input_layer.v2,0),[self.D_in])*tf.square(self.RDD)
                a30 = tf.square(tf.einsum('krd,knr->knd',self.W,self.p_)*tf.expand_dims(self.m_,-1))
                a3  = tf.reduce_sum(tf.reduce_mean(a30,1),0)*tf.square(self.U_)
                a40 = tf.einsum('kr,krd->d',tf.reduce_mean(tf.expand_dims(self.v2_+tf.square(self.m_),-1)*self.p_,1),tf.square(self.W))
		a4  = -a40*tf.square(self.U_)
                return k1+k2+tf.reduce_sum((a1+a2+a3+a4)/(2*self.sigmas2_))-self.sparsity_prior*tf.reduce_sum(tf.square(self.W))
        def KL(self):
                return self.likelihood()-tf.reduce_sum(tf.reduce_mean(self.p_*tf.log(self.p_+0.00000000000000000000000000001),1))+tf.reduce_sum(tf.reduce_mean(tf.log(self.v2_+eps),1))/2
#                                           ----      UPDATES      -----
        def update_v2(self):
                a40     = tf.reduce_sum(tf.square(self.W*tf.reshape(self.U_,[1,1,self.D_in]))*(tf.reshape(self.next_layer.sigmas2_,[self.K,1,1])/tf.reshape(self.sigmas2_,[1,1,self.D_in])),2) # ( K R)
                value   = tf.reshape(self.next_layer.sigmas2_,[self.K,1])/(tf.expand_dims(tf.square(self.next_layer.RDD[0]),-1)+tf.einsum('knr,kr->kn',self.p_,a40))
#                a40     = tf.reduce_sum(tf.square(self.W)/tf.reshape(self.sigmas2_,[1,1,self.D_in]),2) # ( K R)
#                value   = tf.expand_dims(tf.square(self.next_layer.RDD[0])/self.next_layer.sigmas2_,-1)+tf.einsum('knr,kr->kn',self.p_,a40)
                return tf.assign(self.v2_,value)
	def update_U(self):
		back = self.backward(1)
		numerator = tf.reduce_mean(((self.input_-self.emean)*self.RDD-self.b)*back,0)
		backk = tf.einsum('krd,knr->knd',self.W,self.p_)
                denominator = tf.reduce_mean(tf.square(back),0)-tf.reduce_sum(tf.reduce_mean(tf.square(backk*tf.expand_dims(self.m_,-1)),1),0)+tf.einsum('kr,krd->d',tf.reduce_mean(tf.expand_dims(self.v2_+tf.square(self.m_),-1)*self.p_,1),tf.square(self.W))
                return tf.assign(self.U_,numerator/denominator)
	def update_rho(self):
		return [None]
	def update_DD(self):
		op_mean = tf.assign(self.emean_,self.batch_norm.center*(1*tf.reduce_mean(self.input_,0)+0.*self.emean_))
                op_std  = tf.assign(self.DD_,self.batch_norm.scale*(1*mystd(self.input_,0)+0.*self.DD_)+(1-self.batch_norm.scale))
                return tf.group(op_mean,op_std)
        def update_p(self):
                k       = self.k_
		a       = tf.expand_dims((self.input_-self.emean)*self.RDD-self.backwardmk(self.k_)*self.U-self.b,1) # (N 1 D)
		b       = tf.expand_dims(self.W[self.k_]*self.U,0)*tf.reshape(self.m_[k],[self.bs,1,1])
		forward = tf.tensordot(tf.square(a-b),1/(2*self.sigmas2_),[[2],[0]]) # ( N R )
                prior   = tf.expand_dims(tf.log(self.pi[k]),0) # (1 R)
		v2      = tf.expand_dims(self.v2_[self.k_],-1)*tf.expand_dims(tf.reduce_sum(tf.square(self.W[self.k_]*self.U)/(2*self.sigmas2),1),0)
                V       = mysoftmax(-forward+prior-v2)#proj-m2v2+prior)
                return tf.scatter_update(self.p_,[k],[V])
	def update_m(self):
                k       = self.k_
		a       = (self.input_-self.emean)*self.RDD-self.backwardmk(self.k_)*self.U-self.b # (N D)
                proj    = tf.reduce_sum(tf.tensordot(a*self.next_layer.sigmas2_[k]/self.sigmas2,self.W[k]*self.U,[[1],[1]])*self.p_[k],axis=1) #(N)
                priorm  = (self.next_layer.backwardk(k)*self.next_layer.U_[k]+self.next_layer.b_[k]+self.next_layer.emean_[self.k_]*self.next_layer.RDD[0,self.k_])*self.next_layer.RDD[:,self.k_] # ( N )
                a40     = tf.reduce_sum(tf.square(self.W[k]*self.U)*(self.next_layer.sigmas2_[k]/self.sigmas2),1) # (R)
		v_value = tf.square(self.next_layer.RDD[:,self.k_])+tf.tensordot(self.p_[k],a40,[[1],[0]]) #(N)
                new_m   = tf.scatter_update(self.m_,[k],[(priorm+proj)/v_value])
                return [new_m]
        def update_sigma(self):              
                a1  = tf.reduce_mean(tf.square((self.input_-self.emean)*self.RDD-self.b-self.backward(1)*self.U),0)
                a2  = tf.square(self.RDD[0])*tf.reshape(tf.reduce_mean(self.input_layer.v2,0),[self.D_in])
		a4  = tf.einsum('kr,krd->d',tf.reduce_mean(tf.expand_dims(self.v2_+tf.square(self.m_),-1)*self.p_,1),tf.square(self.W))*tf.square(self.U_) # ( D)
#                a4  = tf.einsum('kn,knd->d',self.v2_+tf.square(self.m_),tf.einsum('knr,krd->knd',self.p_,tf.square(self.W))*tf.reshape(self.U_,[1,1,self.D_in]))))/self.bs
		a30 = tf.square(tf.einsum('krd,knr->knd',self.W,self.p_)*tf.expand_dims(self.m_,-1))
                a3  = -tf.reduce_sum(tf.reduce_mean(a30,1),0)*tf.square(self.U_)
                value   = a1+a3+a2+a4
                if(self.sigma_opt=='local'):
                    return tf.assign(self.sigmas2_,value)
		else:
		    return tf.assign(self.sigmas2_,tf.fill([self.D_in],tf.reduce_mean(value)))
        def update_pi(self):
                a44     = tf.reduce_mean(self.p,axis=0)
                pi_     = a44/tf.reduce_sum(eps+a44,axis=1,keepdims=True)#+0.
                new_pi  = tf.assign(self.pi,pi_)
                return [new_pi]
        def update_Wk(self):
                k       = self.k_
		a = (self.input_-self.emean)*self.RDD-self.backwardmk(k)*self.U-self.b # (N D)
                if(self.nonlinearity is None):
                    numerator   = tf.tensordot(tf.expand_dims(self.m_[k],-1)*self.p_[k],a,[[0],[0]])/self.bs # (R D)
                    denominator = tf.reduce_mean(tf.expand_dims(tf.square(self.m_[k])+self.v2_[k],-1)*self.p_[k],0) #(R)
                    new_w       = tf.scatter_update(self.W_,[k],[numerator/(self.sparsity_prior/self.U+tf.expand_dims(denominator,-1)*self.U)])
                elif(self.nonlinearity=='relu'):
                    numerator   = tf.tensordot(self.m_[k]*self.p_[k,:,0],((self.input_-self.emean)*self.RDD-self.backwardmk(k)-self.b),[[0],[0]]) # (Din)
                    denominator = tf.reduce_sum((tf.square(self.m_[k])+self.v2_[k])*self.p_[k,:,0])
                    new_w       = tf.scatter_nd_update(self.W_,[[k,0]],[numerator/(self.sparsity_prior+denominator)])
                elif(self.nonlinearity=='abs'):
                    numerator   = tf.tensordot(self.m_[k]*(self.p_[k,:,0]-self.p_[k,:,1]),(self.input_-self.backwardmk(k)-self.b),[[0],[0]]) # (Din)
                    denominator = tf.reduce_sum((tf.square(self.m_[k])+self.v2_[k]))
                    new_w       = tf.scatter_nd_update(self.W_,[[k,0]],[numerator/(self.sparsity_prior+denominator)])
                return [new_w]
        def update_b(self):
	       	P = (self.input_-self.emean)*self.RDD-self.backward(1)*self.U
        	new_b = tf.assign(self.b_,tf.reduce_mean(P,axis=[0]))
                return [new_b]




class ConvPoolLayer:
	def __init__(self,input_layer,K,Ic,Jc,Ir,Jr,R,sparsity_prior = 0,nonlinearity='relu',batch_norm=BatchNorm(1,1),sigma='local'):
                self.nonlinearity      = nonlinearity
		self.sigma_opt         = sigma
		self.batch_norm = batch_norm
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
                self.input_patches     = tf.reshape(tf.extract_image_patches(self.input,(1,self.Ic,self.Jc,1),(1,1,1,1),(1,1,1,1),"VALID"),
							(self.bs,self.conv_output_shape[1],self.conv_output_shape[2],self.Ic,self.Jc,self.C))
                self.pi_               = PI_CONST
                if(nonlinearity is None):
		    init     = tf.random_normal_initializer(0,0.1)#tf.orthogonal_initializer()
		    self.W_  = tf.Variable(init([self.K,self.R,self.Ic,self.Jc,self.C]))
                    self.W   = self.W_
                elif(nonlinearity=='relu'):
                    self.W_  = tf.Variable(tf.truncated_normal((self.K,1,self.Ic,self.Jc,self.C),0,1))# (K,R,Ic,Jc,C) always D_in last
                    self.W   = tf.concat([self.W_,tf.zeros_like(self.W_)],axis=1)
                elif(nonlinearity=='abs'):
                    self.W_  = tf.Variable(tf.random_normal((self.K,1,self.Ic,self.Jc,self.C)))# (K,R,Ic,Jc,C) always D_in last
                    self.W   = tf.concat([self.W_,-self.W_],axis=1) 
		# WE DEFINE THE PARAMETERS
                self.pi      = tf.Variable(mysoftmax(tf.random_normal((K,R)),axis=1))
		self.sigmas2_= tf.Variable(tf.ones((self.Iin,self.Jin,self.C)))
		self.sigmas2 = tf.expand_dims(self.sigmas2_,0)
                self.sigmas2_large_patch_=tf.reshape(tf.extract_image_patches(self.sigmas2,(1,Ir+Ic-1,Jr+Jc-1,1),(1,Ir,Jr,1),(1,1,1,1),"VALID"),
                                                        (self.output_shape[1],self.output_shape[2],Ir+Ic-1,Jr+Jc-1,self.C)) 
		self.sigmas2_large_patch = tf.expand_dims(self.sigmas2_large_patch_,0)
                self.sigmas2_small_patch_=tf.reshape(tf.extract_image_patches(self.sigmas2,(1,Ic,Jc,1),(1,1,1,1),(1,1,1,1),"VALID"),
                                                        (self.conv_output_shape[1],self.conv_output_shape[2],Ic,Jc,self.C))
		self.sigmas2_small_patch = tf.expand_dims(self.sigmas2_small_patch_,0)
                self.b_     = tf.Variable(tf.zeros(self.input_shape[1:]))
                self.b      = tf.expand_dims(self.b_,0)
                self.DD_     = tf.Variable(tf.ones(self.input_shape[1:]))#mystd(self.input_,0)
                self.DD      = tf.expand_dims(self.DD_,0)
                self.RDD     = 1/self.DD
		self.emean_  = tf.Variable(tf.zeros(self.input_shape[1:]))
		self.emean   = tf.expand_dims(self.emean_,0)
		self.U_      = tf.Variable(tf.ones(self.input_shape[1:]))
		self.U       = tf.expand_dims(self.U_,0)
                self.U_large_patch_=tf.reshape(tf.extract_image_patches(self.U,(1,Ir+Ic-1,Jr+Jc-1,1),(1,Ir,Jr,1),(1,1,1,1),"VALID"),
                                                        (self.output_shape[1],self.output_shape[2],Ir+Ic-1,Jr+Jc-1,self.C))
		self.U_large_patch = tf.expand_dims(self.U_large_patch_,0)
                self.U_small_patch_=tf.reshape(tf.extract_image_patches(self.U,(1,Ic,Jc,1),(1,1,1,1),(1,1,1,1),"VALID"),
                                                        (self.conv_output_shape[1],self.conv_output_shape[2],Ic,Jc,self.C))
		self.U_small_patch = tf.expand_dims(self.U_small_patch_,0)
		###############################   WE DEFINE SOME HELPER FUNCTIONS ###############################################
                self.dx           = tf.zeros((self.bs,self.conv_output_shape[1],self.conv_output_shape[2],1)) # (N I' J' 1)
                self.dxK          = tf.zeros((self.bs,self.conv_output_shape[1],self.conv_output_shape[2],self.K)) # (N I' J' K)
                self.dp            = tf.reshape(tf.extract_image_patches(self.dx,(1,Ir,Jr,1),(1,Ir,Jr,1),(1,1,1,1),"VALID"),
                                                        (self.bs,self.I,self.J,Ir,Jr)) # (N I J Ir Jr)
                self.dpK           = tf.reshape(tf.extract_image_patches(self.dxK,(1,Ir,Jr,1),(1,Ir,Jr,1),(1,1,1,1),"VALID"),
                                                        (self.bs,self.I,self.J,Ir,Jr,K)) # (N I J Ir Jr K)
		self.dp_pool       = tf.nn.avg_pool(self.dxK,[1,self.Ir,self.Jr,1],[1,self.Ir,self.Jr,1],'VALID')*self.Ir*self.Jr # (N I J K)
		### third to reverse the convolution from the patches to the well shaped 3D tensors
                self.dxxx          = tf.reshape(tf.extract_image_patches(self.input,(1,self.Ic,self.Jc,1),(1,1,1,1),(1,1,1,1),"VALID"),
                                                (self.bs,self.conv_output_shape[1],self.conv_output_shape[2],self.Ic,self.Jc,self.C))
                self.dlargexxx     = tf.reshape(tf.extract_image_patches(self.input,(1,self.Ic+self.Ir-1,self.Jc+self.Jr-1,1),(1,self.Ir,self.Jr,1),(1,1,1,1),"VALID"),
                                                (self.bs,self.I,self.J,self.Ic+self.Ir-1,self.Jc+self.Jr-1,self.C))
		self.dxx           = tf.zeros((self.bs,self.J,self.Ic+self.Ir-1,self.Jc+self.Jr-1,self.C)) #(N I J ?? ?? C) 
		# SOME OTHER VARIABLES
		self.m_      = tf.Variable(tf.truncated_normal((K,self.I,self.J,self.bs),0,1))#tf.concat(self.mk,axis=3) # (K,I,J,N)
                self.m       = tf.transpose(self.m_,[3,1,2,0]) # (N I J K)
		self.p_      = tf.Variable(mysoftmax(tf.random_normal((K,self.I,self.J,self.R,self.bs)),axis=3))# (K,I,J,R,N)
                self.p       = tf.transpose(self.expand(tf.transpose(self.p_,[3,4,1,2,0])),[1,2,3,4,0]) # (N I' J' K R)
		self.rho_    = tf.Variable(mysoftmax(tf.random_normal((K,self.I,self.J,self.Ir,self.Jr,self.bs)),axis=[3,4])) # (K I J Ir Jr N)
                self.rho     = self.revert(tf.transpose(self.rho_,[5,1,2,0,3,4])) # (N I' J' K)
                self.v2_     = tf.Variable(tf.ones((self.K,self.I,self.J,self.bs))) # (K I J N)
                self.v2      = tf.transpose(self.v2_,[3,1,2,0])
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
		if(self.nonlinearity is None):
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
	def revertk(self,r):
            # the following takes as input  (N I J Ir Jr) and returns (N I' J' 1)
            return tf.gradients(self.dp,self.dx,r)[0]
	def revert(self,r):
            # the following takes as input (N I J K Ir Jr) returns (N I' J' K)
            return tf.gradients(self.dpK,self.dxK,tf.transpose(r,[0,1,2,4,5,3]))[0]
	def expand(self,r):
            return tf.map_fn(lambda ri:tf.gradients(self.dp_pool,self.dxK,r[ri])[0],tf.range(self.R,dtype=tf.int32),back_prop=False,dtype=tf.float32) # takes (R N I J K) returns (R N I' J' K)
	def deconv_(self,u):
            return tf.gradients(self.dxxx,self.input,tf.reduce_sum(u,3))[0] # takes (N I' J' K Ic Jc C) and returns (N Iin Jin C)
        def deconvk_(self,u):
            return tf.gradients(self.dxxx,self.input,u)[0]                  # takes (N I' J' Ic Jc C) and returns (N Iin Jin C)
	def large_deconv_(self,u):
            return tf.gradients(self.dlargexxx,self.input,u)[0] # takes (N I J ?? ?? C) and returns (N Iin Jin C)
        def W_back(self,u,w):
	    #takes w of shape [R Ic Jc C] and u of shape [N I J Ir Jr R] and returns [N I J ?? ?? C]
	    return tf.transpose(tf.map_fn(lambda uu:tf.gradients(tf.nn.convolution(self.dxx,tf.reshape(tf.transpose(w,[1,2,3,0]),[1,self.Ic,self.Jc,self.C,self.R]),'VALID'),self.dxx,uu)[0],tf.transpose(u,[1,0,2,3,4,5]),back_prop=False),[1,0,2,3,4,5])
        def helper(self,k):
            flat_mrho  = myexpand(self.m_[k],[2,3])*self.rho_[k] # (I J Ir Jr N) 
            flat_mrhop = myexpand(flat_mrho,[-2])*myexpand(self.p_[k],[2,3])# (I J Ir Jr R N)#
            Wflat = self.W_back(tf.transpose(flat_mrhop,[5,0,1,2,3,4]),self.W[k]) # (N I J ?? ?? C)
            return tf.reduce_sum(tf.reduce_mean(tf.square(Wflat),0)*tf.square(self.U_large_patch_)/(2*self.sigmas2_large_patch_))#(2*tf.transpose(re_sigmas,[3,0,1,2])),[0,1,2])
        def helper_(self,k):
            flat_mrho  = myexpand(self.m_[k],[2,3])*self.rho_[k] # (I J Ir Jr N) 
            flat_mrhop = myexpand(flat_mrho,[-2])*myexpand(self.p_[k],[2,3])# (I J Ir Jr R N)#tf.transpose(tf.tensordot(self.p_[k],self.W[k],[[2],[0]]),[3,4,5,2,0,1]) # (I J N Ic Jc C) -> ( N I J)
            return self.W_back(tf.transpose(flat_mrhop,[5,0,1,2,3,4]),self.W[k])*self.U_large_patch # (N I J ?? ?? C)
        def helper2(self,k):
            flat_mrho  = myexpand(self.m_[k],[2,3])*self.rho_[k] # (I J Ir Jr N) 
            flat_mrhop = myexpand(flat_mrho,[-2])*myexpand(self.p_[k],[2,3])# (I J Ir Jr R N)#tf.transpose(tf.tensordot(self.p_[k],self.W[k],[[2],[0]]),[3,4,5,2,0,1]) # (I J N Ic Jc C) -> ( N I J)
            return self.W_back(tf.transpose(flat_mrhop,[5,0,1,2,3,4]),self.W[k])*self.sigmas2_large_patch # (N I J ?? ?? C)
#                                           ----  INITIALIZER    ----
        def init_thetaq(self):
            new_p       = tf.assign(self.p_,mysoftmax(tf.random_uniform((self.K,self.I,self.J,self.R,self.bs)),axis=3))
	    new_rho     = tf.assign(self.rho_,mysoftmax(tf.random_uniform((self.K,self.I,self.J,self.Ir,self.Jr,self.bs)),axis=[3,4]))
            new_v       = tf.assign(self.v2_,tf.ones((self.K,self.I,self.J,self.bs)))
            new_m       = tf.assign(self.m_,tf.random_normal((self.K,self.I,self.J,self.bs)))
            return [new_m,new_p,new_v,new_rho]
#                                           ---- BACKWARD OPERATOR ---- 
        def deconv(self,input=None,masked_m=0,masked_w=0):
		if(input is not None):
                    return self.deconv_(input)
                if(masked_w):
                    mask  = tf.reshape(tf.one_hot(self.i_,self.Ic),[1,1,self.Ic,1,1])*tf.reshape(tf.one_hot(self.j_,self.Jc),[1,1,1,self.Jc,1])*tf.reshape(tf.one_hot(self.k_,self.K),[self.K,1,1,1,1])*tf.reshape(tf.one_hot(self.r_,self.R),[1,self.R,1,1,1])
                    p_w   = tf.einsum('nijkr,krabc->nijkabc',self.p,self.W*(1-mask)) # (N I' J' K Ic Jc C)
                else:
                    p_w   = tf.einsum('nijkr,krabc->nijkabc',self.p,self.W) # (N I' J' K Ic Jc C)
		if(masked_m==1):
                    m_rho = self.revert(tf.transpose(self.rho_*tf.reshape(self.m_*(1-self.mask),[self.K,self.I,self.J,1,1,self.bs]),[5,1,2,0,3,4]))# (N I' J' K)
		elif(masked_m==-1):
                    m_rho = self.revertk(tf.transpose(self.rho_[self.k_]*tf.reshape(self.mask[self.k_],[self.I,self.J,1,1,1]),[4,0,1,2,3]))# (N I' J' 1)
		    return self.deconvk_(myexpand(m_rho,[4,5])*p_w[:,:,:,self.k_])
                else:
                    m_rho = self.revert(tf.transpose(self.rho_*tf.reshape(self.m_,[self.K,self.I,self.J,1,1,self.bs]),[5,1,2,0,3,4]))# (N I' J' K)
                return self.deconv_(myexpand(m_rho,[4,5,6])*p_w)
	def sample(self,M,K=None,sigma=1):
		#multinomial returns [K,n_samples] with integer value 0,...,R-1
		noise = sigma*tf.random_normal(self.input_shape)*tf.sqrt(self.sigmas2)*self.DD
                if(K==None):
                    sigma_hot        = tf.one_hot(tf.reshape(tf.multinomial(tf.log(self.pi),self.bs*self.I*self.J),(self.K,self.I,self.J,self.bs)),self.R) # ( K I J N R)
		    sigma_hot_reshape= tf.transpose(self.expand(tf.transpose(sigma_hot,[4,3,1,2,0])),[1,2,3,4,0]) # (R N I' J' K) -> (N I' J' K R)
		    sigma_hot_w      = tf.einsum('nijkr,krabc->nijkabc',sigma_hot_reshape,self.W) # (N I' J' K Ic Jc C)
                    pool_hot         = tf.one_hot(tf.reshape(tf.multinomial(tf.log(tf.ones((self.K*self.I*self.J,self.R))),self.bs),(self.K,self.I,self.J,self.bs) ),self.Ir*self.Jr) # (K I J N IrxJr)
		    pool_hot_reshape = tf.transpose(tf.reshape(pool_hot,[self.K,self.I,self.J,self.bs,self.Ir,self.Jr]),[3,1,2,0,4,5]) # (N I J K Ir Jr)
		    reverted         = self.revert(tf.reshape(M,[self.bs,self.I,self.J,self.K,1,1])*pool_hot_reshape) # (N I' J' K)
                return self.deconv(sigma_hot_w*myexpand(reverted,[4,5,6]))*self.U*self.DD+noise+self.b+self.emean*self.DD#+tf.expand_dims(self.bb_,0)
        def likelihood(self):
#                back = self.backward()
		# RECONSTRUCTION
                a1 = -tf.reduce_sum(tf.reduce_mean(tf.square((self.input-self.emean)*self.RDD-self.b-self.deconv()*self.U),0)/(2*self.sigmas2))
                a2 = -tf.reduce_sum(tf.reduce_mean(self.input_layer.v2,0)*tf.square(self.RDD)/(2*self.sigmas2))
		# HERE THE ADDITIONAL NORM
                a3  = tf.reduce_sum(tf.map_fn(self.helper,tf.range(self.K),dtype=tf.float32,back_prop=False)) # norm of the per patch
		a40 = tf.reduce_sum(tf.square(myexpand(self.W,[2,3])*myexpand(self.U_small_patch_,[0,1]))/myexpand(2*self.sigmas2_small_patch_,[0,1]),[4,5,6]) #(K R I' J' Ic Jc C) -> (K R I' J')
		a41 = tf.einsum('nijkr,krij->nijk',self.p,a40)*self.rho # (N I' J' K)
                a4  = -tf.reduce_sum(tf.reduce_mean(tf.nn.avg_pool(a41,[1,self.Ir,self.Jr,1],[1,self.Ir,self.Jr,1],'VALID')*self.Ir*self.Jr*(tf.square(self.m)+self.v2),0))
                k1  = -tf.reduce_sum(tf.log(self.sigmas2_+eps))/2
                k2  = tf.reduce_sum(tf.log(self.pi+eps)*tf.reduce_mean(tf.reduce_sum(self.p_,[1,2]),2))
                return k1+k2+(a1+a2+a3+a4)-self.sparsity_prior*tf.reduce_sum(tf.square(self.W))
        def KL(self):
                return self.likelihood()-tf.reduce_sum(tf.reduce_mean(self.p_*tf.log(self.p_+0.00000000000000001),4))-tf.reduce_sum(tf.reduce_mean(self.rho_*tf.log(self.rho_+0.00000000000000000001),5))+tf.reduce_sum(tf.reduce_mean(tf.log(self.v2_+eps),3)/2)
        def update_v2(self):# DONE
                a40 = tf.reduce_sum(tf.square(myexpand(self.W,[2,3])*myexpand(self.U_small_patch_,[0,1]))/myexpand(self.sigmas2_small_patch_,[0,1]),[4,5,6]) # K R I' J')
                a41 = tf.einsum('nijkr,krij->nijk',self.p,a40)*self.rho # (N I' J' K)
                a4  = tf.nn.avg_pool(a41,[1,self.Ir,self.Jr,1],[1,self.Ir,self.Jr,1],'VALID')*self.Ir*self.Jr #( N I J K)
                if(isinstance(self.next_layer,ConvPoolLayer)):
	                v_value      = tf.square(self.next_layer.RDD)/self.next_layer.sigmas2+a4 # (N I J K)
                elif(isinstance(self.next_layer,FinalLayer) or isinstance(self.next_layer,DenseLayer)):
                        v_value      = tf.square(self.next_layer.reshape_RDD)/self.next_layer.reshape_sigmas2+a4 # (N I J K)
                update_value = 1/tf.transpose(v_value,[3,1,2,0])
                new_v2       = tf.assign(self.v2_,update_value)
                return [new_v2]
        def update_m(self):# DONE
		# Forward Information
		deconv_without,deconv_with = self.deconv(None,1,0)*self.U,self.deconv(None,-1,0)*self.U
		patches = tf.extract_image_patches(((self.input-self.emean)*self.RDD-deconv_without-self.b)*deconv_with/self.sigmas2,
					(1,self.Ic+self.Ir-1,self.Jc+self.Jr-1,1),(1,self.Ir,self.Jr,1),(1,1,1,1),"VALID")
		selected_patches = patches[:,self.i_::self.ratio,self.j_::self.ratio] # (N I J ??) -> (N I'' J'' ??)
                forward      = tf.reduce_sum(selected_patches,3)  # (N I'' J'')
                # Backward Information
                if(isinstance(self.next_layer,ConvPoolLayer)):
		    b        = (self.next_layer.deconv()*self.next_layer.U+self.next_layer.b+self.next_layer.emean*self.next_layer.RDD)*self.next_layer.RDD/self.next_layer.sigmas2
                    backward = b[:,self.i_::self.ratio,self.j_::self.ratio,self.k_] # ( N I'' J'')
                elif(isinstance(self.next_layer,FinalLayer) or isinstance(self.next_layer,DenseLayer)):
		    b        = (self.next_layer.backward(0)*self.next_layer.reshape_U+self.next_layer.reshape_b+self.next_layer.reshape_emean*self.next_layer.reshape_RDD)*self.next_layer.reshape_RDD/self.next_layer.reshape_sigmas2
                    backward = b[:,self.i_::self.ratio,self.j_::self.ratio,self.k_] # (N I'' J'')
		# Sigmas normalization
                a40 = tf.reduce_sum(tf.square(myexpand(self.W[self.k_],[1,2])*self.U_small_patch_)/self.sigmas2_small_patch_,[3,4,5]) # R I' J')
                a41 = tf.einsum('nijr,rij->nij',self.p[:,:,:,self.k_],a40)*self.rho[:,:,:,self.k_] # (N I' J')
                a42 = tf.nn.avg_pool(tf.expand_dims(a41,-1),[1,self.Ir,self.Jr,1],[1,self.Ir,self.Jr,1],'VALID')*self.Ir*self.Jr #( N I J 1)
                a4  = a42[:,self.i_::self.ratio,self.j_::self.ratio,0] # (N I'' J'')
                if(isinstance(self.next_layer,ConvPoolLayer)):
			v_value      = (tf.square(self.next_layer.RDD)/self.next_layer.sigmas2)[:,self.i_::self.ratio,self.j_::self.ratio,self.k_]+a4 # (N I'' J'')
		elif(isinstance(self.next_layer,FinalLayer) or isinstance(self.next_layer,DenseLayer)):
                        v_value      = (tf.square(self.next_layer.reshape_RDD)/self.next_layer.reshape_sigmas2)[:,self.i_::self.ratio,self.j_::self.ratio,self.k_]+a4 # (N I'' J'')
		# Update
		update_value_m = (forward+backward)/v_value # (N I'' J'')
                new_m          = tf.scatter_nd_update(self.m_,self.indices_,tf.transpose(tf.reshape(update_value_m,(self.bs,-1))))
		return [new_m]
	def update_p(self):
		# Forward Information
                deconv_without=self.deconv(None,1,0)
		m_rho = myexpand(self.revertk(tf.expand_dims(tf.transpose(self.rho_[self.k_]*tf.reshape(self.m_[self.k_]*self.mask[self.k_],[self.I,self.J,1,1,self.bs]),[4,0,1,2,3]),-1)),[4,5])# (N I' J' 1 1 1)
		a = (self.input-self.emean)*self.RDD-deconv_without*self.U-self.b
		p_helper= lambda r: tf.reduce_sum(tf.extract_image_patches(a*self.deconvk_(m_rho*myexpand(self.W[self.k_,r],[0,1,2,3]))*self.U/self.sigmas2,
					(1,self.Ic+self.Ir-1,self.Jc+self.Jr-1,1),(1,self.Ir,self.Jr,1),(1,1,1,1),"VALID")[:,self.i_::self.ratio,self.j_::self.ratio],3) # (N I'' J'')
		proj             = tf.map_fn(p_helper,tf.range(self.R,dtype=tf.int32),dtype=tf.float32,back_prop=False) # (R N I'' J'')
		# COMPUTE THE SECOND TERM WITH M^2 and V^2
                a40 = tf.reduce_sum(tf.square(myexpand(self.W[self.k_],[1,2])*self.U_small_patch)/(2*self.sigmas2_small_patch),[3,4,5]) # R I' J')
                a41 = tf.expand_dims(a40,1)*tf.expand_dims(self.rho[:,:,:,self.k_],0) # (R N I' J')
                a42 = tf.transpose(tf.nn.avg_pool(tf.transpose(a41,[1,2,3,0]),[1,self.Ir,self.Jr,1],[1,self.Ir,self.Jr,1],'VALID'),[3,0,1,2])*self.Ir*self.Jr #(R N I J)
		a4  = a42[:,:,self.i_::self.ratio,self.j_::self.ratio]*tf.expand_dims(tf.square(self.m[:,self.i_::self.ratio,self.j_::self.ratio,self.k_])+self.v2[:,self.i_::self.ratio,self.j_::self.ratio,self.k_],0)
		# Prior Information
                prior            = tf.reshape(tf.log(self.pi[self.k_]),[self.R,1,1,1]) # (R 1 1 1)
		# Update Value
                update_value     = mysoftmax(proj-a4+prior,axis=0,coeff=0.000000001)# (R N I'' J'')
                update_op        = tf.scatter_nd_update(self.p_,self.indices_,tf.transpose(tf.reshape(update_value,(self.R,self.bs,-1)),[2,0,1]))
                return [update_op] 
        def update_rho(self):
                # Forward Information
                deconv_without = self.deconv(None,1,0)
                patches = tf.reshape(tf.extract_image_patches(((self.input-self.emean)*self.RDD-deconv_without*self.U-self.b)*self.U/self.sigmas2,(1,self.Ic,self.Jc,1),(1,1,1,1),(1,1,1,1),"VALID"),
                                                                (self.bs,self.conv_output_shape[1],self.conv_output_shape[2],self.Ic,self.Jc,self.C)) # (N I' J' Ic Jc C)
		m_augmented      = tf.reshape(self.m[:,:,:,self.k_],[self.bs,self.I,self.J,1,1])*tf.ones([1,1,1,self.Ir,self.Jr]) # (N I J Ir Jr)
		m_reverted       = self.revertk(m_augmented) # (N I' J' 1)
		backward         = tf.tensordot(m_reverted*self.p[:,:,:,self.k_],self.W[self.k_],[[3],[0]]) # (N I' J' Ic Jc C)
		proj             = tf.reduce_sum(patches*backward,[3,4,5]) # (N I' J')
		proj_patches     = tf.reshape(tf.extract_image_patches(tf.expand_dims(proj,-1),(1,self.Ir,self.Jr,1),(1,self.Ir,self.Jr,1),(1,1,1,1),"VALID"),
                                                                (self.bs,self.I,self.J,self.Ir,self.Jr)) # (N I J Ir Jr)
		selected_patches = proj_patches[:,self.i_::self.ratio,self.j_::self.ratio] # (N I'' J'' Ir Jr)
		# EXTRA TERM
                a40 = tf.reduce_sum(tf.square(myexpand(self.W[self.k_],[1,2])*self.U_small_patch)/(2*self.sigmas2_small_patch),[3,4,5]) # R I' J')
                a41 = tf.einsum('nijr,rij->nij',self.p[:,:,:,self.k_],a40) # (N I' J')
                a42 = tf.reshape(tf.extract_image_patches(tf.expand_dims(a41,-1),(1,self.Ir,self.Jr,1),(1,self.Ir,self.Jr,1),(1,1,1,1),"VALID"),
                                                                (self.bs,self.I,self.J,self.Ir,self.Jr)) # (N I J Ir Jr)
                a4  = a42[:,self.i_::self.ratio,self.j_::self.ratio]*myexpand(tf.square(self.m[:,self.i_::self.ratio,self.j_::self.ratio,self.k_])+self.v2[:,self.i_::self.ratio,self.j_::self.ratio,self.k_],[3,4]) # (N I'' J'')
                # CREATE THE UPDATE VALUE
                update_value     = mysoftmax(selected_patches-a4,axis=[3,4],coeff=0.0000000001)# (N I'' J'' Ir Jr)
                update_op        = tf.scatter_nd_update(self.rho_,self.indices_,tf.transpose(tf.reshape(tf.transpose(update_value,[3,4,0,1,2]),(self.Ir,self.Jr,self.bs,-1)),[3,0,1,2]))
                return [update_op]
        def update_Wk(self):
#		return []
		i = self.i_
		j = self.j_
                k = self.k_
		r = self.r_
		## FIRST COMPUTE DENOMINATOR
                m2v2_augmented   = self.revertk(myexpand((tf.square(self.m[:,:,:,self.k_])+self.v2[:,:,:,self.k_]),[-1,-1])*tf.ones((1,1,1,self.Ir,self.Jr)))[:,:,:,0]*self.rho[:,:,:,self.k_]*self.p[:,:,:,self.k_,self.r_] # (N I' J')
		denominator      = tf.reduce_sum(tf.expand_dims(m2v2_augmented,-1)*tf.square(self.U[:,self.i_:self.input_shape[1]-self.Ic+1+self.i_,self.j_:self.input_shape[2]-self.Jc+1+self.j_])/self.sigmas2[:,self.i_:self.input_shape[1]-self.Ic+1+self.i_,self.j_:self.input_shape[2]-self.Jc+1+self.j_],[0,1,2])# (N I' J' C) SUM-> (C)
		## NOW COMPUTE NUMERATOR
		m_reverted       = self.revertk(myexpand(self.m[:,:,:,self.k_],[-1,-1])*tf.ones((1,1,1,self.Ir,self.Jr)))[:,:,:,0]*self.rho[:,:,:,self.k_]*self.p[:,:,:,self.k_,self.r_]  # (N I' J')
                if(self.nonlinearity is None or self.nonlinearity is 'relu'):
		    deconv          = self.deconv(None,0,1)#mijkr(self.i_,self.j_,self.k_,self.r_) # (N Iin Jin C)
                    reconstruction  = ((self.input-self.emean)*self.RDD-deconv*self.U-self.b)*self.U/self.sigmas2 # (N Iin Jin C)
		    cropped_reconstruction = reconstruction[:,self.i_:self.input_shape[1]-self.Ic+1+self.i_,self.j_:self.input_shape[2]-self.Jc+1+self.j_] # (N I'' J'' C)
		    mask            = tf.expand_dims(m_reverted,-1) # (N I' J' 1) 
                    proj            = tf.reduce_sum(cropped_reconstruction*mask,[0,1,2]) # ( C )# SUM
		    # COMPUTE ADDITIONAL TERM
                    mask_W          = tf.reshape(tf.one_hot(self.r_,self.R),[self.R,1,1,1])*tf.reshape(tf.one_hot(self.i_,self.Ic),[1,self.Ic,1,1])*tf.reshape(tf.one_hot(self.j_,self.Jc),[1,1,self.Jc,1])
                    masked_W   = self.W[self.k_]*(1-mask_W)# R Ic Jc C
                    flat_mrho  = myexpand(self.m_[k],[2,3])*self.rho_[k] # (I J Ir Jr N) 
                    flat_mrhop = myexpand(flat_mrho,[-2])*myexpand(self.p_[k],[2,3])# (I J Ir Jr R N)#tf.transpose(tf.tensordot(self.p_[k],self.W[k],[[2],[0]]),[3,4,5,2,0,1]) # (I J N Ic Jc C) -> (Ic Jc C N I J)
		    tflat_mrhop=tf.transpose(flat_mrhop,[5,0,1,2,3,4]) # (N I J Ir Jr R)
                    back_W     = self.W_back(tflat_mrhop,masked_W)*tf.square(self.U_large_patch)/self.sigmas2_large_patch # (N I J ?? ?? C)
		    selected_W = back_W[:,:,:,self.i_:self.i_+self.Ir,self.j_:self.j_+self.Jr]#*flat_rho # (N,I,J,Ir,Jr,C)
		    proj2      = tf.reduce_sum(selected_W*tf.expand_dims(tflat_mrhop[:,:,:,:,:,self.r_],-1),[0,1,2,3,4]) # ( C)
                    new_w      = tf.scatter_nd_update(self.W_,[[self.k_,self.r_,self.i_,self.j_]],[(proj+proj2)/(denominator+self.sparsity_prior)])
                    return [new_w]
                else:# TO DOOOO
		    error
                    ii,jj,kk=v[0],v[1],self.k_
                    deconv  = self.minideconvmijk(ii,jj,kk)
                    a       = self.input_patches[:,ii,jj,:]-deconv -self.b# (N Ic Jc C)
                    numerator = self.m_[kk,ii,jj]*(self.p_[kk,ii,jj,0]-self.p_[kk,ii,jj,1]) # (N)
                    new_w       = tf.scatter_nd_update(self.W_,[[self.k_,self.r_,self.i_,self.j_]],[numerator/(denominator+self.sparsity_prior)])
                    return [new_w]
        def update_pi(self):
                a44      = tf.reduce_mean(self.p_,axis=[1,2,4])
                new_pi   = tf.assign(self.pi,a44/tf.reduce_sum(a44,axis=1,keepdims=True))
                return [new_pi]
        def update_b(self):
#		return [None]
	        P = (self.input-self.emean)*self.RDD-self.deconv()*self.U
	        new_b = tf.assign(self.b_,tf.reduce_mean(P,axis=[0]))
	        return [new_b]
        def update_U(self):
		return []
                back = self.deconv()
                numerator = tf.reduce_mean(((self.input-self.emean)*self.RDD-self.b)*back,0)/self.sigmas2_
                patches = tf.reduce_sum(tf.square(tf.map_fn(self.helper2,tf.range(self.K),dtype=tf.float32,back_prop=False)),0) # (N,I,J Ic+Ir-1 Jc+Jr-1 C)
                a3  = tf.reduce_mean(self.large_deconv_(patches),0)
                a40 = tf.square(myexpand(self.W,[2,3]))/tf.expand_dims(self.sigmas2_small_patch,0) # K R I' J' Ic Jc C)
                a41 = self.expand(tf.transpose(self.p_,[3,4,1,2,0])*tf.expand_dims(tf.square(self.m)+self.v2,0))*tf.expand_dims(self.rho,0) # (R N I' J' K)
                a4  = tf.reduce_mean(self.deconv_(tf.einsum('rnijk,krijabc->nijkabc',a41,a40)),0)
		a5  = tf.reduce_mean(tf.square(back),0)/self.sigmas2_
                denominator = a4-a3+a5
                return tf.assign(self.U_,tf.ones((self.input_shape[1],self.input_shape[2],1))*tf.reduce_mean(numerator,[0,1],keepdims=True)/tf.reduce_mean(denominator,[0,1],keepdims=True))
        def update_DD(self):
                op_mean = tf.assign(self.emean_,self.batch_norm.center*(1*tf.reduce_mean(self.input,[0,1,2],keepdims=True)[0]+0.*self.emean_))
                op_std  = tf.assign(self.DD_,self.batch_norm.scale*(1*mystd(self.input,[0,1,2],keepdims=True)[0]+0.*self.DD_)+(1-self.batch_norm.scale))
                return tf.group(op_mean,op_std)
        def update_sigma(self):
                a1 = -tf.reduce_mean(tf.square((self.input-self.emean)*self.RDD-self.b-self.deconv()*self.U),0)
                a2 = -tf.reduce_mean(self.input_layer.v2*tf.square(self.RDD),0)
                # HERE THE ADDITIONAL NORM
		patches = tf.reduce_sum(tf.square(tf.map_fn(self.helper_,tf.range(self.K),dtype=tf.float32,back_prop=False)),0) # (N,I,J Ic+Ir-1 Jc+Jr-1 C)
#		reshaped_patches = tf.transpose(tf.reshape(tf.transpose(patches,[1,2,3,0]),[self.Ic+self.Ir-1,self.Jc+self.Jr-1,self.C,self.bs,self.I,self.J]),[3,4,5,0,1,2])
		a3  = tf.reduce_mean(self.large_deconv_(patches),0)
                a40 = tf.square(myexpand(self.W,[2,3])*tf.expand_dims(self.U_small_patch,0)) # K R I' J' Ic Jc C)
                a41 = self.expand(tf.transpose(self.p_,[3,4,1,2,0])*tf.expand_dims(tf.square(self.m)+self.v2,0))*tf.expand_dims(self.rho,0) # (R N I' J' K)
                a4  = -tf.reduce_mean(self.deconv_(tf.einsum('rnijk,krijabc->nijkabc',a41,a40)),0)
                value = -(a1+a2+a3+a4)
                if(self.sigma_opt=='local'):
                    return tf.assign(self.sigmas2_,value)
                elif(self.sigma_opt=='channel'):
                    return tf.assign(self.sigmas2_,tf.tile(tf.reduce_mean(value,2,keepdims=True),[1,1,self.C]))
		else:
                    return tf.assign(self.sigmas2_,tf.fill([self.Iin,self.Jin,self.C],tf.reduce_mean(value)))








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
                self.m       = tf.Variable(tf.zeros(self.input_shape))
                self.v2      = tf.zeros(self.output_shape)
                self.v2_     = tf.transpose(self.v2,[3,0,1,2])
                self.dn_output = self.m
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
        def __init__(self,input_layer,R,sparsity_prior=0,batch_norm=BatchNorm(0,0),sigma='local'):
                self.input_layer       = input_layer
		self.sigma_opt         = sigma
		self.sparsity_prior    = sparsity_prior
		self.batch_norm        = batch_norm
                input_layer.next_layer = self
		self.input_shape       = input_layer.output_shape
                self.bs                = self.input_shape[0]
                self.output_shape      = (self.bs,1)
                self.D_in              = prod(self.input_shape[1:])
                self.input_shape_      = (self.bs,self.D_in)#potentially different if flattened
                self.input             = input_layer.m
                self.sigmas2_= tf.Variable(tf.ones(self.D_in),trainable=False)
                self.sigmas2 = tf.expand_dims(self.sigmas2_,0)


                self.DD_     = tf.Variable(tf.ones(self.D_in))#mystd(self.input_,0)
                self.DD      = tf.expand_dims(self.DD_,0)
                self.RDD     = 1/self.DD
                self.emean_  = tf.Variable(tf.zeros(self.D_in))
                self.emean   = tf.expand_dims(self.emean_,0)
                if(len(self.input_shape)>2):
                        self.is_flat = False
                        self.input_  = tf.reshape(self.input,(self.bs,self.D_in))
                        self.reshape_sigmas2_ = tf.reshape(self.sigmas2_,self.input_shape[1:])
                        self.reshape_sigmas2  = tf.expand_dims(self.reshape_sigmas2_,0)
                        self.reshape_emean_   = tf.reshape(self.emean_,self.input_shape[1:])
                        self.reshape_emean    = tf.expand_dims(self.reshape_emean_,0)
                        self.reshape_RDD      = tf.expand_dims(tf.reshape(1/self.DD_,self.input_shape[1:]),0)
		else:
                        self.is_flat = True
                        self.input_  = self.input
                self.U_      = tf.Variable(tf.ones(self.D_in))
                self.U       = tf.expand_dims(self.U_,0)
		self.bs      = int32(self.input_shape[0])
		self.R       = R
                self.m       = float32(1)
                self.K       = 1
                self.k_      = tf.placeholder(tf.int32)
		self.W       = tf.Variable(tf.random_normal((1,R,self.D_in),float32(0),0.1))
		self.pi      = tf.Variable(tf.ones((1,R))/R) # (1 R)
		self.b_      = tf.Variable(tf.zeros((self.D_in)))
		self.b       = tf.expand_dims(self.b_,0)
		self.bb      = tf.reshape(self.b_,self.input_shape[1:])
                self.p_      = tf.Variable(mysoftmax(tf.random_normal((1,self.bs,R)),axis=2)) # (1 N R)
		self.p       = tf.transpose(self.p_,[1,0,2]) #  ( N 1 R ) 
		self.m_indices = ones(1)
		self.W_indices = ones(1)
                input_layer.next_layer_sigmas2 = self.sigmas2
        def init_thetaq(self):
                new_p = tf.assign(self.p_,tf.fill([1,self.bs,self.R],float32(1.0/self.R)))
                return [new_p]
        def backward(self,flat=1):
		if(flat):
                        return tf.tensordot(self.p,self.W,[[1,2],[0,1]])
		else:
			return tf.reshape(tf.tensordot(self.p,self.W,[[1,2],[0,1]]),self.input_shape)
        def backwardk(self,k):
                return tf.tensordot(self.p_[0],self.W[0,:,k],[[1],[0]]) #(N)
        def sample(self,samples,K=None,sigma=1):
                """ K must be a pre imposed region used for generation
                if not given it is generated according to pi, its shape 
                must be (N K R) with a one hot vector on the last dimension
                sampels is a dummy variable not used in this layer   """
                noise = sigma*tf.random_normal(self.input_shape_)*tf.sqrt(self.sigmas2)*self.DD
		m     = 1#0*sigma*tf.random_normal((self.bs,))*tf.sqrt(self.next_sigmas2[0])+self.next_m
                if(K is None):
                    K = tf.one_hot(tf.transpose(tf.multinomial(self.pi,self.bs)),self.R)
		if(self.is_flat):
	                return tf.tensordot(K,self.W,[[1,2],[0,1]])*self.DD*self.U+noise+self.bb*self.DD+self.emean
		else:
                        return tf.reshape(tf.tensordot(K,self.W,[[1,2],[0,1]])*self.DD*self.U+noise+self.emean+self.bb*self.DD,self.input_shape)
        def likelihood(self):
                k1  = -tf.reduce_sum(tf.log(self.sigmas2_+eps)/2)#-self.bs*(tf.log(self.next_sigmas2[0]*2*3.14159)/2)
                k2  = tf.reduce_sum(tf.reduce_mean(self.p,axis=0)*tf.log(self.pi+eps))
                a1  = -tf.reduce_mean(tf.square((self.input_-self.emean)*self.RDD-self.b-self.U*self.backward(1)),0)
                a2  = -tf.square(self.RDD)*tf.reshape(tf.reduce_mean(self.input_layer.v2,0),[self.D_in])
                a3  = -tf.reduce_mean(tf.tensordot(self.p_[0],tf.square(self.W[0]),[[1],[0]]),0)*tf.square(self.U_) #(N D) -> (D)
                a4  = tf.reduce_mean(tf.square(tf.tensordot(self.p_[0],self.W[0],[[1],[0]])),0)*tf.square(self.U_) #(D)
                return k1+k2+tf.reduce_sum((a1+a2+a3+a4)/(2*self.sigmas2_))#-1*utils.SSE(self.next_m,self.m_)/(2*self.next_sigmas2[0])-tf.reduce_sum(self.v2_)/(2*self.next_sigmas2[0])
        def KL(self):
                return self.likelihood()-tf.reduce_sum(self.p*tf.log(self.p+0.00000000000000000001)/self.bs)#+tf.reduce_sum(tf.log(2*3.14159*self.v2_)/2)#we had a small constant for holding the case where p is fixed and thus a one hot
	def update_U(self):
                back = self.backward(1)
                numerator = tf.reduce_mean(((self.input_-self.emean)*self.RDD-self.b)*back,0)
#                backk = tf.einsum('rd,nr->nd',self.W[0],self.p_[0])
                denominator = tf.reduce_mean(tf.einsum('rd,nr->nd',tf.square(self.W[0]),self.p_[0]),0)
                return tf.assign(self.U_,numerator/denominator)
	def update_v2(self):
		return None
	def update_DD(self):
		op_mean = tf.assign(self.emean_,self.batch_norm.center*(1*tf.reduce_mean(self.input_,0)+0.*self.emean_))
		op_std  = tf.assign(self.DD_,self.batch_norm.scale*(1*mystd(self.input_,0)+0.*self.DD_)+(1-self.batch_norm.scale))
		return tf.group(op_mean,op_std)
        def update_p(self):
		proj    = tf.tensordot(((self.input_-self.emean)*self.RDD-self.b)/self.sigmas2,self.W[0]*self.U,[[1],[1]]) # ( N R)
                prior   = tf.expand_dims(self.pi[0],0)
                m2v2    = -tf.expand_dims(tf.reduce_sum(tf.square(self.W[0]*self.U)/(2*self.sigmas2),1),0) # ( 1 R )
                V       = mysoftmax(proj+m2v2+tf.log(prior+eps),coeff=0.0)
                new_p = tf.assign(self.p_,tf.expand_dims(V,0))
                return [new_p]
        def update_sigma(self):
                rec = tf.reduce_mean(tf.square((self.input_-self.emean)*self.RDD-self.b-self.backward(1)*self.U),0)
                a1  = tf.square(self.RDD[0])*tf.reshape(tf.reduce_mean(self.input_layer.v2,0),[self.D_in])
                a3  = tf.reduce_mean(tf.tensordot(self.p_[0],tf.square(self.W[0]*self.U),[[1],[0]]),0)
                a2  = -tf.reduce_mean(tf.square(tf.tensordot(self.p_[0],self.W[0]*self.U,[[1],[0]])),0)
                value =  rec+a1+a2+a3
                if(self.sigma_opt=='local'):
                    return tf.assign(self.sigmas2_,value)
		else:
                    return tf.assign(self.sigmas2_,tf.fill([self.D_in],tf.reduce_mean(value)))
        def update_pi(self):
                a44         = tf.reduce_sum(self.p,axis=0)
                return tf.assign(self.pi,a44/tf.reduce_sum(a44,axis=1,keepdims=True))
        def update_Wk(self):
                rec    = tf.einsum('nd,nr->rd',((self.input_-self.emean)*self.RDD-self.b),self.p_[0])/self.bs
                KK     = rec/(self.U*tf.expand_dims(tf.reduce_mean(self.p_[0],0),-1)+self.sparsity_prior/self.U)
                return tf.assign(self.W,[KK])
        def update_b(self):
	        P = (self.input_-self.emean)*self.RDD-self.backward(1)*self.U
	        return tf.assign(self.b_,tf.reduce_mean(P,axis=[0]))






