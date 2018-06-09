import tensorflow as tf
from pylab import *
import utils

eps = 0.000000001

#tf.get_collection('latent')
#tf.get_collection('params')



#def compute_WmpWmp(layer):
#	Wmp=tf.reduce_sum(tf.expand_dims(layer.W,0)*tf.expand_dims(layer.m*layer.p,-1),axis=2)# (N,K,D)
#	masked = tf.expand_dims(Wmp,1)*(1-tf.reshape(tf.eye(layer.K),(1,layer.K,layer.K,1)))#(N,K,K,D)
#	return tf.reduce_sum(tf.expand_dims(Wmp,1)*masked)


def mynorm(W,axis=None):
    return tf.reduce_sum(tf.square(W),axis=axis)


#########################################################################################################################
#
#
#                                       DENSE/CONV/POOL LAYERS
#
#
#########################################################################################################################



class DenseLayer:
	def __init__(self,input_layer,K,R,sparsity_prior = 0,nonlinearity='relu'):
                if(nonlinearity == 'relu'):
                    self.Wmask = tf.reshape(tf.one_hot(1,2),(1,2,1))
                else:
                    self.Wmask = tf.reshape(tf.ones((2)),(1,2,1))
		#INPUT SHAPE (batch_size,C)
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
		self.W       = tf.Variable(tf.random_normal((K,R,self.D_in))/(K*self.D_in))
                self.pi_     = 3.14159
		self.pi      = tf.Variable(tf.nn.softmax(tf.random_normal((K,R))))
		self.sigmas2 = tf.Variable(tf.ones(1))
                # VI PARAMETERS
		self.m_      = tf.Variable(tf.random_normal((K,self.bs)))
                self.m       = tf.transpose(self.m_)
                self.p_      = tf.Variable(tf.nn.softmax(tf.random_normal((K,self.bs,R)),axis=2)) # convenient dimension ordering for fast updates shape: (D^{(\ell)},N,R^{(\ell)})
                self.p       = tf.transpose(self.p_,[1,0,2])                            # variable for $[p^{(\ell)}_n]_{d,r} of shape (N,D^{(\ell)},R^{(\ell)})$
                self.v2_     = tf.Variable(tf.random_uniform((K,self.bs)))              # variable holding $[v^{(\ell)}_n]^2_{d}, \forall n,d$
                self.v2      = tf.transpose(self.v2_,[1,0])
                self.k_      = tf.placeholder(tf.int32)                                 # placeholder that will indicate which neuron is being updated
#                                           ----  INITIALIZER    ----
        def init_thetaq(self,alpha=0.5):
                # FIRST UPDATE P
                proj        = tf.transpose(tf.tensordot(self.input_,self.W,[[1],[2]])-0.5*tf.expand_dims(mynorm(self.W,[2]),0),[1,0,2])
                proj_random = tf.random_uniform((self.K,self.bs,self.R),-0.1,0.1)
                new_p       = tf.assign(self.p_,tf.nn.softmax(tf.expand_dims(tf.log(self.pi),1)+alpha*proj_random+(1-alpha)*proj,axis=2))
                # GIVEN P WE CAN NOW UPDATE V VIA STANDARD FORMULA WITHOUT NEXT LAYER 
                new_v       = self.update_v2()[0]
                # WITH P and V2 WE NOW UPDATE M WITHOUT THE NEXT LAYER PRIOR AND WITHOUT COLLATERAL INFORMATION
                Wp          = tf.tensordot(self.input_,self.W,[[1],[2]]) #(N K R)
                projm       = tf.reduce_sum(Wp*self.p*Wp,2) #(K N)
                new_m       = tf.assign(self.m_,tf.transpose(projm)/new_v)
                return [new_m,new_p,new_v]
        def init_theta(self):
                init   = tf.random_normal
                new_W  = tf.assign(self.W,self.Wmask*init((self.K,self.R,self.D_in))/sqrt(self.D_in)) 
                new_pi = tf.assign(self.pi,tf.fill([self.K,self.R],1.0/self.R)) # UNIFORM PRIOR
                new_sigma = tf.assign(self.sigmas2,[float32(1)])             # Identity init.
                return [new_W,new_pi,new_sigma]
#                                           ---- BACKWARD OPERATOR ---- 
        def backward(self,flat=1):
		if(flat):
                        return tf.tensordot(self.p*tf.expand_dims(self.m,-1),self.W,[[1,2],[0,1]])
		else:
			return tf.reshape(tf.tensordot(self.p*tf.expand_dims(self.m,-1),self.W,[[1,2],[0,1]]),self.input_shape)
	def backwardmk(self,k):
                return tf.tensordot(self.p*tf.expand_dims(self.m*tf.expand_dims(1-tf.one_hot(k,self.K),0),-1),self.W,[[1,2],[0,1]])
	def sample(self,M,K=None,sigma=1):
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
                value   = 1/(self.next_layer_sigmas2[0]+eps)+tf.reduce_sum(self.p_*tf.expand_dims(mynorm(self.W,2),1),2)/(eps+self.sigmas2[0])
                new_v2  = tf.assign(self.v2_,1/value)
                return [new_v2]
        def update_vmpk(self):
                k       = self.k_
                proj    = tf.tensordot(self.input_-self.backwardmk(k),self.W[k],[[1],[1]])/(eps+self.sigmas2[0]) #(N R)
                # UPDATE V
                value   = 1/(self.next_layer_sigmas2[0]+eps)+tf.reduce_sum(self.p_*tf.expand_dims(mynorm(self.W,2),1),2)/(eps+self.sigmas2[0])
                new_v2  = tf.assign(self.v2_,1/value)
                # UPDATE M 
                K       = new_v2[k]#1/value
                priorm  = self.next_layer.backward()[:,k]/(eps+self.next_layer.sigmas2[0])
                new_m   = tf.scatter_update(self.m_,[k],[K*(priorm+tf.reduce_sum(proj*self.p_[k],axis=1))])
                # UPDATE P
                prior   = tf.expand_dims(self.pi[k],0)
                m2v2    = -tf.expand_dims(tf.square(new_m[k])+K,-1)*tf.expand_dims(mynorm(self.W[k],1),0)/2 # ( N R )
                V       = tf.nn.softmax(tf.clip_by_value(proj*tf.expand_dims(new_m[k],-1)+m2v2/self.sigmas2[0]+tf.log(prior+0.01),-30,30))
                new_p   = tf.scatter_update(self.p_,[k],[V])
                return [new_v2,new_m,new_p]
        def update_sigma(self):              
                rec     = -utils.SSE(self.input_,self.backward())
                a1      = -tf.reduce_sum(self.input_layer.v2)
                a3      = -tf.reduce_sum((self.v2+tf.square(self.m))*tf.reduce_sum(self.p*tf.expand_dims(mynorm(self.W,2),0),2))
                a2      = tf.reduce_sum(tf.square(self.m)*mynorm(tf.reduce_sum(tf.expand_dims(self.W,0)*tf.expand_dims(self.p,-1),2),2))
                value   = -(rec+a1+a3+a2)/(self.bs*self.D_in)
                new_sigmas2 = tf.assign(self.sigmas2,[tf.clip_by_value(value,0.000001,10)])
                return [new_sigmas2]
        def update_pi(self):
                a44     = tf.reduce_sum(self.p,axis=0)
                new_pi  = tf.assign(self.pi,a44/tf.reduce_sum(eps+a44,axis=1,keepdims=True))
                return [new_pi]
        def update_Wk(self):
                k       = self.k_
                rec     = tf.reduce_sum(tf.expand_dims(self.input_-self.backwardmk(k),1)*tf.expand_dims(tf.expand_dims(self.m_[k],-1)*self.p_[k],-1),0)
                K       = tf.reduce_sum(tf.expand_dims(tf.square(self.m_[k])+self.v2_[k],-1)*self.p_[k],0)
                KK      = rec/(self.sparsity_prior+tf.expand_dims(K,-1))
                new_p   = tf.scatter_update(self.W,[k],[self.Wmask[0]*KK])
                return [new_p]









class ConvLayer:
	def __init__(self,input_layer,Ic,Jc,K,R,stride=1,sparsity_prior = 0,nonlinearity='relu'):
                if(nonlinearity == 'relu'):
                        self.Wmask = tf.reshape(tf.one_hot(1,2),(1,2,1,1,1))
                else:
                        self.Wmask = tf.reshape(tf.ones((2)),(1,2,1,1,1))
		#INPUT_SHAPE : (batch_size,input_filter,M,N)
#		M = input_shape[-2]
#		 = input_shape[-1]
                self.sparsity_prior    = sparsity_prior
                self.input_layer       = input_layer
                input_layer.next_layer = self
                self.bs,self.Iin,self.Jin,self.C  = input_layer.output_shape 
                self.Ic,self.Jc,self.K,self.R     = Ic,Jc,K,R
                self.input        = input_layer.m
                self.input_shape = input_layer.output_shape
		self.output_shape = (self.bs,self.input_shape[-3]/stride,self.input_shape[-2]/stride,K)
#                self.output_shape_= (self.bs,K,R,self.input_shape[-2]/stride,self.input_shape[-1]/stride)
		self.D_in     = prod(self.input_shape[1:])
		self.s        = stride
                self.I,self.J = self.Iin/self.s,self.Jin/self.s
                if(self.s>1):
                    self.padding = 1
                    self.padding_norm = 1
                else:
                    padding  = kaiser(Ic,Ic-1).reshape((-1,1))*kaiser(Jc,Jc-1).reshape((1,-1))**2
                    padding /= padding.sum()
                    self.padding_norm = float32(sum(padding**2))
                    self.padding = tf.Variable(padding.reshape((1,1,1,Ic,Jc,1)).repeat(self.C,5).astype('float32'),trainable=False)
                self.input_  = self.padding*tf.reshape(tf.extract_image_patches(self.input,(1,self.Ic,self.Jc,1),(1,self.s,self.s,1),(1,1,1,1),"SAME"),(self.bs,self.I,self.J,self.Ic,self.Jc,self.C))#N I J Ic Jc C
                self.pi_     = 3.14159
		self.W       = tf.Variable(tf.random_normal((self.K,self.R,self.Ic,self.Jc,self.C)))# (K,R,Ic,Jc,C) always D_in last
		self.pi      = tf.Variable(tf.nn.softmax(tf.random_uniform((K,R)),axis=1))
		self.sigmas2 = tf.Variable(tf.ones(1))
                self.mk      = [tf.Variable(tf.random_uniform((self.bs,self.I,self.J,1))) for k in xrange(self.K)]
		self.m_      = tf.Variable(tf.random_uniform((K,self.bs,self.I,self.J)))#tf.concat(self.mk,axis=3) # (bs,I,J,K)
                self.m       = tf.transpose(self.m_,[1,2,3,0])
                self.pk      = [tf.Variable(tf.nn.softmax(tf.random_uniform((self.bs,self.I,self.J,1,self.R)),axis=4)) for k in xrange(self.K)]
		self.p_      = tf.Variable(tf.nn.softmax(tf.random_uniform((K,self.bs,self.I,self.J,self.R)),axis=4))#tf.concat(self.pk,axis=3) # (bs,I,J,K,R)
                self.p       = tf.transpose(self.p_,[1,2,3,0,4])
                self.v2_     = tf.Variable(tf.random_uniform((self.K,self.bs,self.I,self.J)))
                self.v2      = tf.transpose(self.v2_,[1,2,3,0])
                input_layer.next_layer = self
                # PREPARE THE DECONV OPERATION
                x      = tf.random_uniform((self.bs,self.R,self.input_shape[1],self.input_shape[2],self.input_shape[3]))
                #takes as input size (N,R,Iin,Jin,Cin) and filter (R Ic,Jc,C,K)
                y      = tf.nn.conv3d(x,tf.transpose(self.W,[1,2,3,4,0]),padding="SAME",strides=[1,1,self.s,self.s,1],data_format="NDHWC")
                self.deconv_ = lambda v:tf.reduce_sum(tf.gradients(y,x,v)[0],axis=1)
                self.k_ = tf.placeholder(tf.int32)
#                                           ----  INITIALIZER    ----
        def init_thetaq(self,alpha=0.5):
                # FIRST UPDATE P
                proj        = tf.transpose(tf.tensordot(self.input_,self.W,[[3,4,5],[2,3,4]]),[3,0,1,2,4]) #(K N I J R)
                proj_renorm = proj - 0.5*tf.reshape(mynorm(self.W,[2,3,4]),[self.K,1,1,1,self.R])
                proj_random = tf.random_uniform((self.K,self.bs,self.I,self.J,self.R),-0.1,0.1)
                new_p       = tf.assign(self.p_,tf.nn.softmax(tf.reshape(tf.log(self.pi),[self.K,1,1,1,self.R])+alpha*proj_random+(1-alpha)*proj,axis=4))
                # GIVEN P WE CAN NOW UPDATE V VIA STANDARD FORMULA WITHOUT NEXT LAYER 
                new_v       = self.update_v2()[0]
                # WITH P and V2 WE NOW UPDATE M WITHOUT THE NEXT LAYER PRIOR AND WITHOUT COLLATERAL INFORMATION
                projm       = tf.reduce_sum(proj*new_p,4) #(K N I J)
                new_m       = tf.assign(self.m_,projm/new_v)
                return [new_m,new_p,new_v]
        def init_theta(self):
                init   = tf.random_normal
                new_W  = tf.assign(self.W,self.Wmask*init((self.K,self.R,self.Ic,self.Jc,self.C))) 
                new_pi = tf.assign(self.pi,tf.fill([self.K,self.R],1.0/self.R)) # UNIFORM PRIOR
                new_sigma = tf.assign(self.sigmas2,tf.stack([float32(1)]))             # Identity init.
                return [new_W,new_pi,new_sigma]
#                                           ---- BACKWARD OPERATOR ---- 
        def deconv(self,v=None):
                if(v==None):
                        v = tf.expand_dims(self.m,-1)*self.p # (N I J K R)
                v = tf.transpose(v,[0,4,1,2,3])#TO GET IT (N,R,I,J,K)
                return self.deconv_(v)
	def backward(self):
                return tf.tensordot(tf.expand_dims(self.m,-1)*self.p,self.W,[[3,4],[0,1]])#(N,I,J,Ic,Jc,C)
        def backwardmk(self,k):
                mmk  = self.m*(1-tf.reshape(tf.one_hot(k,self.K),(1,1,1,self.K))) # (bs,I,J,K)
                return tf.tensordot(tf.expand_dims(mmk,-1)*self.p,self.W,[[3,4],[0,1]])#(bs,I,J,Ic,Jc,C)
        def backwardk(self,k):#tf.tensordot(tf.expand_dims(self.mk[k][:,:,:,0],-1)*self.pk[k][:,:,:,0,:],self.W[k],[[3],[0]])
            return tf.tensordot(tf.expand_dims(self.m_[k],-1)*self.p_[k],self.W[k],[[3],[0]])#(N,I,J,Ic,Jc,C)
	def sample(self,M,K=None,sigma=1):
		#multinomial returns [K,n_samples] with integer value 0,...,R-1
		noise = sigma*tf.random_normal(self.input_shape)*tf.sqrt(self.sigmas2[0])
                if(K==None):
                    K    = tf.reshape(tf.one_hot(tf.multinomial(tf.log(self.pi),self.bs*self.I*self.J),self.R),(self.K,self.bs,self.I,self.J,self.R))
                return self.deconv(tf.expand_dims(M,-1)*tf.transpose(K,[1,2,3,0,4]))+noise
        def likelihood(self):
                v_patch   = tf.reshape(tf.extract_image_patches(self.input_layer.v2,(1,self.Ic,self.Jc,1),(1,self.s,self.s,1),(1,1,1,1),"SAME"),(self.bs,self.I,self.J,self.Ic,self.Jc,self.C))#N I J Ic Jc C
                rec_patch = self.backward()
                a1 = -tf.reduce_sum(mynorm(self.input_-rec_patch ,[3,4,5]))
                a2 = -tf.reduce_sum(mynorm(self.padding*tf.sqrt(v_patch),[3,4,5]))
                a3 =  tf.add_n([tf.reduce_sum(mynorm(self.backwardk(k),[3,4,5])) for k in xrange(self.K)])
                a4 = -tf.reduce_sum(tf.reduce_sum(tf.reshape(mynorm(self.W,[2,3,4]),(1,1,1,self.K,self.R))*self.p,axis=4)*(tf.square(self.m)+self.v2))
                k1 = -self.bs*self.D_in*tf.log(self.sigmas2[0]*2*self.pi_)/2
                k2 = tf.reduce_sum(tf.log(self.pi+eps)*tf.reduce_sum(self.p,(0,1,2)))
                return k1+k2+(a1+a2+a3+a4)/(2*self.sigmas2[0])
        def KL(self):
                return self.likelihood()-tf.reduce_sum(self.p*(tf.log(self.p+eps)))+tf.reduce_sum(tf.log(2*self.pi_*self.v2+eps)/2)
        def init_latent(self):
                new_m  = tf.assign(self.m_,tf.random_uniform((self.K,self.bs,self.I,self.J))*0.01)
                new_p  = tf.assign(self.p_,tf.nn.softmax(tf.random_uniform((self.K,self.bs,self.I,self.J,self.R)),axis=4))
                new_v2 = tf.assign(self.v2,tf.random_uniform((self.bs,self.I,self.J,self.K)))
                return [new_v2,new_m,new_p]
        def update_v2(self):# DONE
                value  = tf.reduce_sum(tf.reshape(mynorm(self.W,[2,3,4]),(self.K,1,1,1,self.R))*self.p_,axis=4)# (K,bs,I,J)
                if(isinstance(self.next_layer,ConvLayer)):
                        v2     = self.next_layer.padding_norm/self.next_layer.sigmas2[0]+value/self.sigmas2[0]
                else:
                        v2     = 1/self.next_layer.sigmas2[0]+value/self.sigmas2[0]
                new_v2 = tf.assign(self.v2_,1/v2)
                return [new_v2]
        def update_vmpk(self):# DONE
                k      = self.k_
                back_w = self.backwardmk(k)
                proj   = tf.tensordot(self.input_-back_w,self.W[k],[[3,4,5],[1,2,3]]) # (N I J R)
                ######## V
                value  = tf.reduce_sum(tf.reshape(mynorm(self.W[k],[1,2,3]),(1,1,1,self.R))*self.p_[k],axis=3)# (bs,I,J)
                if(isinstance(self.next_layer,ConvLayer)):
                    v2     = self.next_layer.padding_norm/self.next_layer.sigmas2[0]+value/self.sigmas2[0]
                else:
                    v2     = 1/self.next_layer.sigmas2[0]+value/self.sigmas2[0]
                new_v2 = tf.scatter_update(self.v2_,[k],[1/v2])
                return [new_v2]
                ######## M
                if(isinstance(self.next_layer,ConvLayer) or isinstance(self.next_layer,PoolLayer)):
                        prior = self.next_layer.deconv()[:,:,:,k]/self.next_layer.sigmas2[0]
                else:
                        prior = self.next_layer.backward(0)[:,:,:,k]/self.next_layer.sigmas2[0]
                new_value = tf.reduce_sum(proj*self.p_[k],3)/self.sigmas2[0]# (N I J)
                new_m   = tf.scatter_update(self.m_,[k],[(prior+new_value)*new_v2[k]])
#                return [new_v2,new_m]
                ######## P
                a1      = proj*tf.expand_dims(new_m[k],-1)/self.sigmas2[0]# (N I J)
                a2      = -tf.expand_dims(tf.square(new_m[k])+new_v2[k],-1)*tf.reshape(mynorm(self.W[k] ,[1,2,3]),(1,1,1,self.R))/(2*self.sigmas2[0])
                prior   = tf.reshape(self.pi[k],(1,1,1,self.R))
                K       = tf.nn.softmax(tf.clip_by_value(a1+a2+tf.log(prior+0.01),-1,1),axis=3)
                new_p   = tf.scatter_update(self.p_,[k],[K])
                return [new_v2,new_m,new_p]
        def update_Wk(self):
                k=self.k_
                m_patch = tf.reshape(tf.extract_image_patches(self.input,(1,self.Ic,self.Jc,1),(1,self.s,self.s,1),(1,1,1,1),"SAME"),(self.bs,self.I,self.J,self.Ic,self.Jc,self.C))#N I J Ic Jc C
                back_w  = self.backwardmk(k)
                acc  = tf.expand_dims(self.m_[k],-1)*self.p_[k]#()N I J R
                up   = tf.reduce_sum(tf.expand_dims(m_patch*self.padding-back_w,3)*tf.reshape(acc,(self.bs,self.I,self.J,self.R,1,1,1)),(0,1,2))#(R Ic Jc C)
                down = tf.reshape(tf.reduce_sum(self.p_[k]*tf.expand_dims(tf.square(self.m_[k])+self.v2[:,:,:,k],-1),axis=(0,1,2)),(self.R,1,1,1))
                new_w = tf.scatter_update(self.W,[k],[self.Wmask[0]*up/down])
                return [new_w]
        def update_pi(self):
                a44      = tf.reduce_sum(self.p,axis=[0,1,2])
                new_pi   = tf.assign(self.pi,a44/tf.reduce_sum(eps+a44,axis=1,keepdims=True))
                return [new_pi]
        def update_sigma(self):
                v_patch   = tf.reshape(tf.extract_image_patches(self.input_layer.v2,(1,self.Ic,self.Jc,1),(1,self.s,self.s,1),(1,1,1,1),"SAME"),(self.bs,self.I,self.J,self.Ic,self.Jc,self.C))#N I J Ic Jc C
                rec_patch = self.backward()
                a1 = -tf.reduce_sum(mynorm(self.input_-rec_patch ,[3,4,5]))
                a2 = -tf.reduce_sum(mynorm(self.padding*tf.sqrt(v_patch),[3,4,5]))
                a3 = tf.add_n([tf.reduce_sum(mynorm(self.backwardk(k),[3,4,5])) for k in xrange(self.K)])
                a4 = -tf.reduce_sum(tf.reduce_sum(tf.reshape(mynorm(self.W,[2,3,4]),(1,1,1,self.K,self.R))*self.p,axis=4)*(tf.square(self.m)+self.v2))
                value = (a1+a2+a3+a4)/(self.bs*self.D_in)
                new_sigma = tf.assign(self.sigmas2,[tf.clip_by_value(-value,0.00001,10)])
                return [new_sigma]







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
		self.pi      = tf.Variable(tf.nn.softmax(tf.random_uniform([self.R])))
		self.sigmas2 = tf.Variable(tf.ones(1))
		self.m_      = tf.Variable(tf.random_uniform((self.C,self.bs,self.I,self.J))) # (C,bs,I,J)
                self.m       = tf.transpose(self.m_,[1,2,3,0])          # (bs,I,J,C)
		self.p_      = tf.Variable(tf.nn.softmax(tf.random_uniform((self.C,self.bs,self.I,self.J,self.R)),axis=4))#tf.concat(self.pk,axis=3) # (C,bs,I,J,R)
                self.p       = tf.transpose(self.p_,[1,2,3,0,4])        # (bs,I,J,C,R)
                self.v2_     = tf.Variable(tf.random_uniform((self.C,self.bs,self.I,self.J))) # (C,bs,I,J)
                self.v2      = tf.transpose(self.v2_,[1,2,3,0])         # (bs,I,J,C)
                # PREPARE THE DECONV OPERATION
#                x      = tf.random_uniform(self.input_shape)
                w      = tf.expand_dims(tf.transpose(self.W,[1,2,0]),2)
                self.deconv_ = lambda z: tf.concat([tf.nn.conv2d_backprop_input([self.bs,self.Iin,self.Jin,1],w,z[:,:,:,c,:],[1,self.stride,self.stride,1],"VALID") for c in xrange(self.C)],axis=3)
#                                           ----  INITIALIZER    ----
        def init_thetaq(self,alpha=0.5):
                # FIRST UPDATE P
                proj        = tf.transpose(tf.tensordot(self.input_,self.W,[[3,4],[1,2]]),[3,0,1,2,4]) #(K N I J R)
                proj_renorm = proj - 0.5
                proj_random = tf.random_uniform((self.C,self.bs,self.I,self.J,self.R),-0.1,0.1)
                new_p       = tf.assign(self.p_,tf.nn.softmax(tf.reshape(tf.log(self.pi),[1,1,1,1,self.R])+alpha*proj_random+(1-alpha)*proj,axis=4))
                # GIVEN P WE CAN NOW UPDATE V VIA STANDARD FORMULA WITHOUT NEXT LAYER 
                new_v       = self.update_v2()[0]
                # WITH P and V2 WE NOW UPDATE M WITHOUT THE NEXT LAYER PRIOR AND WITHOUT COLLATERAL INFORMATION
                projm       = tf.reduce_sum(proj*new_p,4) #(K N I J)
                new_m       = tf.assign(self.m_,projm/new_v)
                return [new_m,new_p,new_v]
        def init_theta(self):
                init   = tf.contrib.layers.xavier_initializer()
                new_pi = tf.assign(self.pi,tf.fill([self.R],1.0/self.R)) # UNIFORM PRIOR
                new_sigma = tf.assign(self.sigmas2,[float32(1)])  # Identity init.
                return [new_pi,new_sigma]
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
                a1 = -tf.reduce_sum(mynorm(self.input_-rec_patch ,[1,2,3,4,5]))
                a2 = -tf.reduce_sum(self.input_layer.v2_)
                a3 = -tf.reduce_sum(self.v2_)
                a4 = tf.reduce_sum(self.m_*(1-mynorm(self.p_,4)))
                k1 = -self.bs*self.D_in*tf.log(self.sigmas2[0]*2*self.pi_)/2
                k2 = tf.reduce_sum(tf.log(self.pi+eps)*tf.reduce_sum(self.p_,(0,1,2,3)))
                return k1+k2+(a1+a2+a3+a4)/(2*self.sigmas2[0])
        def KL(self):
                return self.likelihood()-tf.reduce_sum(self.p*(tf.log(self.p+eps)))+tf.reduce_sum(tf.log(2*self.pi_*self.v2+eps)/2)
        def update_v2(self):# DONE
                v2     = 1/self.next_layer.sigmas2[0]+1/self.sigmas2[0]
                new_v2 = tf.assign(self.v2_,tf.ones_like(self.v2_)/v2)
                return [new_v2]
        def update_vmpk(self):# DONE
                back_w  = self.backward()
                proj    = tf.transpose(tf.tensordot(self.input_-back_w,self.W,[[3,4],[1,2]]),[3,0,1,2,4])/self.sigmas2[0] # (N I J C R) -> (C N I J R)
                ####### V
                new_v2 = self.update_v2()[0]
                ####### M
                if(isinstance(self.next_layer,ConvLayer)):
                        priorm = self.next_layer.deconv()/self.next_layer.sigmas2[0]
                else:
                        priorm = self.next_layer.backward(0)/self.next_layer.sigmas2[0]
                new_value = tf.reduce_sum(proj*self.p_,4) # (C N I J)
                new_m   = tf.assign(self.m_,(tf.transpose(priorm,[3,0,1,2])+new_value)*new_v2)
                ####### P
                a1      = proj*tf.expand_dims(new_m,-1)# (C N I J R)
                a2      = -tf.expand_dims(tf.square(new_m)+new_v2,-1)/(2*self.sigmas2[0]) # (C N I J)
                K       = tf.nn.softmax(tf.clip_by_value(a1+a2+tf.log(tf.reshape(self.pi,(1,1,1,1,self.R))+0.01),-30,30),axis=4)
                new_p   = tf.assign(self.p_,K)
                return [new_v2,new_m,new_p]
        def update_Wk(self):
                return [None]
        def update_pi(self):
                a44 = tf.reduce_sum(self.p_,axis=[0,1,2,3])
                new_pi   = tf.assign(self.pi,a44/tf.reduce_sum(eps+a44))
                return [new_pi]
        def update_sigma(self):
#                rec_patch = self.backward()
                a1 = utils.SSE(self.input,self.deconv())
                a2 = tf.reduce_sum(self.input_layer.v2_)
                a3 = tf.reduce_sum(self.v2_)
                a4 = tf.reduce_sum(self.m_*(1-mynorm(self.p_,4)))
                value = (a1+a2+a3+a4)/(self.bs*self.D_in)
                new_sigma = tf.assign(self.sigmas2,[value])
                return [new_sigma]




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



class SupFinalLayer:
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
		self.W       = tf.Variable(tf.random_normal((1,R,self.D_in)))
		self.pi      = tf.Variable(tf.nn.softmax(tf.random_normal((1,R))))
		self.sigmas2 = tf.Variable(tf.ones(1))
                self.p_      = tf.Variable(tf.nn.softmax(tf.random_normal((1,self.bs,R)),axis=2))
		self.p       = tf.transpose(self.p_,[1,0,2])
                input_layer.next_layer_sigmas2 = self.sigmas2
#                                           ----  INITIALIZER    ----
        def init_thetaq(self,alpha=0.5):
                return []
        def init_theta(self):
                init   = tf.random_normal
                new_W  = tf.assign(self.W,init((self.K,self.R,self.D_in))/sqrt(self.D_in)) 
                new_pi = tf.assign(self.pi,tf.fill([self.K,self.R],1.0/self.R)) # UNIFORM PRIOR
                new_sigma = tf.assign(self.sigmas2,tf.stack([float32(1)]))             # Identity init.
                return [new_W,new_pi,new_sigma]
#                                           ---- BACKWARD OPERATOR ---- 
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
                if(K==None):
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
                return self.likelihood()#-tf.reduce_sum(self.p*tf.log(self.p+eps))
        def init_latent(self):
                new_p = tf.assign(self.p,tf.nn.softmax(tf.random_uniform((self.bs,1,self.R),-0.1,0.1),axis=2))
                return []
        def update_v2(self):
                return []
        def update_v2k(self):
                return [None]
        def update_vmpk(self):
                return []
        def update_sigma(self):
                rec = -utils.SSE(self.input_,self.backward())
                a1  = -tf.reduce_sum(self.input_layer.v2)
                a3  = -tf.reduce_sum(tf.reduce_sum(self.p*tf.expand_dims(mynorm(self.W,2),0),2))
                a2  = tf.reduce_sum(mynorm(tf.reduce_sum(tf.expand_dims(self.W,0)*tf.expand_dims(self.p,-1),2),2))
                value =-(rec+a1+a2+a3)/(self.bs*self.D_in)
                new_sigmas2 = tf.assign(self.sigmas2,[value])
                return [new_sigmas2]
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









class UnsupFinalLayer:
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
		self.W       = tf.Variable(tf.random_normal((1,R,self.D_in)))
		self.pi      = tf.Variable(tf.nn.softmax(tf.random_normal((1,R))))
		self.sigmas2 = tf.Variable(tf.ones(1))
                self.p_      = tf.Variable(tf.nn.softmax(tf.random_normal((1,self.bs,R)),axis=2))
		self.p       = tf.transpose(self.p_,[1,0,2])
                input_layer.next_layer_sigmas2 = self.sigmas2
#                                           ----  INITIALIZER    ----
        def init_thetaq(self,alpha=0.5):
                # FIRST UPDATE P
                proj        = tf.transpose(tf.tensordot(self.input_,self.W,[[1],[2]])-0.5*tf.expand_dims(mynorm(self.W,[2]),0),[1,0,2])
                proj_random = tf.random_uniform((self.K,self.bs,self.R),-0.1,0.1)
                new_p       = tf.assign(self.p_,tf.nn.softmax(tf.expand_dims(tf.log(self.pi),1)+alpha*proj_random+(1-alpha)*proj,axis=2))
                return [new_p]
        def init_theta(self):
                init   = tf.random_normal
                new_W  = tf.assign(self.W,init((self.K,self.R,self.D_in))) 
                new_pi = tf.assign(self.pi,tf.fill([self.K,self.R],1.0/self.R)) # UNIFORM PRIOR
                new_sigma = tf.assign(self.sigmas2,tf.stack([float32(1)]))             # Identity init.
                return [new_W,new_pi,new_sigma]
#                                           ---- BACKWARD OPERATOR ---- 
        def backward(self,flat=1):
		if(flat):
                        return tf.tensordot(self.p,self.W,[[1,2],[0,1]])
		else:
			return tf.reshape(tf.tensordot(self.p,self.W,[[1,2],[0,1]]),self.input_shape)
        def sample(self,samples, K=None,sigma=1):
                """ K must be a pre imposed region used for generation
                if not given it is generated according to pi, its shape 
                must be (N K R) with a one hot vector on the last dimension
                sampels is a dummy variable not used in this layer   """
                noise = sigma*tf.random_normal(self.input_shape)*tf.sqrt(self.sigmas2[0])
                if(K==None):
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
                return self.likelihood()-tf.reduce_sum(self.p*tf.log(self.p+eps))
        def init_latent(self):
                new_p = tf.assign(self.p,tf.nn.softmax(tf.random_uniform((self.bs,1,self.R),-0.1,0.1),axis=2))
                return [new_p]
        def update_v2(self):
                return []
        def update_v2k(self):
                return [None]
        def update_vmpk(self):
                a2    = tf.tensordot(self.input_,self.W,[[1],[2]])-tf.expand_dims(mynorm(self.W,2)/2,0) # (N K R)
                V     = tf.nn.softmax(a2/self.sigmas2[0]+tf.log(tf.expand_dims(self.pi,0)),axis=2)
                new_p = tf.assign(self.p_,tf.transpose(V,[1,0,2]))
                return [new_p]
        def update_sigma(self):
                rec = -utils.SSE(self.input_,self.backward())
                a1  = -tf.reduce_sum(self.input_layer.v2)
                a3  = -tf.reduce_sum(tf.reduce_sum(self.p*tf.expand_dims(mynorm(self.W,2),0),2))
                a2  = tf.reduce_sum(mynorm(tf.reduce_sum(tf.expand_dims(self.W,0)*tf.expand_dims(self.p,-1),2),2))
                value =-(rec+a1+a2+a3)/(self.bs*self.D_in)
                new_sigmas2 = tf.assign(self.sigmas2,tf.stack([value]))
                return [new_sigmas2]
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












