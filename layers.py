import tensorflow as tf
from pylab import *
import utils

eps = 0.000000000000001

tf.get_collection('latent')
tf.get_collection('params')



def compute_WmpWmp(layer):
	Wmp=tf.reduce_sum(tf.expand_dims(layer.W,0)*tf.expand_dims(layer.m*layer.p,-1),axis=2)# (N,K,D)
	masked = tf.expand_dims(Wmp,1)*(1-tf.reshape(tf.eye(layer.K),(1,layer.K,layer.K,1)))#(N,K,K,D)
	return tf.reduce_sum(tf.expand_dims(Wmp,1)*masked)


def mynorm(W,axis):
    return tf.reduce_sum(tf.pow(W,2),axis=axis)




class DenseLayer:
	def __init__(self,input_layer,K,R):
		#INPUT SHAPE (batch_size,C)
                self.input_layer = input_layer
                input_layer.next_layer = self
                self.input_shape  = input_layer.output_shape
                self.bs           = int32(self.input_shape[0])
                self.output_shape = (self.bs,K)
                self.D_in        = prod(self.input_shape[1:])
                self.D_out       = K
                self.input_shape_ = (self.bs,self.D_in)#potentially different if flattened
                self.input   = input_layer.m
		if(len(self.input_shape)>2):
                        self.is_flat = False
                        self.input_  = tf.reshape(self.input,(self.bs,self.D_in))
		else:
                        self.is_flat = True
                        self.input_  = self.input
		self.R       = R
                self.K       = K
#                self.k       = tf.Variable(int32(0))
		self.W       = tf.Variable(tf.random_normal((K,R,self.D_in)))
		self.pi      = tf.Variable(tf.nn.softmax(tf.random_normal((K,R))*0.01))
#		self.pi      = tf.nn.softmax(self.pi_)+eps
		self.sigmas2 = tf.Variable(tf.ones(1)*0.15)
#		self.sigmas2 = tf.pow(self.sigmas2_,2)#+0.001
		self.m       = tf.Variable(tf.random_normal((self.bs,K)))
		self.p       = tf.Variable(tf.nn.softmax(tf.random_normal((self.bs,K,R)),axis=2))
#		self.p       = tf.nn.softmax(self.p_)
                self.v2      = tf.Variable(tf.random_uniform((self.bs,K)))
#                self.M       = tf.reduce_sum(self.m*self.p,axis=2)
#                tf.add_to_collection('latent',self.p_)
                tf.add_to_collection('latent',self.m)
#                tf.add_to_collection('params',self.pi_)
                tf.add_to_collection('params',self.W)
#                tf.add_to_collection('params',self.sigmas2_)
                input_layer.next_layer_sigmas2 = self.sigmas2
	def backward(self,flat=1):
		if(flat):
                        return tf.tensordot(self.p*tf.expand_dims(self.m,-1),self.W,[[1,2],[0,1]])
		else:
			return tf.reshape(tf.tensordot(self.p*tf.expand_dims(self.m,-1),self.W,[[1,2],[0,1]]),self.input_shape)
	def backwardmk(self,k):
                return tf.tensordot(self.p*tf.expand_dims(self.m*tf.expand_dims(1-tf.one_hot(k,self.K),0),-1),self.W,[[1,2],[0,1]])
	def sample(self,M):
		noise = tf.random_normal((self.bs,self.input_shape[1]))*tf.sqrt(self.sigmas2[0]) 
		K = tf.one_hot(tf.transpose(tf.multinomial(self.pi,self.bs)),self.R)
		if(self.is_flat):
			return tf.tensordot(tf.expand_dims(M,-1)*K,self.W,[[1,2],[0,1]])+noise
		else:
                        return tf.reshape(tf.tensordot(tf.expand_dims(M,-1)*K,self.W,[[1,2],[0,1]])+noise,self.input_shape)
        def likelihood1(self):
                rec = -utils.SSE(self.input,self.backward(0))/(2*self.sigmas2[0])
                a1  = -self.bs*self.D_in*(tf.log(self.sigmas2[0]*2*3.14159)/2)
                a2  = tf.reduce_sum(tf.reduce_sum(self.p,axis=0)*tf.log(self.pi+eps))
                a31 = 1/(2*self.next_layer_sigmas2[0])+tf.reduce_sum(self.W*self.W,axis=2)/(2*self.sigmas2[0])
                a3  = tf.reduce_sum(tf.reduce_sum(tf.pow(self.m,2)*(tf.pow(self.p,2)-self.p),axis=0)*a31)
                a4  = -tf.reduce_sum((self.v2*a31)*tf.reduce_sum(self.p,axis=0))
                return rec+a1+a2+a3+a4
        def likelihood(self):
                rec = -utils.SSE(self.input,self.backward(0))
                a1  = -self.bs*self.D_in*(tf.log(self.sigmas2[0]*2*3.14159)/2)
                a2  = tf.reduce_sum(tf.reduce_sum(self.p,axis=0)*tf.log(self.pi+eps))
                a3  = -tf.reduce_sum(self.input_layer.v2)
                a4  = -tf.reduce_sum((self.v2+tf.pow(self.m,2))*tf.reduce_sum(self.p*tf.expand_dims(mynorm(self.W,2),0),2))
                a5  = tf.reduce_sum(tf.pow(self.m,2)*mynorm(tf.reduce_sum(tf.expand_dims(self.W,0)*tf.expand_dims(self.p,-1),2),2))
                return a1+(rec+a1+a2+a3+a4+a5)/(2*self.sigmas2[0])
        def KL(self):
                return self.likelihood()-tf.reduce_sum(self.p*tf.log(self.p+eps))+tf.reduce_sum(tf.log(self.v2+eps)/2)
        def init_latent(self):
                new_m  = tf.assign(self.m,tf.random_uniform((self.bs,self.K),-0.1,0.1))
                new_p  = tf.assign(self.p,tf.nn.softmax(tf.random_uniform((self.bs,self.K,self.R),-0.1,0.1),axis=2))
                new_v  = tf.assign(self.v2,tf.random_uniform((self.bs,self.K)))
                return [new_m,new_p,new_v]
        def update_v2(self):
                value   = 1/(self.next_layer_sigmas2[0]+eps)+tf.reduce_sum(self.p*tf.expand_dims(mynorm(self.W,2),0),2)/(eps+self.sigmas2[0])
                K       = 1/value
                new_v2  = tf.assign(self.v2,K)
                return [new_v2]
        def update_m(self):
                value   = 1/(self.next_layer_sigmas2[0]+eps)+tf.reduce_sum(self.p*tf.expand_dims(mynorm(self.W,2),0),2)/(eps+self.sigmas2[0])
                K       = 1/value
                prior   = self.next_layer.backward()/(eps+self.next_layer.sigmas2[0])
                proj    = tf.reduce_sum(tf.expand_dims(self.input_-self.backward(),1)*tf.reduce_sum(tf.expand_dims(self.W,0)*tf.expand_dims(self.p,-1),axis=2),axis=2)
                cor     = self.m*mynorm(tf.reduce_sum(tf.expand_dims(self.W,0)*tf.expand_dims(self.p,-1),axis=2),2)
                new_m = tf.assign(self.m,K*(prior+(proj+cor)/(eps+self.sigmas2[0])))
                return [new_m]
        def update_mk(self,k):
                value   = 1/(self.next_layer_sigmas2[0]+eps)+tf.reduce_sum(self.p[:,k]*tf.expand_dims(mynorm(self.W[k],1),0),1)/(eps+self.sigmas2[0])#(N)
                K       = 1/value
                prior   = self.next_layer.backward()[:,k]/(eps+self.next_layer.sigmas2[0])
                proj    = tf.reduce_sum((self.input_-self.backwardmk(k))*tf.reduce_sum(tf.expand_dims(self.W[k],0)*tf.expand_dims(self.p[:,k],-1),axis=1),axis=1)/(eps+self.sigmas2[0])
                indices = tf.transpose(tf.stack([tf.range(self.bs),tf.constant(k,shape=[self.bs])]))
                new_m = tf.scatter_nd_update(self.m,indices,K*(prior+proj),use_locking=False)
                return [new_m]
        def update_p(self):#HEHRHEHRHEHRHEHRHEHR
                a00 = tf.reduce_sum(tf.expand_dims(self.p,-1)*tf.expand_dims(tf.reduce_sum(tf.expand_dims(self.W,1)*tf.expand_dims(self.W,2),3),0),axis=2)
                a0  = a00*tf.expand_dims(tf.pow(self.m,2),-1)
                a1  = -tf.expand_dims(tf.pow(self.m,2)+self.v2,-1)*tf.expand_dims(mynorm(self.W,2),0)/2
                a2  = tf.tensordot(self.input_-self.backward(),self.W,[[1],[2]])*tf.expand_dims(self.m,-1)
                V = tf.exp(tf.clip_by_value(a0+a1+a2,-10,5)/self.sigmas2[0])*tf.expand_dims(self.pi,0)
                new_p = tf.assign(self.p,V/tf.reduce_sum(V,axis=2,keep_dims=True))
                return [new_p]
        def update_pk(self,k):
                prior   = tf.expand_dims(self.pi[k],0)#self.next_layer.backward()[:,k]/(eps+self.next_layer.sigmas2[0])
                proj    = tf.tensordot(self.input_-self.backwardmk(k),self.W[k],[[1],[1]])*tf.expand_dims(self.m[:,k],-1)
                m2v2    = -tf.expand_dims((tf.pow(self.m[:,k],2)+self.v2[:,k]),-1)*tf.expand_dims(mynorm(self.W[k],1),0)/2
                indices = tf.transpose(tf.stack([tf.range(self.bs),tf.constant(k,shape=[self.bs])]))
                V       = tf.exp(tf.clip_by_value(proj+m2v2,-7,4)/self.sigmas2[0])*prior
                new_m   = tf.scatter_nd_update(self.p,indices,V/tf.reduce_sum(V,axis=1,keep_dims=True),use_locking=False)
                return [new_m]
        def update_sigma(self):              
                rec = -utils.SSE(self.input_,self.backward())
#                a2  = tf.reduce_sum(tf.reduce_sum(self.p,axis=0)*tf.log(self.pi+eps))
                a1  = -tf.reduce_sum(self.input_layer.v2)
                a3  = -tf.reduce_sum((self.v2+tf.pow(self.m,2))*tf.reduce_sum(self.p*tf.expand_dims(mynorm(self.W,2),0),2))
                a2  = tf.reduce_sum(tf.pow(self.m,2)*mynorm(tf.reduce_sum(tf.expand_dims(self.W,0)*tf.expand_dims(self.p,-1),2),2))
                value =-(rec+a1+a3+a2)/(self.bs*self.D_in)
                new_sigmas2 = tf.assign(self.sigmas2,tf.stack([value]))
                return [new_sigmas2]
        def update_pi(self):
                a44          = tf.reduce_sum(self.p,axis=0)
                new_pi      = tf.assign(self.pi,a44/tf.reduce_sum(eps+a44,axis=1,keep_dims=True))
                return [new_pi]
        def update_W(self):
#                return []
                rec     = tf.reshape(self.input_-self.backward(),(self.bs,1,1,self.D_in))
                recplus = tf.expand_dims(self.m,-1)*tf.reduce_sum(tf.expand_dims(self.W,0)*tf.expand_dims(self.p,-1),axis=2)#(N K  D)
                rec2    = tf.reduce_mean((rec+tf.expand_dims(recplus,2))*tf.expand_dims(tf.expand_dims(self.m,-1)*self.p,-1),0)
#                recplus = tf.reduce_mean(tf.reduce_sum(tf.expand_dims(self.W,0)*tf.expand_dims(self.p,-1),axis=2,keep_dims=True)*tf.expand_dims(tf.expand_dims(tf.pow(self.m,2),-1)*self.p,-1),0)
                K       = tf.reduce_mean(tf.expand_dims(tf.pow(self.m,2)+self.v2,-1)*self.p,0)
                KK = (rec2)/(0.000001+tf.expand_dims(K,-1))
                new_W   = tf.assign(self.W,KK)
                return [new_W]#,new_pi]
        def update_Wk(self,k):
#                return []
                rec     = tf.reduce_sum(tf.reshape(self.input_-self.backwardmk(k),(self.bs,1,self.D_in))*tf.expand_dims(tf.expand_dims(self.m[:,k],-1)*self.p[:,k],-1),0)
                K       = tf.reduce_sum(tf.expand_dims(tf.pow(self.m[:,k],2)+self.v2[:,k],-1)*self.p[:,k],0)
                KK      = rec/(0.000001+tf.expand_dims(K,-1))
                indices = tf.transpose(tf.stack([tf.constant(k,shape=[self.R]),tf.range(self.R)]))
                new_p   = tf.scatter_nd_update(self.W,indices,KK,use_locking=False)
                return [new_p]#,new_pi]





class ConvLayer:
	def __init__(self,input_layer,I,J,K,R,stride=1):
		#INPUT_SHAPE : (batch_size,input_filter,M,N)
#		M = input_shape[-2]
#		N = input_shape[-1]
                self.input        = input_layer.M
                self.input_shape  = input_layer.output_shape
                self.bs           = self.input_shape[0]
		self.output_shape = (self.bs,K,self.input_shape[-2]/stride,self.input_shape[-1]/stride)
                self.output_shape_= (self.bs,K,R,self.input_shape[-2]/stride,self.input_shape[-1]/stride)
#		self.C       = input_shape[1]
		self.D_in    = prod(self.input_shape[1:])
		self.R       = R
		self.s       = stride
		self.K       = K
		self.W       = tf.Variable(tf.random_uniform((R,I,J,self.input_shape[1],self.K),-0.3,0.3))
		self.pi      = tf.Variable(tf.random_uniform((K,R),-0.3,0.3))
#		self.pi      = tf.nn.softmax(self.pi_)+eps
		self.sigmas2 = tf.Variable(tf.ones(1))
#		self.sigmas2 = tf.pow(self.sigmas2_,2)+eps
		self.m       = tf.Variable(tf.random_uniform(self.output_shape_,-0.1,0.1))
		self.p_      = tf.Variable(tf.random_uniform(self.output_shape_,-0.1,0.1))
                self.v2      = tf.Variable(tf.random_uniform(self.output_shape_[1:],-0.1,0.1))
		self.p       = tf.nn.softmax(self.p_,axis=2)
		self.M       = tf.reduce_sum(self.m*self.p,axis=2)
                tf.add_to_collection('latent',self.p_)
                tf.add_to_collection('latent',self.m)
#                tf.add_to_collection('params',self.pi_)
                tf.add_to_collection('params',self.W)
#                tf.add_to_collection('params',self.sigmas2_)
                input_layer.next_layer_sigmas2 = self.sigmas2
        def deconv(self,v):
                x      = tf.random_uniform((self.bs,self.input_shape[1],self.R,self.input_shape[-2],self.input_shape[-1]))
                y      = tf.nn.conv3d(x,self.W,padding="SAME",strides=[1,1,1,self.s,self.s],data_format="NCDHW")
                return tf.gradients(y,x,v)[0]
	def backward(self):
                return tf.reduce_sum(self.deconv(self.m*self.p),axis=2)
	def sample(self,M):
		#multinomial returns [K,n_samples] with integer value 0,...,R-1
		noise = tf.random_normal(self.input_shape)*tf.sqrt(self.sigmas2[0])
                mp    = tf.reshape(tf.one_hot(tf.multinomial(self.pi,self.bs*prod(self.output_shape[2:])),self.R),(self.K,self.bs,self.output_shape[-2],self.output_shape[-1],self.R))#(self.batch_size,self.K,self.M,self.N,self.R)
                return tf.reduce_sum(self.deconv(tf.expand_dims(M,2)*tf.transpose(mp,[1,0,4,2,3])),axis=2)+noise
        def likelihood(self):
                rec = -utils.SSE(self.input,self.backward())/(2*self.sigmas2[0])
                a1  = -self.bs*self.D_in*(tf.log(self.sigmas2[0]*2*3.14159)/2)
                a2  = tf.reduce_sum(self.p*tf.reshape(tf.log(self.pi+eps),(1,self.K,self.R,1,1)))
                a31 = tf.reshape(tf.transpose(tf.reduce_sum(self.W*self.W,axis=[1,2,3])),(1,self.K,self.R,1,1))
                a3  = tf.reduce_sum(tf.reduce_sum((tf.pow(self.m,2))*(tf.pow(self.p,2)-self.p),axis=0)*(1/(2*self.next_layer_sigmas2[0])+a31/(2*self.sigmas2[0])))
                return rec+a1+a2+a3-prod(self.output_shape)
        def KL(self):
                return self.likelihood()-tf.reduce_sum(self.p*(tf.log(self.p+eps)-tf.log(self.v2+eps)/2))
        def init_latent(self):
                new_m  = tf.assign(self.m,tf.random_uniform(self.output_shape_,-0.1,0.1))
                new_p_ = tf.assign(self.m,tf.random_uniform(self.output_shape_,-0.1,0.1))
                return [new_m,new_p_]
        def update_v2(self):
                value = tf.reshape(tf.transpose(tf.reduce_sum(self.W*self.W,axis=[1,2,3])),(1,self.K,self.R,1,1))
                self.v2=tf.stop_gradient(1/(2*self.next_layer_sigmas2[0])+value/(2*self.sigmas2[0]))










class PoolLayer:
        def __init__(self,input_layer,s):
                #INPUT_SHAPE : (batch_size,input_filter,M,N)
                self.input        = input_layer.M
                self.input_shape  = input_layer.output_shape
                self.output_shape = (input_shape[0],input_shape[1],input_shape[-2]/stride,input_shape[-1]/stride)
                self.D_in  = prod(self.input_shape[1:])
#                self.D_out = prod(self.output_shape[1:])
		self.bs    = input_shape[0]
		self.s       = s
		self.sigmas2_= tf.Variable(tf.ones(1))
		self.sigmas2 = tf.pow(self.sigmas2_,2)+eps
		self.m       = tf.Variable(tf.random_uniform(self.input_shape,-0.1,0.1))
		self.p_      = tf.Variable(tf.random_uniform(self.input_shape,-0.1,0.1))
		p            = tf.nn.exp(self.p_)
                self.p       = p/self.deconv(tf.nn.pool(p,[1,1,self.s,self.s],'AVG',"VALID"))
		self.M       = tf.nn.pool(self.m*self.p,[1,1,self.s,self.s],'AVG',"VALID")*(self.s*self.s)
                tf.add_to_collection('latent',self.p_)
                tf.add_to_collection('latent',self.m)
                tf.add_to_collection('params',self.sigmas2_)
                input_layer.next_layer_sigmas2 = self.sigmas2
        def deconv(self,v):
                x   = tf.random_uniform(self.input_shape)
                y      = tf.nn.pool(x,(1,1,self.s,self.s),"AVG","VALID",data_format="NCHW")*(self.s*self.s)
                return tf.gradients(y,x,v)[0]
	def backward(self):
                return self.m*self.p
	def sample(self,M):
		#multinomial returns [K,n_samples] with integer value 0,...,R-1
		noise = tf.random_normal(self.input_shape)*tf.sqrt(self.sigmas2[0])
                nn    = tf.random_normal(self.input_shape)*tf.sqrt(self.sigmas2[0])
                return tf.gradients(tf.nn.pool(nn,[1,1,self.s,self.s],'MAX',"VALID",data_format="NCHW"),nn,M)[0]+noise
        def likelihood(self):
                rec = -utils.SSE(self.input,self.backward())/(2*self.sigmas2[0])
                a1  = -self.bs*self.D_in*(tf.log(self.sigmas2[0]*2*3.14159)/2)+tf.reduce_sum(self.p*tf.log(1.0/(self.s*self.s)))
                a2  = tf.reduce_sum(tf.pow(layers[l].m,2)*(tf.pow(layers[l].p,2)-layers[l].p))*(1/(2*self.sigmas2[0])+1/(2*self.next_layer_sigmas2[0]))
                return rec+a1+a2-prod(self.output_shape)
        def KL(self):
                return self.likelihood()-tf.reduce_sum(self.p*(tf.log(self.p+eps)-tf.log(self.v2+eps)/2))
        def init_latent(self):
                new_m  = tf.assign(self.m,tf.random_uniform(self.input_shape,-0.1,0.1))
                new_p_ = tf.assign(self.m,tf.random_uniform(self.input_shape,-0.1,0.1))
                return [new_m,new_p_]
        def update_v2(self):
                self.v2=tf.stop_gradient(1/(2*self.next_layer_sigmas2[0])+1/(2*self.sigmas2[0]))








class InputLayer:
        def __init__(self,input_shape):
		self.input_shape = input_shape
		self.output_shape = input_shape
                self.m       = tf.Variable(tf.random_normal(self.input_shape))
#		self.M       = self.m
                self.v2      = tf.zeros(self.output_shape)
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





class UnsupFinalLayer:
        def __init__(self,input_layer,R):
                self.input_layer = input_layer
                input_layer.next_layer = self
		self.input_shape  = input_layer.output_shape
                self.bs           = self.input_shape[0]
                self.output_shape = (self.bs,1)
                self.K = 1
                self.D_in         = prod(self.input_shape[1:])
                self.input_shape_ = (self.bs,self.D_in)#potentially different if flattened
                self.input   = input_layer.m
		if(len(self.input_shape)>2):
                        self.is_flat = False
                        self.input_  = tf.reshape(self.input,(self.bs,self.D_in))
		else:
                        self.is_flat = True
                        self.input_  = self.input
		self.bs      = int32(self.input_shape[0])
		self.R       = R
                self.m = float32(1)
		self.W       = tf.Variable(tf.random_normal((1,R,self.D_in)))
		self.pi      = tf.Variable(tf.nn.softmax(tf.random_normal((1,R))*0.01))
#		self.pi      = tf.nn.softmax(self.pi_)+eps
		self.sigmas2 = tf.Variable(tf.ones(1)*0.1151)
#		self.sigmas2 = tf.pow(self.sigmas2_,2)+eps
		self.p       = tf.Variable(tf.nn.softmax(tf.random_normal((self.bs,1,R)),axis=2))
#        	self.p       = tf.nn.softmax(self.p_)
#                tf.add_to_collection('latent',self.p_)
#                tf.add_to_collection('params',self.pi_)
                tf.add_to_collection('params',self.W)
#                tf.add_to_collection('params',self.sigmas2_)
                input_layer.next_layer_sigmas2 = self.sigmas2
	def backward(self,flat=1):
		if(flat):
                        return tf.tensordot(self.p,self.W,[[1,2],[0,1]])
		else:
			return tf.reshape(tf.tensordot(self.p,self.W,[[1,2],[0,1]]),self.input_shape)
        def sample(self,e=0):
                noise = tf.random_normal(self.input_shape)*tf.sqrt(self.sigmas2[0])
                K = tf.one_hot(tf.transpose(tf.multinomial(self.pi,self.bs)),self.R)
		if(self.is_flat):
	                return tf.tensordot(K,self.W,[[1,2],[0,1]])+noise
		else:
                        return tf.reshape(tf.tensordot(K,self.W,[[1,2],[0,1]]),self.input_shape)+noise
        def likelihood(self):
                rec = -utils.SSE(self.input,self.backward(0))
                a1  = -self.bs*self.D_in*(tf.log(self.sigmas2[0]*2*3.14159)/2)
                a2  = tf.reduce_sum(tf.reduce_sum(self.p,axis=0)*tf.log(self.pi+eps))
                a3  = -tf.reduce_sum(self.input_layer.v2)
                a4  = -tf.reduce_sum(tf.reduce_sum(self.p*tf.expand_dims(mynorm(self.W,2),0),2))
                a5  = tf.reduce_sum(mynorm(tf.reduce_sum(tf.expand_dims(self.W,0)*tf.expand_dims(self.p,-1),2),2))
                return a1+(rec+a2+a3+a4+a5)/(2*self.sigmas2[0])
        def KL(self):
                return self.likelihood()-tf.reduce_sum(self.p*tf.log(self.p+eps))
        def init_latent(self):
                new_p = tf.assign(self.p,tf.nn.softmax(tf.random_uniform((self.bs,1,self.R),-0.1,0.1),axis=2))
                return [new_p]
        def update_v2(self):
                return []
        def update_m(self):
                return []
        def update_p(self):
#                a00 = tf.reduce_sum(tf.expand_dims(self.p,-1)*tf.expand_dims(tf.reduce_sum(tf.expand_dims(self.W,1)*tf.expand_dims(self.W,2),3),0),axis=2)
#                a0  = a00
#                a1  = -tf.expand_dims(mynorm(self.W,2),0)/2
                a2  = tf.tensordot(self.input_,self.W,[[1],[2]])-tf.expand_dims(mynorm(self.W,2)/2,0)
                V = tf.exp(a2/self.sigmas2[0])*tf.expand_dims(self.pi,0)
                new_p = tf.assign(self.p,V/tf.reduce_sum(V,axis=2,keep_dims=True))
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
                new_pi      = tf.assign(self.pi,a44/tf.reduce_sum(eps+a44,axis=1,keep_dims=True))
                return [new_pi]
        def update_W(self):
                rec = tf.reduce_mean(tf.reshape(self.input_,(self.bs,1,1,self.D_in))*tf.expand_dims(self.p,-1),0)
#                recplus = tf.reduce_sum(tf.expand_dims(self.p,-1)*tf.expand_dims(tf.reduce_sum(tf.expand_dims(self.p,-1)*tf.expand_dims(self.W,0),axis=2),2),0)
                K   = tf.reduce_mean(self.p,0)
                KK = (rec)/(0.0000001+tf.expand_dims(K,-1))
                new_W  = tf.assign(self.W,KK)
                return [new_W]












