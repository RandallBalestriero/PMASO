import tensorflow as tf
from pylab import *
import utils
eps = 0.00000000000000000

tf.get_collection('latent')
tf.get_collection('params')


class DenseLayer:
	def __init__(self,input_layer,K,R):
		#INPUT SHAPE (batch_size,C)
                self.bs           = int32(input_shape[0])
                self.output_shape = (self.bs,K)
		self.input_shape  = input_layer.output_shape
                self.D_in        = prod(self.input_shape[1:])
                self.D_out       = K
                self.input_shape_ = (self.bs,self.D_in)#potentially different if flattened
		if(len(input_shape)>2):
                        self.is_flat = False
		else:
                        self.is_flat = True
                self.input   = input_layer.M
		self.bs      = int32(input_shape[0])
		self.R       = R
                self.K       = K
		self.W       = tf.Variable(tf.random_uniform((K,R,self.D_in),-0.1,0.1))
		self.pi_     = tf.Variable(tf.random_uniform((K,R),-0.1,0.1))
		self.pi      = tf.nn.softmax(self.pi_)+eps
		self.sigmas2_= tf.Variable(tf.ones(1))
		self.sigmas2 = tf.pow(self.sigmas2_,2)+eps
		self.m       = tf.Variable(tf.random_uniform((self.bs,K,R),-0.1,0.1))
		self.p_      = tf.Variable(tf.random_uniform((self.bs,K,R),-0.1,0.1))
		self.p       = tf.nn.softmax(self.p_)
		self.M       = tf.reduce_sum(self.m*self.p,axis=2)
                tf.add_to_collection('latent',self.p_)
                tf.add_to_collection('latent',self.m)
                tf.add_to_collection('params',self.pi_)
                tf.add_to_collection('params',self.W)
                tf.add_to_collection('params',self.sigmas2_)
                input_layer.next_layer_sigmas2 = self.sigmas2
	def backward(self):
		if(self.is_flat):
                        return tf.tensordot(self.p*self.m,self.W,[[1,2],[0,1]])
		else:
			return tf.reshape(tf.tensordot(self.p*self.m,self.W,[[1,2],[0,1]]),self.input_shape)
	def sample(self,M):
		noise = tf.random_normal((self.bs,self.input_shape[1]))*tf.sqrt(self.sigmas2[0]) 
		K = tf.one_hot(tf.transpose(tf.multinomial(self.pi,self.bs)),self.R)
		if(self.is_flat):
			return tf.tensordot(tf.expand_dims(M,-1)*K,self.W,[[1,2],[0,1]])+noise
		else:
                        return tf.reshape(tf.tensordot(tf.expand_dims(M,-1)*K,self.W,[[1,2],[0,1]])+noise,self.input_shape)
        def likelihood(self):
                rec = -utils.SSE(self.input,self.backward())/(2*self.sigmas2[0])
                a1  = -self.bs*self.D_in*(tf.log(self.sigmas2[0]+eps)/2+tf.log(2*3.14159)/2)
                a2  = tf.reduce_sum(self.p*tf.reshape(tf.log(self.pi+eps),(1,self.K,self.R)))
                a31 = tf.reshape(tf.reduce_sum(self.W*self.W,axis=2),(1,self.K,self.R))
                a3  = tf.reduce_sum(tf.reduce_sum((tf.pow(self.m,2))*(tf.pow(self.p,2)-self.p),axis=0)*(1/(2*self.next_layer_sigmas2[0])+a31/(2*self.sigmas2[0])))
                return rec+a1+a2+a3-prod(self.output_shape)
        def KL(self):
                return self.likelihood()-tf.reduce_sum(self.p*(tf.log(self.p+eps)-tf.log(self.v2+eps)/2))
        def init_latent(self):
                new_m  = tf.assign(self.m,tf.random_uniform((self.bs,self.K,self.R),-0.1,0.1))
                new_p_ = tf.assign(self.m,tf.random_uniform((self.bs,self.K,self.R),-0.1,0.1))
                return [new_m,new_p_]
        def init_v2(self):
                value   = tf.reshape(tf.reduce_sum(self.W*self.W,axis=2),(1,self.K,self.R))
                self.v2 = tf.stop_gradient(1/(2*self.next_layer_sigmas2[0])+value/(2*self.sigmas2[0]))





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
		self.W       = tf.Variable(tf.random_uniform((R,I,J,self.input_shape[1],self.K),-0.1,0.1))
		self.pi_     = tf.Variable(tf.random_uniform((K,R),-0.1,0.1))
		self.pi      = tf.nn.softmax(self.pi_)+eps
		self.sigmas2_= tf.Variable(tf.ones(1))
		self.sigmas2 = tf.pow(self.sigmas2_,2)+eps
		self.m       = tf.Variable(tf.random_uniform(self.output_shape_,-0.1,0.1))
		self.p_      = tf.Variable(tf.random_uniform(self.output_shape_,-0.1,0.1))
		self.p       = tf.nn.softmax(self.p_,axis=2)
		self.M       = tf.reduce_sum(self.m*self.p,axis=2)
                tf.add_to_collection('latent',self.p_)
                tf.add_to_collection('latent',self.m)
                tf.add_to_collection('params',self.pi_)
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
                a1  = -self.bs*self.D_in*(tf.log(self.sigmas2[0]+eps)/2+tf.log(2*3.14159)/2)
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
        def init_v2(self):
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
                a1  = -self.bs*self.D_in*(tf.log(self.sigmas2[0]+eps)/2+tf.log(2*3.14159)/2)+tf.reduce_sum(self.p*tf.log(1.0/(self.s*self.s)))
                a2  = tf.reduce_sum(tf.pow(layers[l].m,2)*(tf.pow(layers[l].p,2)-layers[l].p))*(1/(2*self.sigmas2[0])+1/(2*self.next_layer_sigmas2[0]))
                return rec+a1+a2-prod(self.output_shape)
        def KL(self):
                return self.likelihood()-tf.reduce_sum(self.p*(tf.log(self.p+eps)-tf.log(self.v2+eps)/2))
        def init_latent(self):
                new_m  = tf.assign(self.m,tf.random_uniform(self.input_shape,-0.1,0.1))
                new_p_ = tf.assign(self.m,tf.random_uniform(self.input_shape,-0.1,0.1))
                return [new_m,new_p_]
        def init_v2(self):
                self.v2=tf.stop_gradient(1/(2*self.next_layer_sigmas2[0])+1/(2*self.sigmas2[0]))








class InputLayer:
        def __init__(self,input_shape):
		self.input_shape = input_shape
		self.output_shape = input_shape
                self.m       = tf.Variable(tf.random_normal(self.input_shape))
		self.M       = self.m
        def init_v2(self):
                return None
        def likelihood(self):
                return 0
        def KL(self):
                return 0


class UnsupFinalLayer:
        def __init__(self,input_layer,R):
		self.input_shape  = input_layer.output_shape
                self.bs           = self.input_shape[0]
                self.output_shape = (self.bs,1)
                self.D_in         = prod(self.input_shape[1:])
                self.input_shape_ = (self.bs,self.D_in)#potentially different if flattened
		if(len(self.input_shape)>2):
                        self.is_flat = False
		else:
                        self.is_flat = True
                self.input   = input_layer.M
		self.bs      = int32(self.input_shape[0])
		self.R       = R
		self.W       = tf.Variable(tf.random_uniform((1,R,self.D_in),-0.1,0.1))
		self.pi_     = tf.Variable(tf.random_uniform((1,R),-0.1,0.1))
		self.pi      = tf.nn.softmax(self.pi_)+eps
		self.sigmas2_= tf.Variable(tf.ones(1))
		self.sigmas2 = tf.pow(self.sigmas2_,2)+eps
		self.p_      = tf.Variable(tf.random_uniform((self.bs,1,R),-0.1,0.1))
		self.p       = tf.nn.softmax(self.p_)
                tf.add_to_collection('latent',self.p_)
                tf.add_to_collection('params',self.pi_)
                tf.add_to_collection('params',self.W)
#                tf.add_to_collection('params',self.sigmas2_)
                input_layer.next_layer_sigmas2 = self.sigmas2
        def backward(self):
		if(self.is_flat):
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
                rec = -utils.SSE(self.input,self.backward())/(2*self.sigmas2[0])
                a1  = -self.bs*self.D_in*(tf.log(self.sigmas2[0]+eps)/2+tf.log(2*3.14159)/2)
                a2  = tf.reduce_sum(self.p*tf.reshape(tf.log(self.pi+eps),(1,1,self.R)))
#                if(self.is_flat):
#                        rec = -tf.reduce_sum(tf.reduce_sum(tf.pow(tf.expand_dims(self.input,1)-tf.expand_dims(self.W[0],0),2),axis=2)*self.p[:,0,:])/(2*self.sigmas2[0])
#                else:
#                        rec = -tf.reduce_sum(tf.reduce_sum(tf.pow(tf.expand_dims(tf.reshape(self.M,(self.bs,self.D_in)),1)-tf.expand_dims(layers[-1].W[0],0),2),axis=2)*layers[-1].p[:,0,:])/(2*layers[-1].sigmas2)

                a31 = tf.reshape(tf.reduce_sum(self.W*self.W,axis=2),(1,1,self.R))
                a3  = tf.reduce_sum(tf.reduce_sum(tf.pow(self.p,2)-self.p,axis=0)*(a31/(2*self.sigmas2[0])))
                return rec+a1+a2+a3#-prod(self.output_shape)
        def KL(self):
                return self.likelihood()-tf.reduce_sum(self.p*(tf.log(self.p+eps)))
        def init_latent(self):
                new_p_ = tf.assign(self.p_,tf.random_uniform((self.bs,1,self.R),-0.1,0.1))
                return [new_p_]
        def init_v2(self):
                return None












