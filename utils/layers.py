import tensorflow as tf
from pylab import *
import utils
import itertools
from math import pi as PI_CONST


eps = float32(0.000000001)

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



# Parent class of all the layers implementing the base methods
class Layer:
    def init_W(self):
        return []
    def kwargs_init(self,kwargs):
	keys  = kwargs.keys()
	items = kwargs.items()
	# update_b (bool) is True of False
	if('update_b' in keys): self.update_b = kwargs['update_b']
	else:                   self.update_b = True
	# sparsity_prior (float32) is a positive number
	if('sparsity_prior' in keys): self.sparsity_prior = kwargs['sparsity_prior']
	else:			      self.sparsity_prior = float32(0.0)
	# init_b and init_W (func) can be any initializer instance
	if('init_b' in keys): self.init_b = kwargs['init_b']
	else:		      self.init_b = tf.zeros
	if('init_W' in keys): self.init_W = kwargs['init_W']
	else:		      self.init_W = tf.glorot_normal_initializer()
	# sigma (str) can either be 'diagonal' or 'isotropic'
	if('sigma_opt' in keys): self.sigma_opt = kwargs['sigma_opt']
	else:			 self.sigma_opt = 'diagonal'
	# init_m_data any numpy func
        if('init_m_data' in keys): self.init_m_data = kwargs['init_m_data']
        else:                      self.init_m_data = ones
        # init_v2_data any numpy func
        if('init_v2_data' in keys): self.init_v2_data = kwargs['init_v2_data']
        else:                       self.init_v2_data = ones
        # init_p_data any numpy func
        if('init_p_data' in keys): self.init_p_data = kwargs['init_p_data']
        else:                      self.init_p_data = ones
    def update_BN(self):
        return []
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
    def likelihood(self,E_step=True):
	return float32(0)

#########################################################################################################################
#
#
#                                       DENSE/CONV/POOL LAYERS
#
#
#########################################################################################################################



class DenseLayer(Layer):
    def __init__(self,input_layer,K,**kwargs):
	self.kwargs_init(kwargs)
        self.input_layer       = input_layer
        input_layer.next_layer = self
	self.N                 = input_layer.N
        # RESHAPE IF NEEDED (acting as a flatten layer)
        if(len(self.input_shape)>2):
            self.is_flat     = False
            self.D           = prod(self.input_shape[1:])
            self.input_shape = (self.input_shape[0],self.D)#potentially different if flattened
            self.input       = tf.reshape(self.input_layer.m,self.input_shape)
        else:
            self.is_flat     = True
            self.D           = self.input_shape[-1]
            self.input_shape = (self.input_shape[0],self.D)#potentially different if flattened
            self.input       = self.input
        self.output_shape      = (self.input_shape[0],K)
        # THETA PARAMETERS that are updated during the M step
        self.sigmas2 = tf.Variable(tf.ones(self.D))
        self.W       = tf.Variable(self.init_W((K,self.D)))
	self.b       = tf.Variable(self.init_b(K))
        # THETAQ VARIABLES (Variational Inference) PARAMETERS that are updated during the (VI) E-step
	self.m       = tf.Variable(tf.zeros((self.input_shape[0],K)))
        self.v2      = tf.Variable(tf.zeros((K)))
        # THETAQ DATA for all the dataset, as opposed to the variables which have the size of the batch
	# only need for m as v2 is the same across data (n invariant)
        self.m_data           = self.init_m_data((self.N,K)).astype('float32')
	self.m_placeholder    = tf.placeholder(tf.float32,(self.input_shape[0],K))
	self.thetaq_assign_op = tf.assign(self.m,self.m_placeholder)
	# placeholder for update and indices (to update the parameter self.W, 1 dimension at a time)
        self.k_        = tf.placeholder(tf.int32) # placeholder that will indicate which unit k is being updated
	self.W_indices = asarray(range(K))        # python list of the unit to be updated to complete a cycle
        # LAYER STATISTICS (all the statistics are init at 0 as they are init and updated externally
	self.alpha  = tf.Variable(tf.zeros(1))    # use to compute the geometric mean for the statistics update
        self.Sm2    = tf.Variable(tf.zeros(self.D))
        self.Sv2    = tf.Variable(tf.zeros(self.D))
        self.Sm     = tf.Variable(tf.zeros(self.D))
        self.SM     = tf.Variable(tf.zeros(self.K))
	self.SMm    = tf.Variable(tf.zeros(self.K,self.D))
        self.SMM    = tf.Variable(tf.zeros(self.K,self.K))
    def set_batch(self,session,indices):
	if(len(indices)==self.N): return
	session.run(self.thetaq_assign_op,feed_dict={self.m_placeholder:self.m_data[indices]})
    def save_batch(self,session,indices):
        if(len(indices)==self.N): return
        self.m_data[indices] = session.run(self.m)
    def update_S(self):
	# given the current value of the THETAQ parameters, update
	# the values of the sufficient statistics of the layer
        Sm2    = tf.reduce_sum(tf.square(self.input),0)/self.input_shape[0]
	if(not isinstance(self.input_layer,NonlinearityLayer)):
            Sv2    = tf.reshape(self.input_layer.v2,self.input_shape)
	else:
            Sv2    = tf.reshape(tf.reduce_sum(self.input_layer.v2,0)/self.input_shape[0],self.input_shape)
        Sm     = tf.reduce_sum(self.input,0)/self.input_shape[0]
        SM     = tf.reduce_sum(self.m,0)/self.input_shape[0]
        SMm    = tf.matmul(self.m,self.input,transpose_a=True)/self.input_shape[0]
        SMM    = tf.matmul(self.m,self.m,transpose_a=True)/self.input_shape[0]
        return tf.group(tf.assign(self.Sm2,self.alpha*Sm2+(1-self.alpha)*self.Sm2),tf.assign(self.Sv2,self.alpha*Sv2+(1-self.alpha)*self.Sv2),
                        tf.assign(self.Sm,self.alpha*Sm+(1-self.alpha)*self.Sm),tf.assign(self.SM,self.alpha*SM+(1-self.alpha)*self.SM),
                        tf.assign(self.SMm,self.alpha*SMm+(1-self.alpha)*self.SMm),tf.assign(self.SMM,self.alpha*SMM+(1-self.alpha)*self.SMM))
    def backward(self,flat=0):
	# produces the backward operation given current parameters and VI parameters
	# return a tensor of shape self.input_shape (if flat==1) or self.input_layer.output_shape (if flat==0)
	back = tf.matmul(self.m,self.W)+tf.expand_dims(self.b,0) #(N K).(K D) -> (N D)
	if(flat or self.is_flat): return back
	else:     return tf.reshape(back,self.input_layer.output_shape)
    def sample(self,M,sigma_multiplier=1):
	# given the next layer generated sample (which will be in place of the current variable self.m)
	# generate the input os this layer by multiplying by the matrix self.W, adding the bias self.b and 
	# a random noise based on self.sigmas2
	# return a tensor of shape self.input_layer.m
	if(isinstance(self.input_layer,InputLayer)):sigma_multiplier = 0
	noise = sigma_multiplier*tf.random_normal(self.input_shape)*tf.expand_dims(tf.sqrt(self.sigmas2),0)
	output = tf.matmul(M,self.W)+tf.expand_dims(self.b,0)+noise
	if(self.is_flat): return output
	else:		  return tf.reshape(output,self.input_layer.output_shape)
    def likelihood(self,E_step=True):
        constant        = -tf.log(2*PI_CONST)*float32(self.D_in*0.5)-tf.reduce_sum(tf.log(self.sigmas2))*float32(0.5)
        if(E_step):
            reconstruction  = -tf.reduce_sum(tf.square(self.input-self.backward(flat=1))/tf.expand_dims(self.sigmas2,0))
            v2              = -tf.reduce_sum(self.v2/self.next_layer.sigmas2)
            extra           = -tf.reduce_sum(self.v2*tf.reduce_sum(tf.square(self.W)/tf.expand_dims(self.sigmas2,0),1))
            return constant+(reconstruction+v2+extra)*float32(0.5/self.input_shape[0])-self.sparsity_prior*tf.reduce_sum(tf.square(self.W))*0.5
        else:
            k1 = -tf.reduce_sum((self.Sm2+self.Sv2+tf.square(self.b)-2*self.b*self.Sm)/(2*self.sigmas2))
            k2 = tf.reduce_sum(self.SMm*self.W/tf.expand_dims(self.sigmas2,0))
	    k3 = -0.5*tf.reduce_sum(tf.matmul(self.W/tf.expand_dims(self.sigmas2,0),self.W,transpose_b=True)*self.SMM)
	    k4 = -0.5*tf.reduce_sum(self.Sv2*tf.reduce_sum(tf.square(self.W)/tf.expand_dims(self.sigmas2,0),1))
            return constant+k1+k2+k3+k4-self.sparsity_prior*tf.reduce_sum(tf.square(self.W))*0.5
    def KL(self,pretraining=False):
	return self.likelihood(True)+tf.reduce_sum(tf.log(self.v2+eps))*float32(0.5)/float32(self.bs)+float32(self.D_in/2.0)*tf.log(2*PI_CONST)
    ################### E STEP VARIABLES UPDATES #################################
    def update_v2(self):
	return []
	# method to return the update value of the self.v2 parameter
        scalor = tf.reduce_min(self.sigmas2)
	a4     = tf.einsum('kd,d->k',tf.square(self.W),scalor/self.sigmas2)+scalor/self.next_layer.sigmas2
	return tf.assign(self.v2_,scalor/a4)
    def update_m(self):
	return []
	# method to return the updated value of the self.m parameter
        rescalor = tf.reduce_min(self.sigmas2)
        Wp       = tf.reduce_sum(tf.expand_dims(self.W,1)*tf.expand_dims(self.p_,-1),2) # (k,n,d)
        inputWp  = tf.transpose(tf.reduce_sum(tf.expand_dims((self.input_-self.b)*rescalor/self.sigmas2,0)*Wp,2)) # ( n k )
        WpWp     = tf.matmul(tf.transpose(Wp,[1,0,2]),tf.transpose(Wp*rescalor/tf.expand_dims(self.sigmas2,0),[1,2,0]))*(1-tf.expand_dims(tf.eye(self.K),0)) # (n k k)
        next_sigmas = self.next_layer.sigmas2_
        back        = (self.next_layer.backward()+self.next_layer.b)*rescalor/self.next_layer.sigmas2
        b           = inputWp+back
        D           = tf.matrix_diag(tf.transpose(tf.reduce_sum(tf.expand_dims(tf.reduce_sum(tf.square(self.W)*tf.expand_dims(rescalor/self.sigmas2,0),2),1)*self.p_,2))) # (n k k)
        A           = WpWp+tf.expand_dims(tf.diag(rescalor/next_sigmas),0)+D
        with(tf.device('/device:CPU:0')): new_m = tf.matrix_solve(A,tf.expand_dims(b,-1))[:,:,0]#,l2_regularizer=0,fast=True)[:,:,0]
        return tf.assign(self.m_,tf.transpose(new_m))
    ################### M STEP VARIABLES UPDATES #################################
    def update_sigma(self,get_value=False):
	return []
        k1 = self.Sm2+self.Sv2+tf.square(self.b_)
        k2 = tf.reduce_sum(tf.expand_dims(self.SM2V2,-1)*tf.square(self.W),[0,1])
        k3 = tf.reduce_sum(tf.expand_dims(self.SMpMp,-1)*myexpand(self.W,[0,0])*myexpand(self.W,[2,2]),[0,1,2,3])
        k4 = -2*tf.reduce_sum(self.SMpm*self.W,[0,1])+2*self.b_*(tf.reduce_sum(tf.expand_dims(self.SMp,-1)*self.W,[0,1])-self.Sm)
        value_ = k1+k2+k3+k4
        if(get_value): return value_
        if(self.sigma_opt=='local'):     return tf.assign(self.sigmas2_,value_)
        elif(self.sigma_opt=='global'):  return tf.assign(self.sigmas2_,tf.fill([self.D_in],tf.reduce_sum(value_)/self.D_in))
        elif(self.sigma_opt=='channel'):
            v=tf.reduce_sum(tf.reshape(value_,self.input_shape[1:]),axis=[0,1],keepdims=True)/(self.input_shape[1]*self.input_shape[2])
            return tf.assign(self.sigmas2_,tf.reshape(tf.ones([self.input_shape[1],self.input_shape[2],1])*v,[self.D_in]))
    def update_Wk(self):
	return []
        k = self.k_
        new_w = (self.Sinput_m[:,self.k]-self.Sinput*self.b-tf.reduce_sum(tf.matmul(tf.transpose(self.SMpMp[k],[1,0,2]),self.W),0))/(self.SM2V2[k]+self.sparsity_prior)
        return tf.scatter_update(self.W_,[self.k_],[new_w])
    def update_b(self):
	return []
        if(self.update_b): return tf.assign(self.b,self.Sinput-tf.einsum('k,kd->d',self.Sm,self.W))
        return []






class AltConvLayer(Layer):
    def __init__(self,input_layer,K,Ic,Jc,R,sparsity_prior = 0,leakiness=None,sigma='local',alpha=0.5,init_W = tf.random_normal,update_b=True):
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
        if(leakiness == None):
            self.W_                = tf.Variable(init_W((self.K,self.R,self.Ic,self.Jc,self.C))/sqrt(self.K+self.Ic+self.Jc+self.C))
            self.W                 = self.W_
        else:
            self.W_                = tf.Variable(init_W((self.K,1,self.Ic,self.Jc,self.C))/sqrt(self.K+self.Ic+self.Jc+self.C))
            self.W                 = tf.concat([self.W_,leakiness*self.W_],axis=1)
	# WE DEFINE THE PARAMETERS
        self.pi             = tf.Variable(mysoftmax(tf.zeros((K,R)),axis=1))
	self.sigmas2_       = tf.Variable(tf.ones((self.Iin,self.Jin,self.C)))
	self.sigmas2        = tf.expand_dims(self.sigmas2_,0)
        self.sigmas2_patch_ = self.extract_patch(self.sigmas2,with_n=0)
	self.sigmas2_patch  = tf.expand_dims(self.sigmas2_patch_,0)
	self.b_             = tf.Variable(tf.zeros(self.input_shape[1:]))
	self.b              = tf.expand_dims(self.b_,0)
        self.b_patch_       = self.extract_patch(self.b,with_n=0)
        self.b_patch        = tf.expand_dims(self.b_patch_,0)
        self.apodization_    = ones((Ic,Jc,1))#hamming(self.Ic).reshape((-1,1,1))*hamming(self.Ic).reshape((1,-1,1))
        self.apodization_   /= self.apodization_.sum() #(a b 1)
        self.apodization    = self.apodization_.reshape((1,1,1,self.Ic,self.Ic,1)) # ( 1 1 1 a b 1)
        # VI VARIABLES
	self.m_      = tf.Variable(tf.zeros((K,self.I,self.J,self.bs)))
        self.m       = tf.transpose(self.m_,[3,1,2,0])                            # (N I J K)
	self.p_      = tf.Variable(tf.zeros((K,self.I,self.J,self.R,self.bs)))    # (K,I,J,R,N)
        self.p       = tf.transpose(self.p_,[4,1,2,0,3])                          # (N I J K R)
        self.v2_     = tf.Variable(tf.ones((self.K,self.I,self.J,self.bs)))       # (K I J N)
        self.v2      = tf.transpose(self.v2_,[3,1,2,0])
	self.drop_   = tf.Variable(tf.ones((K,2,self.I,self.J,self.bs))*tf.reshape(tf.one_hot(1,2),(1,2,1,1,1))) # (K 2 I J N)
        self.drop    = tf.transpose(self.drop_,[1,4,2,3,0]) #(2 N I J K)
        # STATISTICS
        with tf.device('/device:CPU:0'):
            self.Sm2   = tf.Variable(tf.zeros((self.Iin,self.Jin,self.C)))
            self.Sv2   = tf.Variable(tf.zeros((self.Iin,self.Jin,self.C)))
            self.SMpm  = tf.Variable(tf.zeros((self.K,self.I,self.J,self.R,self.Ic,self.Jc,self.C)))
            self.SM2V2 = tf.Variable(tf.zeros((self.K,self.I,self.J,self.R)))
            self.SMpp  = tf.Variable(tf.zeros((self.K,self.R,self.K,self.R,self.I,self.J)))
            self.SMp   = tf.Variable(tf.zeros((self.K,self.I,self.J,self.R)))
            self.Sm    = tf.Variable(tf.zeros((self.Iin,self.Jin,self.C)))
            self.Sp    = tf.Variable(tf.zeros((self.K,self.R)))
	#
        input_layer.next_layer = self
        self.k_        = tf.placeholder(tf.int32)
        self.W_indices = asarray(range(self.K)).reshape((-1,1))
        self.m_indices = asarray([0])
	self.p_indices = asarray(range(self.K)).reshape((-1,1))
        self.E         = tf.reshape(tf.eye(self.K),[self.K,1,self.K,1,1,1])
    def init_W(self,W):  return tf.assign(self.W_,W)
    def update_S(self):
        Sm2   = tf.reduce_sum(tf.square(self.input_layer.m),0)/self.bs
        Sv2   = tf.reduce_sum(self.input_layer.v2,0)/self.bs
        Sm    = tf.reduce_sum(self.input_layer.m,0)/self.bs
        Sp   = tf.reduce_sum(tf.clip_by_value(self.p_,0.02,0.98),[1,2,4])/self.bs
        #
        SMpm  = tf.reduce_sum(myexpand(tf.expand_dims(self.m_,-2)*self.p_,[-1,-1,-1])*tf.transpose(myexpand(self.extract_patch(self.input),[0,0]),[0,3,4,1,2,5,6,7]),4)/self.bs
        SM2V2 = tf.reduce_sum(tf.expand_dims(tf.square(self.m_)+self.v2_,-2)*self.p_,4)/self.bs
        mp    = tf.transpose(tf.expand_dims(self.m_,-2)*self.p_,[0,3,1,2,4]) # (K I J R N) -> (K R I J N)
        SMpp  = tf.reduce_sum(myexpand(mp,[2,2])*myexpand(mp,[0,0]),-1)/self.bs
        SMp   = tf.reduce_sum(tf.expand_dims(self.m_,-2)*self.p_,-1)/self.bs
        return tf.group(tf.assign(self.Sm2,self.alpha*Sm2+(1-self.alpha)*self.Sm2),tf.assign(self.Sv2,self.alpha*Sv2+(1-self.alpha)*self.Sv2),
                        tf.assign(self.Sm,self.alpha*Sm+(1-self.alpha)*self.Sm),tf.assign(self.Sp,self.alpha*Sp+(1-self.alpha)*self.Sp),
                        tf.assign(self.SMpm,self.alpha*SMpm+(1-self.alpha)*self.SMpm),tf.assign(self.SM2V2,self.alpha*SM2V2+(1-self.alpha)*self.SM2V2),
                        tf.assign(self.SMpp,self.alpha*SMpp+(1-self.alpha)*self.SMpp),tf.assign(self.SMp,self.alpha*SMp+(1-self.alpha)*self.SMp))
    def extract_patch(self,u,with_n=1,with_reshape=1):
	patches = tf.extract_image_patches(u,(1,self.Ic,self.Jc,1),(1,1,1,1),(1,1,1,1),"VALID")
	if(with_reshape):
	    if(with_n): return tf.reshape(patches,(self.bs,self.output_shape[1],self.output_shape[2],self.Ic,self.Jc,self.C))
	    else:       return tf.reshape(patches,(self.output_shape[1],self.output_shape[2],self.Ic,self.Jc,self.C))
	return patches
#                                           ---- BACKWARD OPERATOR ---- 
    def deconv(self,input=None,masked_m=0,masked_w=0,m=None,p=None,apodization=False):
	if(m is None):m=self.m_
	if(p is None):p=self.p_
	if(masked_m==1):    mask = p*tf.expand_dims(m*(1-self.mask),3)
	elif(masked_m==-1): mask = p*tf.expand_dims(m*self.mask,3)
        else:               mask = p*tf.expand_dims(m,3) # (K I J R N)
        proj = tf.reduce_sum(myexpand(tf.transpose(mask,[4,1,2,0,3]),[-1,-1,-1])*myexpand(self.W,[0,0,0]),[3,4])# (N I J a b c)
        if(apodization):     return tf.gradients(self.input_patch,self.input,proj*self.apodization)[0]
        else:                return tf.gradients(self.input_patch,self.input,proj*self.apodization)[0]
    def sample(self,M,K=None,sigma=1):
	#multinomial returns [K,n_samples] with integer value 0,...,R-1
	if(isinstance(self.input_layer,InputLayer)):sigma=0
	noise      = sigma*tf.random_normal(self.input_shape)*tf.sqrt(self.sigmas2)
        sigma_hot  = tf.one_hot(tf.reshape(tf.multinomial(tf.log(self.pi),self.bs*self.I*self.J),(self.K,self.I,self.J,self.bs)),self.R) # (K I J N R)
        return self.deconv(m=tf.transpose(M,[3,1,2,0]),p=tf.transpose(sigma_hot,[0,1,2,4,3]))+noise+self.b
    def evidence(self): return 0
    def likelihood(self,batch=0,pretraining=False):
        if(batch==0):
            deconv =  tf.reduce_sum(myexpand(tf.transpose(self.p_*tf.expand_dims(self.m_,3),[4,1,2,0,3]),[-1,-1,-1])*myexpand(self.W,[0,0,0]),[3,4])# (N I J a b c)
            patches= (self.extract_patch(self.input)-self.b_patch)
            k1  = -float32(self.D_in/2.0)*tf.log(2*PI_CONST)-tf.reduce_sum(tf.log(self.sigmas2_))/2+tf.reduce_sum(tf.log(self.pi)*tf.reduce_sum(self.p_,[1,2,4]))/float32(self.bs)
            k2  = -tf.reduce_sum((tf.square(patches*self.apodization-deconv)+self.extract_patch(self.input_layer.v2)*self.apodization**2)/self.sigmas2_patch)*0.5/self.bs
            k3  = tf.reduce_sum(tf.square(tf.reduce_sum(myexpand(tf.expand_dims(self.m,-1)*self.p,[-1,-1,-1])*myexpand(self.W,[0,0,0]),4))/tf.expand_dims(self.sigmas2_patch,3))*0.5/self.bs
            k4  = -tf.reduce_sum(tf.reduce_sum(tf.expand_dims(tf.square(self.m_)+self.v2_,-2)*self.p_,4)*tf.reduce_sum(myexpand(tf.square(self.W),[1,1])/myexpand(self.sigmas2_patch_,[0,3]),[4,5,6]))*0.5/self.bs
            return k1+k2+k3+k4-self.sparsity_prior*tf.reduce_sum(tf.square(self.W))/2#+extra_k
        else:
            k1  = -float32(self.D_in/2.0)*tf.log(2*PI_CONST)-tf.reduce_sum(tf.log(self.sigmas2_))/2+tf.reduce_sum(self.Sp*tf.log(self.pi))
            k20 = -tf.reduce_sum((self.extract_patch(tf.expand_dims(self.Sm2+self.Sv2+tf.square(self.b_),0),0)*self.apodization[0]**2)/self.sigmas2_patch_)*0.5
            k21 = tf.reduce_sum((self.extract_patch(tf.expand_dims(self.Sm*self.b_,0),0)*self.apodization[0])/self.sigmas2_patch_)
            k22 = tf.reduce_sum(self.apodization*self.SMpm*myexpand(self.W,[1,1])/tf.expand_dims(self.sigmas2_patch,3))
            k23 = -tf.reduce_sum(tf.reduce_sum(myexpand(self.SMp,[-1,-1,-1])*myexpand(self.W,[1,1]),3)*self.b_patch*self.apodization/self.sigmas2_patch)
            ww  = tf.reduce_sum(myexpand(self.W,[0,0,-4,-4])*myexpand(self.W,[2,2,-4,-4])/myexpand(self.sigmas2_patch_,[0,0,0,0]),[6,7,8]) # (K R K R I J)
            k3  = -tf.reduce_sum(self.SMpp*(1-self.E)*ww)*0.5
            k4  = -tf.reduce_sum(tf.reduce_sum(myexpand(self.SM2V2,[-1,-1,-1])*myexpand(tf.square(self.W),[1,1]),3)/self.sigmas2_patch)*0.5
            return k1+k20+k21+k22+k23+k3+k4-self.sparsity_prior*tf.reduce_sum(tf.square(self.W))/2
    def KL(self,pretraining=False):
        return self.likelihood(0,pretraining)+(-tf.reduce_sum(self.p_*tf.log(self.p_+eps))+float32(0.5)*tf.reduce_sum(tf.log(self.v2_)))/float32(self.bs)+float32(self.D_in/2.0)*tf.log(2*PI_CONST)
    def update_v2(self,pretraining=False):# DONE
        if(isinstance(self.next_layer,ConvLayer) or isinstance(self.next_layer,PoolLayer)): next_sigmas = self.next_layer.sigmas2 # (N I J K)
        else:            next_sigmas = tf.expand_dims(tf.reshape(self.next_layer.sigmas2_,self.output_shape[1:]),0)
        rescalor = tf.reduce_min(self.sigmas2)
        a4       = tf.reduce_sum(self.p_*tf.expand_dims(tf.reduce_sum(myexpand(tf.square(self.W),[1,1])*rescalor/tf.expand_dims(self.sigmas2_patch,3),[4,5,6]),-1),3)
        return tf.assign(self.v2_,rescalor/(tf.transpose(rescalor*sum(self.next_layer.apodization**2)/next_sigmas,[3,1,2,0])+a4))
    def update_m(self,mp_opt=0,pretraining=False):
        Wp             = tf.reduce_sum(myexpand(self.W,[0,0,0])*myexpand(self.p,[-1,-1,-1]),4)                    #(N I J K A B C)
	if(isinstance(self.next_layer,ConvLayer) or isinstance(self.next_layer,PoolLayer) or isinstance(self.next_layer,AltConvLayer)): 
            back = (self.next_layer.deconv()+self.next_layer.b)
            A2   = self.next_layer.sigmas2_
	else:       
            back = self.next_layer.vector2tensor(self.next_layer.backward()+self.next_layer.b)[:,self.i_::self.ratio,self.j_::self.ratio]
            A2   = tf.reshape(self.next_layer.sigmas2_,self.output_shape[1:])[self.i_::self.ratio,self.j_::self.ratio]
        if(pretraining):
            next_back   = tf.zeros_like(back)
            next_sigmas = tf.ones_like(A2)
        else:
            next_back   = back
            next_sigmas = A2
        rescalor = tf.reduce_min(self.sigmas2_)
        bias     = next_back*rescalor/tf.expand_dims(next_sigmas,0)+tf.reduce_sum(tf.expand_dims(self.apodization*self.extract_patch(self.input-self.b)*rescalor/self.sigmas2_patch,3)*Wp,[4,5,6])
        Wpr1     = tf.reshape(Wp*rescalor/myexpand(self.sigmas2_patch_,[0,-4]),[self.bs,self.I,self.J,self.K,-1])
        Wpr2     = tf.reshape(Wp,[self.bs,self.I,self.J,self.K,-1])
        A1       = tf.matmul(Wpr1,Wpr2,transpose_b=True)*(1-tf.reshape(tf.eye(self.K),[1,1,1,self.K,self.K]))
        A2       = tf.reduce_sum(self.p*tf.expand_dims(tf.reduce_sum(myexpand(tf.square(self.W),[0,0])*myexpand(rescalor/self.sigmas2_patch_,[2,2]),[4,5,6]),0),4)
        A        = A1+tf.matrix_diag(A2)+tf.expand_dims(tf.matrix_diag(rescalor*sum(self.next_layer.apodization**2)/next_sigmas),0)
        with(tf.device('/device:CPU:0')):
            new_m = tf.transpose(tf.matrix_solve(A,tf.expand_dims(bias,-1))[:,:,:,:,0],[3,1,2,0]) # (N K I J)
	return tf.assign(self.m_,new_m)
    def update_p(self):
        reconstruction = tf.reduce_sum(myexpand(tf.expand_dims(self.m*myexpand(1-tf.one_hot(self.k_,self.K),[0,0,0]),-1)*self.p,[-1,-1,-1])*myexpand(self.W,[0,0,0]),[3,4])
        forward        = tf.reduce_sum(tf.expand_dims((self.extract_patch(self.input-self.b)-reconstruction)*self.apodization/self.sigmas2_patch,3)*myexpand(self.W[self.k_],[0,0,0]),[4,5,6])*tf.expand_dims(self.m[:,:,:,self.k_],-1) #(N I' J' R)
        m2v2     = tf.expand_dims(tf.square(self.m_[self.k_])+self.v2_[self.k_],-2)*tf.expand_dims(tf.reduce_sum(myexpand(tf.square(self.W[self.k_]),[0,0])*0.5/tf.expand_dims(self.sigmas2_patch_,2),[3,4,5]),-1) #(I J R N)
	value    = tf.transpose(forward,[1,2,3,0])-m2v2+myexpand(tf.log(self.pi[self.k_]),[0,0,-1])# (I' J' R N)
	return tf.scatter_update(self.p_,[self.k_],[tf.nn.softmax(value,axis=2)])
    def update_Wk(self):
        r1 = tf.reduce_sum(myexpand(self.SMpp[self.k_]*(1-myexpand(tf.one_hot(self.k_,self.K),[0,-1,-1,-1])),[-1,-1,-1])*myexpand(self.W,[0,-4,-4]),[1,2]) # (R I J a b c)
        r2 = self.SMpm[self.k_] # (I J R a b c)
        r3 = myexpand(self.b_patch_,[2])*myexpand(self.SMp[self.k_],[-1,-1,-1]) # (I J R a b c)
        numerator      = tf.reduce_sum((-r1+tf.transpose(r2-r3,[2,0,1,3,4,5]))*self.apodization/self.sigmas2_patch,[1,2]) # (R a b c) 
        denominator    = tf.reduce_sum(myexpand(tf.expand_dims(tf.square(self.m_[self.k_])+self.v2_[self.k_],-2)*self.p_[self.k_],[-1,-1,-1])/myexpand(self.sigmas2_patch_,[-4,-4]),[0,1,3])
        return tf.scatter_update(self.W_,[self.k_],[numerator/(denominator+self.sparsity_prior)])
        rescalor = tf.reduce_min(self.sigmas2_)
        # FOR THE BIAS
        KR      = tf.transpose(tf.reshape(tf.stack([rescalor*self.filter_corr(self.W[:,r1],self.W[:,r2])*(1-self.E) for r1,r2 in itertools.product(range(self.R),range(self.R))],-1),[self.K,self.I,self.J,self.K,self.Ic*2-1,self.Jc*2-1,self.R,self.R]),[0,1,2,6,3,4,5,7])
        BBB     = tf.reshape(tf.diag(tf.one_hot(self.k_,self.K)),[self.K,1,1,1,self.K,1,1,1])*tf.reshape(tf.diag(tf.one_hot(self.r_,self.R)),[1,1,1,self.R,1,1,1,self.R])#myexpand(tf.one_hot(self.r_,self.R),[0,0,0,-1,-1,-1,-1])
        k1      = -0.5*tf.gradients(tf.reduce_sum(KR*self.SMpMp*(1-BBB)),self.W)[0][self.k_,self.r_]# (a b C)
        #
#        BB      = self.SMpMp[self.k_][:,:,self.r_,self.k_,self.Ic-1,self.Jc-1,:]
#        bsamek  = -tf.reduce_sum(tf.reduce_sum(myexpand(self.W[self.k_],[0,0])*myexpand(BB*(1-myexpand(tf.one_hot(self.r_,self.R),[0,0])),[-1,-1,-1]),2)/self.sigmas2_patch_,[0,1])# (Ic Jc C)
        #
        binput  = tf.reduce_sum(self.SmMp[self.k_,:,:,self.r_]*rescalor/self.sigmas2_patch_-myexpand(self.SMp[self.k_,:,:,self.r_],[-1,-1,-1])*self.b_patch_*rescalor/self.sigmas2_patch_,[0,1]) # (self.C self.Ic self.Jc)
        B       = tf.reshape(tf.transpose(binput+k1,[2,0,1]),[self.C,-1])
        # FOR THE MATRIX
        DM2V2   = tf.transpose(tf.reduce_sum(myexpand(self.SM2V2[self.k_,:,:,self.r_],[-1,-1,-1])*rescalor/self.sigmas2_patch_,[0,1]),[2,0,1]) # (self.C self.Ic self.Jc)
        BBB     = tf.reduce_sum(self.SMpMp*myexpand(tf.one_hot(self.k_,self.K),[1,1,1,1,1,1,1]),0)
        rS5     = tf.transpose(tf.reshape(tf.extract_image_patches(tf.transpose(BBB[:,:,self.r_,self.k_,:,:,self.r_],[0,2,3,1]),(1,self.Ic,self.Jc,1),(1,1,1,1),(1,1,1,1),"VALID"),(self.I,self.Ic,self.Jc,self.Ic,self.Jc,self.J)),[0,5,1,2,3,4])
        a1      = tf.einsum('ijabcd,ijabk->kabcd',rS5[:,:,:,:,::-1,::-1],0.5*rescalor/self.sigmas2_patch_)
        a2      = tf.einsum('ijabcd,ijcdk->kabcd',rS5[:,:,::-1,::-1],0.5*rescalor/self.sigmas2_patch_)
        A       = tf.reshape(a1+a2,[self.C,self.Ic*self.Jc,self.Ic*self.Jc])*(1-tf.expand_dims(tf.eye(self.Ic*self.Jc),0))+tf.matrix_diag(tf.reshape(DM2V2,[self.C,-1])+rescalor*self.sparsity_prior)
        with(tf.device('/device:CPU:0')):
            W      = tf.reshape(tf.matrix_solve(A,tf.expand_dims(B,-1)),[self.C,self.Ic,self.Jc])
        return tf.scatter_nd_update(self.W_,[[self.k_,self.r_]],[tf.transpose(W,[1,2,0])])
    def update_pi(self): return tf.assign(self.pi,self.Sp/tf.reduce_sum(self.Sp,axis=1,keepdims=True))
    def update_BV(self):
        if(self.update_b == False): return []
        value = self.Sm-tf.gradients(self.sigmas2_patch_,self.sigmas2_,tf.reduce_sum(myexpand(self.SMp,[-1,-1,-1])*myexpand(self.W,[1,1]),[0,3]))[0]
        if(self.update_b=='local'): return tf.assign(self.b_,value/sum(self.apodization**2))
        rescalor = tf.reduce_min(self.sigmas2_)
        value1 = value*rescalor/self.sigmas2_
        value2 = rescalor/self.sigmas2_
        if(self.update_b=='channel'): return tf.assign(self.b_,tf.ones_like(self.b_)*tf.reduce_sum(value1,[0,1],keepdims=True)/tf.reduce_sum(value2,[0,1],keepdims=True))
        return tf.assign(self.b_,tf.ones_like(self.b_)*tf.reduce_sum(value1,keepdims=True)/tf.reduce_sum(value2,keepdims=True))
    def update_sigma(self,get_value=False):
        k20 = sum(self.apodization**2)*(self.Sm2+self.Sv2+tf.square(self.b_))
        k21 = -2*self.Sm*self.b_
        k22 = -2*tf.gradients(self.sigmas2_patch_,self.sigmas2_,tf.reduce_sum(self.SMpm*myexpand(self.W,[1,1]),[0,3]))[0]
        k23 = 2*tf.gradients(self.sigmas2_patch_,self.sigmas2_,tf.reduce_sum(myexpand(self.SMp,[-1,-1,-1])*myexpand(self.W,[1,1]),[0,3])*self.b_patch_)[0]
        ww  = myexpand(self.W,[0,0])*myexpand(self.W,[2,2]) # (K R K R a b c)
        k3  = tf.gradients(self.sigmas2_patch_,self.sigmas2_,tf.reduce_sum(myexpand(self.SMpp*(1-self.E),[-1,-1,-1])*myexpand(ww,[-4,-4]),[0,1,2,3]))[0]
        k4  = tf.gradients(self.sigmas2_patch_,self.sigmas2_,tf.reduce_sum(myexpand(self.SM2V2,[-1,-1,-1])*myexpand(tf.square(self.W),[1,1]),[0,3]))[0]
        value = k20+k21+k22+k23+k3+k4
        if(get_value): return value
        if(self.sigma_opt=='local'):     return tf.assign(self.sigmas2_,value)
        elif(self.sigma_opt=='channel'): return tf.assign(self.sigmas2_,tf.reduce_sum(value,[0,1],keepdims=True)*tf.ones([self.Iin,self.Jin,1])/(self.Iin*self.Jin))
	elif(self.sigma_opt=='global'):  return tf.assign(self.sigmas2_,tf.fill([self.Iin,self.Jin,self.C],tf.reduce_sum(value)/(self.Iin*self.Jin*self.C)))







class ConvLayer(Layer):
    def __init__(self,input_layer,K,Ic,Jc,R,sparsity_prior = 0,leakiness=0,sigma='local',alpha=0.5,init_W = tf.orthogonal_initializer(),update_b=True):
        self.alpha             = tf.Variable(alpha)
        self.update_b          = update_b
        self.leakiness         = leakiness
	self.sigma_opt         = sigma
        self.sparsity_prior    = sparsity_prior
        self.input_layer       = input_layer
        input_layer.next_layer = self
        self.bs,self.Iin,self.Jin,self.C  = input_layer.output_shape 
        self.Ic,self.Jc,self.K,self.R     = Ic,Jc,K,R
        self.input_shape       = input_layer.output_shape
        self.output_shape      = (self.bs,self.input_shape[-3]-self.Ic+1,self.input_shape[-2]-self.Jc+1,K)
        self.D_in              = prod(self.input_shape[1:])
        self.I,self.J          = self.output_shape[1],self.output_shape[2]
        self.input             = input_layer.m
        self.input_patch       = self.extract_patch(self.input,with_n=1)
        # WE DEFINE THE PARAMETERS
        self.W_                = tf.Variable(init_W((self.K,self.Ic,self.Jc,self.C))/sqrt(self.Ic+self.Jc+self.C))
        self.W                 = tf.stack([self.W_,leakiness*self.W_],axis=1)
        self.pi             = tf.Variable(tf.ones(K)*0.5)
	self.sigmas2_       = tf.Variable(tf.ones((self.Iin,self.Jin,self.C)))
	self.sigmas2        = tf.expand_dims(self.sigmas2_,0)
        self.sigmas2_patch_ = self.extract_patch(self.sigmas2,with_n=0)
	self.sigmas2_patch  = tf.expand_dims(self.sigmas2_patch_,0)
	self.b_             = tf.Variable(tf.zeros(self.input_shape[1:]))
	self.b              = tf.expand_dims(self.b_,0)
        self.b_patch_       = self.extract_patch(self.b,with_n=0)
        self.b_patch        = tf.expand_dims(self.b_patch_,0)
        # VI VARIABLES
	self.m_      = tf.Variable(tf.zeros((K,self.I,self.J,self.bs)))           # (K,I,J,N)
        self.m       = tf.transpose(self.m_,[3,1,2,0])                            # (N I J K)
	self.p_      = tf.Variable(tf.zeros((K,self.I,self.J,self.bs)))           # (K,I,J,N)
        self.p       = tf.transpose(self.p_,[3,1,2,0])                            # (N I J K)
        self.v2_     = tf.Variable(tf.ones((self.K,self.I,self.J,self.bs)))       # (K I J N)
        self.v2      = tf.transpose(self.v2_,[3,1,2,0])                           # (N I J K)
        # STATISTICS
        self.Sm2   = tf.Variable(tf.zeros((self.Iin,self.Jin,self.C)))
        self.Sv2   = tf.Variable(tf.zeros((self.Iin,self.Jin,self.C)))
        self.SmMp  = tf.Variable(tf.zeros((self.K,self.I,self.J,self.Ic,self.Jc,self.C)))
        self.SM2V2 = tf.Variable(tf.zeros((self.K,self.I,self.J)))
        self.SMpMp = tf.Variable(tf.zeros((self.K,self.I,self.J,self.K,self.Ic*2-1,self.Jc*2-1)))
        self.SMp   = tf.Variable(tf.zeros((self.K,self.I,self.J)))
        self.Sm    = tf.Variable(tf.zeros((self.Iin,self.Jin,self.C)))
        self.Sp    = tf.Variable(tf.zeros((self.K)))
	#
        input_layer.next_layer = self
        self.k_      = tf.placeholder(tf.int32)
        self.i_      = tf.placeholder(tf.int32)
        self.j_      = tf.placeholder(tf.int32)
	self.ratio   = Ic
        self.Ni      = tf.cast((self.I-self.i_-1)/self.ratio+1,tf.int32) # NUMBER OF TERMS
        self.Nj      = tf.cast((self.J-self.j_-1)/self.ratio+1,tf.int32) # NUMBER OF TERMS
        self.xi,self.yi = tf.meshgrid(tf.range(self.j_,self.J,self.ratio),tf.range(self.i_,self.I,self.ratio)) # THE SECOND IS CONSTANT (meshgrid)
        self.indices_   = tf.concat([tf.fill([self.Ni*self.Nj,1],self.k_),tf.reshape(self.yi,(self.Ni*self.Nj,1)),tf.reshape(self.xi,(self.Nj*self.Ni,1))],axis=1) # (V 3) indices where the 1 pops
        self.m_indices_ = tf.concat([tf.concat([tf.fill([self.Ni*self.Nj,1],KK),tf.reshape(self.yi,(self.Ni*self.Nj,1)),tf.reshape(self.xi,(self.Nj*self.Ni,1))],axis=1) for KK in range(self.K)],axis=0)
        self.W_indices = asarray([a for a in itertools.product(range(self.K),range(self.Ic),range(self.Jc))])
        self.m_indices = asarray([a for a in itertools.product(range(self.ratio),range(self.ratio))])#ratio REMOVE K KK K K K K
	self.p_indices = asarray([a for a in itertools.product(range(self.K),range(self.ratio),range(self.ratio))])
	mask           = tf.reshape(tf.one_hot(self.k_,self.K),(self.K,1,1))*tf.reshape(tf.tile(tf.one_hot(self.i_,self.ratio),[(self.I/self.ratio+1)]),(1,(self.I/self.ratio+1)*self.ratio,1))*tf.reshape(tf.tile(tf.one_hot(self.j_,self.ratio),[self.J/self.ratio+1]),(1,1,(self.J/self.ratio+1)*self.ratio))
        self.mask      = tf.expand_dims(mask[:,:self.I,:self.J],-1) # (I J K)
        m_mask         = tf.reshape(tf.tile(tf.one_hot(self.i_,self.ratio),[(self.I/self.ratio+1)]),((self.I/self.ratio+1)*self.ratio,1))*tf.reshape(tf.tile(tf.one_hot(self.j_,self.ratio),[self.J/self.ratio+1]),(1,(self.J/self.ratio+1)*self.ratio))
        self.m_mask    = m_mask[:self.I,:self.J]
        self.E= tf.einsum('ij,kc,a,b->kijcab',tf.ones((self.I,self.J)),tf.eye(self.K),tf.one_hot(self.Ic-1,2*self.Ic-1),tf.one_hot(self.Ic-1,2*self.Ic-1))
    def init_W(self,W):  return tf.assign(self.W_,W)
    def filter_corr(self,A,B):
        #takes as input filter banks A and B of same shape (K a b c)
        k1 = tf.reshape(myexpand(A,[1,1])/self.sigmas2_patch,[self.K*self.I*self.J,self.Ic,self.Jc,self.C]) # (K I J a b c)->(KIJ a b c)
        k2 = tf.pad(k1,[[0,0],[self.Ic-1,self.Ic-1],[self.Jc-1,self.Jc-1],[0,0]]) # (KIJ 3a-2 3b-2 c)
        k3 = tf.reshape(tf.extract_image_patches(k2,(1,self.Ic,self.Jc,1),(1,1,1,1),(1,1,1,1),"VALID"),(self.K*self.I*self.J,2*self.Ic-1,2*self.Jc-1,self.Ic,self.Jc,self.C)) # (KIJ 2a-1 2b-1 a b c)
        k4 = tf.reduce_sum(tf.expand_dims(k3,1)*myexpand(B,[0,-4,-4]),[4,5,6]) # (KIJ K 2a-1 2b-1)
        return tf.reshape(k4,[self.K,self.I,self.J,self.K,2*self.Ic-1,2*self.Jc-1])
    def update_S(self):
        Sm2   = tf.reduce_sum(tf.square(self.input_layer.m),0)/self.bs
        Sv2   = tf.reduce_sum(self.input_layer.v2,0)/self.bs
        SmMp  = tf.transpose(tf.reduce_sum(tf.expand_dims(self.input_patch,3)*myexpand(self.m*self.p,[-1,-1,-1]),0),[2,0,1,3,4,5])/self.bs
        SM2V2 = tf.reduce_sum((tf.square(self.m_)+self.v2_)*self.p_,3)/self.bs
        mp_patches = tf.transpose(tf.reshape(tf.extract_image_patches(tf.pad(self.m*self.p,[[0,0],[self.Ic-1,self.Ic-1],[self.Jc-1,self.Jc-1],[0,0]]),(1,2*self.Ic-1,2*self.Jc-1,1),(1,1,1,1),(1,1,1,1),"VALID"),(self.bs,self.I,self.J,2*self.Ic-1,2*self.Jc-1,self.K)),[0,1,2,5,3,4])# (N I J I' J' K)-> (N I J K I' J')
        SMpMp      =tf.reduce_sum(tf.expand_dims(mp_patches,1)*myexpand(tf.transpose(self.m*self.p,[0,3,1,2]),[-1,-1,-1]),0)/self.bs#(K I J K I'J')
        SMp = tf.reduce_sum(self.m_*self.p_,-1)/self.bs
        Sm  = tf.reduce_sum(self.input_layer.m,0)/self.bs
        Sp  = tf.reduce_sum(self.p_,[1,2,3])/self.bs
        return tf.group(tf.assign(self.Sm2,self.alpha*Sm2+(1-self.alpha)*self.Sm2),tf.assign(self.Sv2,self.alpha*Sv2+(1-self.alpha)*self.Sv2),
                        tf.assign(self.SmMp,self.alpha*SmMp+(1-self.alpha)*self.SmMp),tf.assign(self.SM2V2,self.alpha*SM2V2+(1-self.alpha)*self.SM2V2),
                        tf.assign(self.SMpMp,self.alpha*SMpMp+(1-self.alpha)*self.SMpMp),tf.assign(self.SMp,self.alpha*SMp+(1-self.alpha)*self.SMp),
                        tf.assign(self.Sm,self.alpha*Sm+(1-self.alpha)*self.Sm),tf.assign(self.Sp,self.alpha*Sp+(1-self.alpha)*self.Sp))
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
	if(masked_m==1):    mask = p*m*(1-self.mask)
	elif(masked_m==-1): mask = p*m*self.mask
        else:               mask = p*m               # (K I J N)
        proj = tf.reduce_sum(myexpand(tf.transpose(mask,[3,1,2,0]),[-1,-1,-1])*myexpand(self.W_,[0,0,0]),3)# (N I J a b c)
        return tf.gradients(self.input_patch,self.input,proj)[0]
    def sample(self,M,K=None,sigma=1):
	#multinomial returns [K,n_samples] with integer value 0,...,R-1
	if(isinstance(self.input_layer,InputLayer)):sigma=0
	noise      = sigma*tf.random_normal(self.input_shape)*tf.sqrt(self.sigmas2)
        sigma_hot  = tf.reshape(tf.multinomial(tf.log(tf.stack([1-self.pi,self.pi],-1)),self.bs*self.I*self.J),(self.K,self.I,self.J,self.bs)) # (K I J N)
        return self.deconv(m=tf.transpose(M,[3,1,2,0]),p=tf.cast(sigma_hot,tf.float32))+noise+self.b
    def evidence(self): return 0
    def likelihood(self,batch=0,pretraining=False):
        if(batch==0):
            if(pretraining==False): extra_k = 0
            else:                   extra_k = -0.5*tf.reduce_sum(tf.square(self.m_)+self.v2_)/self.bs
            k1  = -float32(self.D_in/2.0)*tf.log(2*PI_CONST)-tf.reduce_sum(tf.log(self.sigmas2_))/2+tf.reduce_sum(tf.log(self.pi)*tf.reduce_sum(self.p_,[1,2,3])+tf.log(1-self.pi)*tf.reduce_sum(1-self.p_,[1,2,3]))/float32(self.bs)
            k2  = -tf.reduce_sum((tf.square(self.input-self.deconv()-self.b)+self.input_layer.v2)/(2*self.sigmas2*self.bs))
            k   = tf.expand_dims(tf.reduce_sum(myexpand(tf.square(self.W_),[1,1])/self.sigmas2_patch,[3,4,5]),-1) # (K I J 1)
            k3  = tf.reduce_sum(tf.square(self.m_*self.p_)*k)*0.5/self.bs
            k4  = -tf.reduce_sum((tf.square(self.m_)+self.v2_)*self.p_*k)*0.5/self.bs
            return k1+k2+k3+k4-self.sparsity_prior*tf.reduce_sum(tf.square(self.W))/2+extra_k
        else:
            k1 = -float32(self.D_in/2.0)*tf.log(2*PI_CONST)-tf.reduce_sum(tf.log(self.sigmas2_))/2+tf.reduce_sum(self.Sp*tf.log(self.pi)+(self.I*self.J-self.Sp)*tf.log(1-self.pi))
            k2 = -tf.reduce_sum((self.Sm2+self.Sv2+tf.square(self.b_)-2*self.Sm*self.b_)/self.sigmas2_)*0.5
            k3 = tf.reduce_sum(self.SmMp*myexpand(self.W_,[1,1])/self.sigmas2_patch)
            k4 = -tf.reduce_sum(myexpand(self.SMp,[-1,-1,-1])*myexpand(self.W_,[1,1])*self.b_patch/self.sigmas2_patch)
            k5 = -tf.reduce_sum(myexpand(self.SM2V2,[-1,-1,-1])*myexpand(tf.square(self.W_),[1,1])/self.sigmas2_patch)*0.5
            k6 = -tf.reduce_sum(self.SMpMp*self.filter_corr(self.W_,self.W_)*(1-self.E))/2
            return k1+k2+k3+k4+k5+k6-self.sparsity_prior*tf.reduce_sum(tf.square(self.W))/2
    def KL(self,pretraining=False):
        return self.likelihood(0,pretraining)+(-tf.reduce_sum(self.p_*tf.log(self.p_+eps)+(1-self.p_)*tf.log(1-self.p_+eps))+float32(0.5)*tf.reduce_sum(tf.log(self.v2_)))/float32(self.bs)+float32(self.D_in/2.0)*tf.log(2*PI_CONST)
    def update_v2(self,pretraining=False):# DONE
        if(isinstance(self.next_layer,ConvLayer) or isinstance(self.next_layer,PoolLayer)): v_value = self.next_layer.sigmas2 # (N I J K)
        else:            v_value = tf.expand_dims(tf.reshape(self.next_layer.sigmas2_,self.output_shape[1:]),0)
        if(pretraining==False):next_sigmas = v_value
        else:                  next_sigmas = tf.ones_like(v_value)
        rescalor = tf.reduce_min(self.sigmas2)
        a4       = self.p_*tf.expand_dims(tf.reduce_sum(myexpand(tf.square(self.W_),[1,1])*rescalor/self.sigmas2_patch,[3,4,5]),-1)
        return tf.assign(self.v2_,rescalor/(tf.transpose(rescalor/next_sigmas,[3,1,2,0])+a4))
    def update_m(self,mp_opt=0,pretraining=False):
	if(isinstance(self.next_layer,ConvLayer) or isinstance(self.next_layer,PoolLayer)): 
            back = (self.next_layer.deconv()[:,self.i_::self.ratio,self.j_::self.ratio]+self.next_layer.b[:,self.i_::self.ratio,self.j_::self.ratio])
            A22   = self.next_layer.sigmas2_[self.i_::self.ratio,self.j_::self.ratio]
	else:       
            back = self.next_layer.vector2tensor(self.next_layer.backward()+self.next_layer.b)[:,self.i_::self.ratio,self.j_::self.ratio]
            A22   = tf.reshape(self.next_layer.sigmas2_,self.output_shape[1:])[self.i_::self.ratio,self.j_::self.ratio]
        if(pretraining):
            next_back   = tf.zeros_like(back)
            next_sigmas = tf.ones_like(A22)
        else:
            next_back   = back
            next_sigmas = A22
        rescalor = tf.reduce_min(self.sigmas2_)#patch_[self.i_::self.ratio,self.j_::self.ratio])
        ###
#        forward  = tf.reduce_sum(self.extract_patch((self.input-self.deconv(masked_m=True)-self.b)*rescalor/self.sigmas2)[:,self.i_::self.ratio,self.j_::self.ratio]*myexpand(self.W_[self.k_],[0,0,0]),[3,4,5])*self.p[:,self.i_::self.ratio,self.j_::self.ratio,self.k_]
#        backward = back[:,:,:,self.k_]*rescalor/tf.expand_dims(next_sigmas[:,:,self.k_],0)
#        denominator = rescalor/tf.expand_dims(next_sigmas[:,:,self.k_],-1)+tf.reduce_sum(myexpand(tf.square(self.W_[self.k_]),[0,0,0])*myexpand(self.p_[self.k_,self.i_::self.ratio,self.j_::self.ratio],[-1,-1,-1])*rescalor/tf.expand_dims(self.sigmas2_patch_[self.i_::self.ratio,self.j_::self.ratio],2),[3,4,5]) # ( I' J' N)
#        return tf.scatter_nd_update(self.m_,self.indices_,tf.transpose(tf.reshape((forward+backward)/tf.transpose(denominator,[2,0,1]),(self.bs,-1))))
        ###
        m_masked       = self.m*myexpand(1-self.m_mask,[0,-1])*self.p # (N I J K)
        reconstruction = tf.gradients(self.input_patch,self.input,tf.reduce_sum(myexpand(m_masked,[-1,-1,-1])*myexpand(self.W_,[0,0,0]),3))[0]
        Wp             = myexpand(self.W_,[0,0,0])*myexpand(self.p[:,self.i_::self.ratio,self.j_::self.ratio],[-1,-1,-1])                    #(N I J K A B C)
        rescalor = tf.reduce_min(self.sigmas2_)#patch_[self.i_::self.ratio,self.j_::self.ratio])
        bias     = next_back*rescalor/tf.expand_dims(next_sigmas,0)+tf.reduce_sum(tf.expand_dims(self.extract_patch((self.input-reconstruction-self.b)*rescalor/self.sigmas2)[:,self.i_::self.ratio,self.j_::self.ratio],3)*Wp,[4,5,6])
        #
        Wprs     = Wp*rescalor/myexpand(self.sigmas2_patch_[self.i_::self.ratio,self.j_::self.ratio],[0,-4])#[N I' J' K a b c)
        A1       = tf.reduce_sum(myexpand(Wprs,[4])*myexpand(Wp,[3]),[5,6,7])*myexpand(1-tf.eye(self.K),[0,0,0]) # ( N I J K K)
        A2       = tf.reduce_sum(myexpand(tf.square(self.W_),[0,0,0])*myexpand(self.p[:,self.i_::self.ratio,self.j_::self.ratio],[-1,-1,-1])*rescalor/myexpand(self.sigmas2_patch_[self.i_::self.ratio,self.j_::self.ratio],[0,-4,]),[4,5,6]) #(N I' J' K)
        A        = A1+tf.matrix_diag(A2)+tf.expand_dims(tf.matrix_diag(rescalor/next_sigmas),0)
        with(tf.device('/device:CPU:0')):
            new_m = tf.transpose(tf.matrix_solve(A,tf.expand_dims(bias,-1))[:,:,:,:,0],[0,3,1,2]) # (N K I J)
	return tf.scatter_nd_update(self.m_,self.m_indices_,tf.transpose(tf.reshape(new_m,(self.bs,-1))))
    def update_p(self):
#        return []
        forward  = tf.reduce_sum(self.extract_patch((self.input-self.deconv(masked_m=True)-self.b)/self.sigmas2)[:,self.i_::self.ratio,self.j_::self.ratio]*myexpand(self.W_[self.k_],[0,0,0]),[3,4,5])*self.m[:,self.i_::self.ratio,self.j_::self.ratio,self.k_] #(N I' J')
        m2v2     = (tf.square(self.m[:,self.i_::self.ratio,self.j_::self.ratio,self.k_])+self.v2[:,self.i_::self.ratio,self.j_::self.ratio,self.k_])*tf.reduce_sum(myexpand(tf.square(self.W_[self.k_]),[0,0,0])/self.sigmas2_patch[:,self.i_::self.ratio,self.j_::self.ratio],[3,4,5]) #(N I' J')
        update_value = tf.nn.sigmoid(forward-0.5*m2v2+tf.log(self.pi[self.k_])-tf.log(1-self.pi[self.k_])) # (N I' J')
	return tf.scatter_nd_update(self.p_,self.indices_,tf.transpose(tf.reshape(update_value,(self.bs,-1))))
    def update_Wk(self):
        rescalor = tf.reduce_min(self.sigmas2_)
        k1 = tf.reduce_sum((self.SmMp[self.k_,:,:,self.i_,self.j_]-tf.expand_dims(self.SMp[self.k_],-1)*self.b_[self.i_:self.i_+self.I,self.j_:self.j_+self.J])*rescalor/self.sigmas2_[self.i_:self.i_+self.I,self.j_:self.j_+self.J],[0,1])
        k2 = 0.5*tf.reduce_sum(tf.expand_dims((self.SMpMp*(1-self.E))[self.k_,:,:,:,self.i_:self.i_+self.Ic,self.j_:self.j_+self.Jc],-1)*myexpand(self.W_[:,::-1,::-1],[0,0])*rescalor/myexpand(self.sigmas2_patch_,[2]),[0,1,2,3,4])
        WW = (self.SMpMp*(1-self.E))[:,:,:,self.k_,self.Ic-self.i_-1:2*self.Ic-self.i_-1,self.Jc-self.j_-1:2*self.Jc-self.j_-1]
        k3 = 0.5*tf.reduce_sum(tf.expand_dims(WW,-1)*myexpand(self.W_,[1,1])*rescalor/myexpand(self.sigmas2_patch_,[0]),[0,1,2,3,4]) # second case
        c1 = tf.reduce_sum(tf.expand_dims(self.SM2V2[self.k_],-1)*rescalor/self.sigmas2_[self.i_:self.i_+self.I,self.j_:self.j_+self.J],[0,1]) # (C)
        return tf.scatter_nd_update(self.W_,[[self.k_,self.i_,self.j_]],[(k1-k2-k3)/c1])
        rescalor = tf.reduce_min(self.sigmas2_)
        # FOR THE BIAS
        KR      = rescalor*self.filter_corr(self.W_,self.W_)*(1-self.E)
        BBB     = tf.reshape(tf.diag(tf.one_hot(self.k_,self.K)),[self.K,1,1,self.K,1,1])#*tf.reshape(tf.diag(tf.one_hot(self.r_,self.R)),[1,1,1,self.R,1,1,1,self.R])
        k1      = -0.5*tf.gradients(tf.reduce_sum(KR*self.SMpMp*(1-BBB)),self.W_)[0][self.k_]# (a b C)
        #
        binput  = tf.reduce_sum(self.SmMp[self.k_]*rescalor/self.sigmas2_patch_-myexpand(self.SMp[self.k_],[-1,-1,-1])*self.b_patch_*rescalor/self.sigmas2_patch_,[0,1]) # (self.C self.Ic self.Jc)
        B       = tf.reshape(tf.transpose(binput+k1,[2,0,1]),[self.C,-1])
        # FOR THE MATRIX
        DM2V2   = tf.transpose(tf.reduce_sum(myexpand(self.SM2V2[self.k_],[-1,-1,-1])*rescalor/self.sigmas2_patch_,[0,1]),[2,0,1]) # (self.C self.Ic self.Jc)
        rS5     = tf.transpose(tf.reshape(tf.extract_image_patches(tf.transpose(self.SMpMp[self.k_,:,:,self.k_,:,:],[0,2,3,1]),(1,self.Ic,self.Jc,1),(1,1,1,1),(1,1,1,1),"VALID"),(self.I,self.Ic,self.Jc,self.Ic,self.Jc,self.J)),[0,5,1,2,3,4])
        a1      = tf.einsum('ijabcd,ijabk->kabcd',rS5[:,:,:,:,::-1,::-1],0.5*rescalor/self.sigmas2_patch_)
        a2      = tf.einsum('ijabcd,ijcdk->kabcd',rS5[:,:,::-1,::-1],0.5*rescalor/self.sigmas2_patch_)
        A       = tf.reshape(a1+a2,[self.C,self.Ic*self.Jc,self.Ic*self.Jc])*(1-tf.expand_dims(tf.eye(self.Ic*self.Jc),0))+tf.matrix_diag(tf.reshape(DM2V2,[self.C,-1])+rescalor*self.sparsity_prior)
        with(tf.device('/device:CPU:0')):
            W      = tf.reshape(tf.matrix_solve(A,tf.expand_dims(B,-1)),[self.C,self.Ic,self.Jc])
        return tf.scatter_update(self.W_,[self.k_],[tf.transpose(W,[1,2,0])])
    def update_pi(self): return tf.assign(self.pi,self.Sp/(self.I*self.J))
    def update_BV(self):
        if(self.update_b == False): return []
        value = self.Sm-tf.gradients(self.sigmas2_patch_,self.sigmas2_,tf.reduce_sum(myexpand(self.SMp,[-1,-1,-1])*myexpand(self.W_,[1,1]),0))[0]
        if(self.update_b=='local'): return tf.assign(self.b_,value)
        rescalor = tf.reduce_min(self.sigmas2_)
        value1 = value*rescalor/self.sigmas2_
        value2 = rescalor/self.sigmas2_
        if(self.update_b=='channel'): return tf.assign(self.b_,tf.ones_like(self.b_)*tf.reduce_sum(value1,[0,1],keepdims=True)/tf.reduce_sum(value2,[0,1],keepdims=True))
        return tf.assign(self.b_,tf.ones_like(self.b_)*tf.reduce_sum(value1,keepdims=True)/tf.reduce_sum(value2,keepdims=True))
    def update_sigma(self,get_value=False):
        k1  = self.Sm2+self.Sv2-2*self.b_*self.Sm+tf.square(self.b_)
        k20 = tf.reduce_sum(myexpand(self.SM2V2,[-1,-1,-1])*myexpand(tf.square(self.W_),[1,1]),0) # (I J a b c)
        k21 = tf.reduce_sum(self.SmMp*myexpand(self.W_,[1,1]),0) # (I J a b c)
        k2  = tf.gradients(self.sigmas2_patch_,self.sigmas2_,k20-2*k21)[0]
        k3  = tf.gradients(self.sigmas2_patch_,self.sigmas2_,tf.reduce_sum(myexpand(self.SMp,[-1,-1,-1])*myexpand(self.W_,[1,1]),0))[0]*self.b_*2
        u   = tf.zeros_like(self.sigmas2_)
        pu  = self.extract_patch(tf.expand_dims(u,0),with_n=0) # ( I J a b c)
        def helper(A,B):
            k1 = tf.reshape(myexpand(A,[1,1])*pu,[self.K*self.I*self.J,self.Ic,self.Jc,self.C]) # (K I J a b c)->(KIJ a b c)
            k2 = tf.pad(k1,[[0,0],[self.Ic-1,self.Ic-1],[self.Jc-1,self.Jc-1],[0,0]]) # (KIJ 3a-2 3b-2 c)
            k3 = tf.reshape(tf.extract_image_patches(k2,(1,self.Ic,self.Jc,1),(1,1,1,1),(1,1,1,1),"VALID"),(self.K*self.I*self.J,2*self.Ic-1,2*self.Jc-1,self.Ic,self.Jc,self.C)) # (KIJ 2a-1 2b-1 a b c)
            k4 = tf.reduce_sum(tf.expand_dims(k3,1)*myexpand(B,[0,-4,-4]),[4,5,6]) # (KIJ K 2a-1 2b-1)
            return tf.reshape(k4,[self.K,self.I,self.J,self.K,2*self.Ic-1,2*self.Jc-1])
        scalar = tf.reduce_sum(self.SMpMp*helper(self.W_,self.W_)*(1-self.E))#(K I J K' I' J')
        k4     = tf.gradients(scalar,u)[0]
        value  = k1+k2+k3+k4
        if(get_value): return value
        if(self.sigma_opt=='local'):     return tf.assign(self.sigmas2_,value)
        elif(self.sigma_opt=='channel'): return tf.assign(self.sigmas2_,tf.reduce_sum(value,[0,1],keepdims=True)/(self.Iin*self.Jin)*tf.ones([self.Iin,self.Jin,1]))
	elif(self.sigma_opt=='global'):  return tf.assign(self.sigmas2_,tf.fill([self.Iin,self.Jin,self.C],tf.reduce_sum(value)/(self.Iin*self.Jin*self.C)))











class ConvLayer2(Layer):
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
        self.gamma_ = tf.Variable(tf.ones(self.K))
        self.input_shape       = input_layer.output_shape
        self.output_shape      = (self.bs,self.input_shape[-3]-self.Ic+1,self.input_shape[-2]-self.Jc+1,K)
	self.D_in              = prod(self.input_shape[1:])
        self.I,self.J          = self.output_shape[1],self.output_shape[2]
        self.input_patch       = self.extract_patch(self.input,with_n=1)
        if(leakiness == None):
            self.W_                = tf.Variable(init_W((self.K,self.R,self.Ic,self.Jc,self.C)))#/sqrt(self.K+self.Ic+self.Jc+self.C))
            self.W                 = self.W_*myexpand(self.gamma_,[-1,-1,-1,-1])
        else:
            self.W_                = tf.Variable(init_W((self.K,1,self.Ic,self.Jc,self.C)))#/sqrt(self.K+self.Ic+self.Jc+self.C))
            self.W                 = tf.concat([self.W_,leakiness*self.W_],axis=1)
	# WE DEFINE THE PARAMETERS
        self.pi             = tf.Variable(mysoftmax(tf.zeros((K,R)),axis=1))
	self.sigmas2_       = tf.Variable(tf.ones((self.Iin,self.Jin,self.C)))
	self.sigmas2        = tf.expand_dims(self.sigmas2_,0)
        self.sigmas2_patch_ = self.extract_patch(self.sigmas2,with_n=0)
	self.sigmas2_patch  = tf.expand_dims(self.sigmas2_patch_,0)
	self.b_             = tf.Variable(tf.zeros(self.input_shape[1:]))
	self.b              = tf.expand_dims(self.b_,0)
        self.b_patch_       = self.extract_patch(self.b,with_n=0)
        self.b_patch        = tf.expand_dims(self.b_patch_,0)
        # VI VARIABLES
	self.m_      = tf.Variable(tf.zeros((K,self.I,self.J,self.bs)))
        self.m       = tf.transpose(self.m_,[3,1,2,0])                            # (N I J K)
	self.p_      = tf.Variable(tf.zeros((K,self.I,self.J,self.R,self.bs)))    # (K,I,J,R,N)
        self.p       = tf.transpose(self.p_,[4,1,2,0,3])                          # (N I J K R)
        self.v2_     = tf.Variable(tf.ones((self.K,self.I,self.J,self.bs)))       # (K I J N)
        self.v2      = tf.transpose(self.v2_,[3,1,2,0])
	self.drop_   = tf.Variable(tf.ones((K,2,self.I,self.J,self.bs))*tf.reshape(tf.one_hot(1,2),(1,2,1,1,1))) # (K 2 I J N)
        self.drop    = tf.transpose(self.drop_,[1,4,2,3,0]) #(2 N I J K)
        # STATISTICS
        with tf.device('/device:GPU:0'):
            self.Sm2   = tf.Variable(tf.zeros((self.C,self.Iin,self.Jin)))
            self.Sv2   = tf.Variable(tf.zeros((self.C,self.Iin,self.Jin)))
            self.SmMp  = tf.Variable(tf.zeros((self.K,self.I,self.J,self.R,self.Ic,self.Jc,self.C)))
            self.SM2V2 = tf.Variable(tf.zeros((self.K,self.I,self.J,self.R)))
            self.SMpMp = tf.Variable(tf.zeros((self.K,self.I,self.J,self.R,self.K,self.Ic*2-1,self.Jc*2-1,self.R)))
            self.SMp   = tf.Variable(tf.zeros((self.K,self.I,self.J,self.R)))
            self.Sm    = tf.Variable(tf.zeros((self.C,self.Iin,self.Jin)))
            self.Sp    = tf.Variable(tf.zeros((self.K,self.R)))
	#
        input_layer.next_layer = self
        self.k_      = tf.placeholder(tf.int32)
        self.i_      = tf.placeholder(tf.int32)
        self.j_      = tf.placeholder(tf.int32)
        self.r_      = tf.placeholder(tf.int32)
	self.ratio   = Ic
        self.Ni      = tf.cast((self.I-self.i_-1)/self.ratio+1,tf.int32) # NUMBER OF TERMS
        self.Nj      = tf.cast((self.J-self.j_-1)/self.ratio+1,tf.int32) # NUMBER OF TERMS
        self.xi,self.yi = tf.meshgrid(tf.range(self.j_,self.J,self.ratio),tf.range(self.i_,self.I,self.ratio)) # THE SECOND IS CONSTANT (meshgrid)
        self.indices_   = tf.concat([tf.fill([self.Ni*self.Nj,1],self.k_),tf.reshape(self.yi,(self.Ni*self.Nj,1)),tf.reshape(self.xi,(self.Nj*self.Ni,1))],axis=1) # (V 3) indices where the 1 pops
        self.m_indices_ = tf.concat([tf.concat([tf.fill([self.Ni*self.Nj,1],KK),tf.reshape(self.yi,(self.Ni*self.Nj,1)),tf.reshape(self.xi,(self.Nj*self.Ni,1))],axis=1) for KK in range(self.K)],axis=0)
        if(leakiness is None):
            self.W_indices = asarray([a for a in itertools.product(range(self.K),range(self.R))])
        else:
            self.W_indices = asarray([a for a in itertools.product(range(self.K),range(1))])
        self.m_indices = asarray([a for a in itertools.product(range(self.ratio),range(self.ratio))])#ratio
	self.p_indices = asarray([a for a in itertools.product(range(self.K),range(self.ratio),range(self.ratio))])
	mask           = tf.reshape(tf.one_hot(self.k_,self.K),(self.K,1,1))*tf.reshape(tf.tile(tf.one_hot(self.i_,self.ratio),[(self.I/self.ratio+1)]),(1,(self.I/self.ratio+1)*self.ratio,1))*tf.reshape(tf.tile(tf.one_hot(self.j_,self.ratio),[self.J/self.ratio+1]),(1,1,(self.J/self.ratio+1)*self.ratio))
        self.mask      = tf.expand_dims(mask[:,:self.I,:self.J],-1) # (I J K)
        m_mask         = tf.reshape(tf.tile(tf.one_hot(self.i_,self.ratio),[(self.I/self.ratio+1)]),((self.I/self.ratio+1)*self.ratio,1))*tf.reshape(tf.tile(tf.one_hot(self.j_,self.ratio),[self.J/self.ratio+1]),(1,(self.J/self.ratio+1)*self.ratio))
        self.m_mask    = m_mask[:self.I,:self.J]
        self.E= tf.einsum('ij,kc,a,b->kijcab',tf.ones((self.I,self.J)),tf.eye(self.K),tf.one_hot(self.Ic-1,2*self.Ic-1),tf.one_hot(self.Ic-1,2*self.Ic-1))
    def init_W(self,W):  return tf.assign(self.W_,W)
    def update_BN(self):
        return []
        alpha = 0.8
        mu    = tf.reduce_sum(self.m_,[1,2,3],keepdims=True)#/self.bs+(1-alpha)*self.mu_
        return tf.assign(self.gamma_,alpha*(tf.sqrt(tf.reduce_sum(tf.square(self.m_-mu),[1,2,3])/self.bs)+0.1)+(1-alpha)*self.gamma_)
    def filter_corr(self,A,B):
        #takes as input filter banks A and B of same shape (K a b c)
        k1 = tf.reshape(myexpand(A,[1,1])/self.sigmas2_patch,[self.K*self.I*self.J,self.Ic,self.Jc,self.C]) # (K I J a b c)->(KIJ a b c)
        k2 = tf.pad(k1,[[0,0],[self.Ic-1,self.Ic-1],[self.Jc-1,self.Jc-1],[0,0]]) # (KIJ 3a-2 3b-2 c)
        k3 = tf.reshape(tf.extract_image_patches(k2,(1,self.Ic,self.Jc,1),(1,1,1,1),(1,1,1,1),"VALID"),(self.K*self.I*self.J,2*self.Ic-1,2*self.Jc-1,self.Ic,self.Jc,self.C)) # (KIJ 2a-1 2b-1 a b c)
        k4 = tf.reduce_sum(tf.expand_dims(k3,1)*myexpand(B,[0,-4,-4]),[4,5,6]) # (KIJ K 2a-1 2b-1)
        return tf.reshape(k4,[self.K,self.I,self.J,self.K,2*self.Ic-1,2*self.Jc-1])
        sigma_W = tf.pad(tf.transpose(tf.reshape(tf.einsum('ijabc,kabc->abckij',1/self.sigmas2_patch_,A),[self.Ic,self.Jc,self.C,self.K*self.I*self.J]),[3,0,1,2]),[[0,0],[self.Ic-1,self.Ic-1],[self.Jc-1,self.Jc-1],[0,0]])
        patches = tf.reshape(tf.extract_image_patches(sigma_W,(1,self.Ic,self.Jc,1),(1,1,1,1),(1,1,1,1),"VALID"),(self.K*self.I*self.J,2*self.Ic-1,2*self.Jc-1,self.Ic,self.Jc,self.C))
        return tf.transpose(tf.reshape(tf.einsum('nijabc,kabc->kijn',patches,B),[self.K,2*self.Ic-1,2*self.Jc-1,self.K,self.I,self.J]),[3,4,5,0,1,2])#*(1-self.E)#(K I J K' I' J')
    def update_S(self):
        Sm2   = tf.reduce_sum(tf.square(self.input_layer.m_),3)/self.bs
        Sv2   = tf.reduce_sum(self.input_layer.v2_,3)/self.bs
        SmMp  = tf.transpose(tf.reduce_sum(myexpand(self.input_patch,[3,4])*myexpand(tf.expand_dims(self.m,-1)*self.p,[-1,-1,-1]),0),[2,0,1,3,4,5,6])/self.bs
        SM2V2 = tf.reduce_sum(tf.expand_dims(tf.square(self.m_)+self.v2_,-2)*self.p_,4)/self.bs

        mp_patches = [tf.transpose(tf.reshape(tf.extract_image_patches(tf.pad(self.m*self.p[:,:,:,:,r],[[0,0],[self.Ic-1,self.Ic-1],[self.Jc-1,self.Jc-1],[0,0]]),(1,2*self.Ic-1,2*self.Jc-1,1),(1,1,1,1),(1,1,1,1),"VALID"),(self.bs,self.I,self.J,2*self.Ic-1,2*self.Jc-1,self.K)),[0,1,2,5,3,4]) for r in xrange(self.R)] # R x (N I J I' J' K)->R x (N I J K I' J')
        SMpMp=mp_patches = tf.transpose(tf.reshape(tf.stack([tf.reduce_sum(tf.expand_dims(mp_patches[r2],1)*myexpand(tf.transpose(self.m*self.p[:,:,:,:,r1],[0,3,1,2]),[-1,-1,-1]),0)/self.bs for r1,r2 in itertools.product(range(self.R),range(self.R))],-1),[self.K,self.I,self.J,self.K,self.Ic*2-1,self.Jc*2-1,self.R,self.R]),[0,1,2,6,3,4,5,7])
        SMp = tf.reduce_sum(tf.expand_dims(self.m_,-2)*self.p_,-1)/self.bs
        Sm  = tf.reduce_sum(self.input_layer.m_,3)/self.bs
        Sp  = tf.reduce_sum(tf.clip_by_value(self.p_,0.02,0.98),[1,2,4])/self.bs
        return tf.group(tf.assign(self.Sm2,self.alpha*Sm2+(1-self.alpha)*self.Sm2),tf.assign(self.Sv2,self.alpha*Sv2+(1-self.alpha)*self.Sv2),
                        tf.assign(self.SmMp,self.alpha*SmMp+(1-self.alpha)*self.SmMp),tf.assign(self.SM2V2,self.alpha*SM2V2+(1-self.alpha)*self.SM2V2),
                        tf.assign(self.SMpMp,self.alpha*SMpMp+(1-self.alpha)*self.SMpMp),tf.assign(self.SMp,self.alpha*SMp+(1-self.alpha)*self.SMp),
                        tf.assign(self.Sm,self.alpha*Sm+(1-self.alpha)*self.Sm),tf.assign(self.Sp,self.alpha*Sp+(1-self.alpha)*self.Sp))
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
	if(masked_m==1):    mask = p*tf.expand_dims(m*(1-self.mask),3)
	elif(masked_m==-1): mask = p*tf.expand_dims(m*self.mask,3)
        else:               mask = p*tf.expand_dims(m,3) # (K I J R N)
        proj = tf.reduce_sum(myexpand(tf.transpose(mask,[4,1,2,0,3]),[-1,-1,-1])*myexpand(self.W,[0,0,0]),[3,4])# (N I J a b c)
        return tf.gradients(self.input_patch,self.input,proj)[0]
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
            k1  = -float32(self.D_in/2.0)*tf.log(2*PI_CONST)-tf.reduce_sum(tf.log(self.sigmas2_))/2+tf.reduce_sum(tf.log(self.pi)*tf.reduce_sum(self.p_,[1,2,4]))/float32(self.bs)
            k2  = -tf.reduce_sum((tf.square(self.input-self.deconv()-self.b)+self.input_layer.v2)/(2*self.sigmas2*self.bs))
            k30 = tf.reduce_sum(tf.square(tf.reduce_sum(myexpand(self.p_,[-1,-1,-1])*myexpand(self.W,[1,1,-4]),3))/myexpand(self.sigmas2_patch_,[0,-4]),[4,5,6])
            k3  = tf.reduce_sum(tf.square(self.m_)*k30)*0.5/self.bs
            k4  = -tf.reduce_sum(tf.reduce_sum(tf.expand_dims(tf.square(self.m_)+self.v2_,-2)*self.p_,4)*tf.reduce_sum(myexpand(tf.square(self.W),[1,1])/myexpand(self.sigmas2_patch_,[0,3]),[4,5,6]))*0.5/self.bs
            return k1+k2+k3+k4-self.sparsity_prior*tf.reduce_sum(tf.square(self.W))/2+extra_k
        else:
            k1 = -float32(self.D_in/2.0)*tf.log(2*PI_CONST)-tf.reduce_sum(tf.log(self.sigmas2_))/2+tf.reduce_sum(self.Sp*tf.log(self.pi))
            k2 = -tf.reduce_sum((self.Sm2+self.Sv2+tf.square(tf.transpose(self.b_,[2,0,1]))-2*self.Sm*tf.transpose(self.b_,[2,0,1]))/tf.transpose(self.sigmas2_,[2,0,1]))*0.5
            k3 = tf.reduce_sum(self.SmMp*myexpand(self.W,[1,1])/tf.expand_dims(self.sigmas2_patch,3))
            k4 = -tf.reduce_sum(tf.reduce_sum(myexpand(self.SMp,[-1,-1,-1])*myexpand(self.W,[1,1]),3)*self.b_patch/self.sigmas2_patch)
            k5 = -tf.reduce_sum(tf.reduce_sum(myexpand(self.SM2V2,[-1,-1,-1])*myexpand(tf.square(self.W),[1,1]),3)/self.sigmas2_patch)*0.5
            filters = tf.transpose(tf.reshape(tf.stack([self.filter_corr(self.W[:,r1],self.W[:,r2])*(1-self.E) for r1,r2 in itertools.product(range(self.R),range(self.R))],-1),[self.K,self.I,self.J,self.K,self.Ic*2-1,self.Jc*2-1,self.R,self.R]),[0,1,2,6,3,4,5,7]) # ( k i j r k i' j' r)
            k6 = -tf.reduce_sum(self.SMpMp*filters)/2
            return k1+k2+k3+k4+k5+k6-self.sparsity_prior*tf.reduce_sum(tf.square(self.W))/2
    def KL(self,pretraining=False):
        return self.likelihood(0,pretraining)+(-tf.reduce_sum(self.p_*tf.log(self.p_+eps))+float32(0.5)*tf.reduce_sum(tf.log(self.v2_)))/float32(self.bs)+float32(self.D_in/2.0)*tf.log(2*PI_CONST)
    def update_v2(self,pretraining=False):# DONE
        if(isinstance(self.next_layer,ConvLayer) or isinstance(self.next_layer,PoolLayer)): v_value = self.next_layer.sigmas2 # (N I J K)
        else:            v_value = tf.expand_dims(tf.reshape(self.next_layer.sigmas2_,self.output_shape[1:]),0)
        if(pretraining==False):next_sigmas = v_value
        else:                  next_sigmas = tf.ones_like(v_value)
        rescalor = tf.reduce_min(self.sigmas2)
        a4       = tf.reduce_sum(self.p_*tf.expand_dims(tf.reduce_sum(myexpand(tf.square(self.W),[1,1])*rescalor/tf.expand_dims(self.sigmas2_patch,3),[4,5,6]),-1),3)
        return tf.assign(self.v2_,rescalor/(tf.transpose(rescalor/next_sigmas,[3,1,2,0])+a4))
    def update_m(self,mp_opt=0,pretraining=False):
        m_masked       = tf.expand_dims(self.m*myexpand(1-self.m_mask,[0,-1]),-1)*self.p # (N I J K R)
        reconstruction = tf.gradients(self.input_patch,self.input,tf.reduce_sum(myexpand(m_masked,[-1,-1,-1])*myexpand(self.W,[0,0,0]),[3,4]))[0]
        Wp             = tf.reduce_sum(myexpand(self.W,[0,0,0])*myexpand(self.p[:,self.i_::self.ratio,self.j_::self.ratio],[-1,-1,-1]),4)                    #(N I J K A B C)
	if(isinstance(self.next_layer,ConvLayer) or isinstance(self.next_layer,PoolLayer)): 
            back = (self.next_layer.deconv()[:,self.i_::self.ratio,self.j_::self.ratio]+self.next_layer.b[:,self.i_::self.ratio,self.j_::self.ratio])
            A22   = self.next_layer.sigmas2_[self.i_::self.ratio,self.j_::self.ratio]
	else:       
            back = self.next_layer.vector2tensor(self.next_layer.backward()+self.next_layer.b)[:,self.i_::self.ratio,self.j_::self.ratio]
            A22   = tf.reshape(self.next_layer.sigmas2_,self.output_shape[1:])[self.i_::self.ratio,self.j_::self.ratio]
        if(pretraining):
            next_back   = tf.zeros_like(back)
            next_sigmas = tf.ones_like(A22)
        else:
            next_back   = back
            next_sigmas = A22
        rescalor = tf.reduce_min(self.sigmas2_patch_[self.i_::self.ratio,self.j_::self.ratio])
        bias0    = tf.reduce_sum(tf.expand_dims(self.extract_patch(self.input-reconstruction-self.b)[:,self.i_::self.ratio,self.j_::self.ratio]*rescalor/self.sigmas2_patch[:,self.i_::self.ratio,self.j_::self.ratio],3)*Wp,[4,5,6])
        bias     = next_back*rescalor/tf.expand_dims(next_sigmas,0)+bias0
        Wprs     = Wp*rescalor/myexpand(self.sigmas2_patch_[self.i_::self.ratio,self.j_::self.ratio],[0,-4])#,[self.bs,self.Ni,self.Nj,self.K,-1])
        A1       = tf.reduce_sum(myexpand(Wprs,[4])*myexpand(Wp,[3]),[5,6,7])*myexpand(1-tf.eye(self.K),[0,0,0]) # ( N I J K K)
        A2       = tf.reduce_sum(myexpand(tf.square(self.W),[0,0,0])*myexpand(self.p[:,self.i_::self.ratio,self.j_::self.ratio],[-1,-1,-1])*rescalor/myexpand(self.sigmas2_patch_[self.i_::self.ratio,self.j_::self.ratio],[0,-4,-4]),[4,5,6,7]) #(K I' J' R)
        A        = A1+tf.matrix_diag(A2)+tf.expand_dims(tf.matrix_diag(rescalor/next_sigmas),0)
        with(tf.device('/device:CPU:0')):
            new_m = tf.transpose(tf.matrix_solve(A,tf.expand_dims(bias,-1))[:,:,:,:,0],[0,3,1,2]) # (N K I J)
	return tf.scatter_nd_update(self.m_,self.m_indices_,tf.transpose(tf.reshape(new_m,(self.bs,-1))))
    def update_p(self):
        reconstruction = self.deconv(masked_m=True)# (N Iin Jin C)
        forward  = tf.reduce_sum(tf.expand_dims(self.extract_patch((self.input-reconstruction-self.b)/self.sigmas2)[:,self.i_::self.ratio,self.j_::self.ratio],3)*myexpand(self.W[self.k_],[0,0,0]),[4,5,6])*tf.expand_dims(self.m[:,self.i_::self.ratio,self.j_::self.ratio,self.k_],-1) #(N I' J' R)
        m2v2     = tf.expand_dims(tf.square(self.m[:,self.i_::self.ratio,self.j_::self.ratio,self.k_])+self.v2[:,self.i_::self.ratio,self.j_::self.ratio,self.k_],-1)*tf.expand_dims(tf.reduce_sum(myexpand(tf.square(self.W[self.k_]),[0,0])*0.5/tf.expand_dims(self.sigmas2_patch_[self.i_::self.ratio,self.j_::self.ratio],2),[3,4,5]),0) #(N I' J' R)
	value    = forward-m2v2+myexpand(tf.log(self.pi[self.k_]),[0,0,0])# (N I' J' R)
        update_value = tf.transpose(tf.nn.softmax(value),[3,0,1,2])
	return tf.scatter_nd_update(self.p_,self.indices_,tf.transpose(tf.reshape(update_value,(self.R,self.bs,-1)),[2,0,1]))
    def update_Wk(self):
        rescalor = tf.reduce_min(self.sigmas2_)
        # FOR THE BIAS
        KR      = tf.transpose(tf.reshape(tf.stack([rescalor*self.filter_corr(self.W[:,r1],self.W[:,r2])*(1-self.E) for r1,r2 in itertools.product(range(self.R),range(self.R))],-1),[self.K,self.I,self.J,self.K,self.Ic*2-1,self.Jc*2-1,self.R,self.R]),[0,1,2,6,3,4,5,7])
        BBB     = tf.reshape(tf.diag(tf.one_hot(self.k_,self.K)),[self.K,1,1,1,self.K,1,1,1])*tf.reshape(tf.diag(tf.one_hot(self.r_,self.R)),[1,1,1,self.R,1,1,1,self.R])#myexpand(tf.one_hot(self.r_,self.R),[0,0,0,-1,-1,-1,-1])
        k1      = -0.5*tf.gradients(tf.reduce_sum(KR*self.SMpMp*(1-BBB)),self.W)[0][self.k_,self.r_]# (a b C)
        #
#        BB      = self.SMpMp[self.k_][:,:,self.r_,self.k_,self.Ic-1,self.Jc-1,:]
#        bsamek  = -tf.reduce_sum(tf.reduce_sum(myexpand(self.W[self.k_],[0,0])*myexpand(BB*(1-myexpand(tf.one_hot(self.r_,self.R),[0,0])),[-1,-1,-1]),2)/self.sigmas2_patch_,[0,1])# (Ic Jc C)
        #
        binput  = tf.reduce_sum(self.SmMp[self.k_,:,:,self.r_]*rescalor/self.sigmas2_patch_-myexpand(self.SMp[self.k_,:,:,self.r_],[-1,-1,-1])*self.b_patch_*rescalor/self.sigmas2_patch_,[0,1]) # (self.C self.Ic self.Jc)
        B       = tf.reshape(tf.transpose(binput+k1,[2,0,1]),[self.C,-1])
        # FOR THE MATRIX
        DM2V2   = tf.transpose(tf.reduce_sum(myexpand(self.SM2V2[self.k_,:,:,self.r_],[-1,-1,-1])*rescalor/self.sigmas2_patch_,[0,1]),[2,0,1]) # (self.C self.Ic self.Jc)
        BBB     = tf.reduce_sum(self.SMpMp*myexpand(tf.one_hot(self.k_,self.K),[1,1,1,1,1,1,1]),0)
        rS5     = tf.transpose(tf.reshape(tf.extract_image_patches(tf.transpose(BBB[:,:,self.r_,self.k_,:,:,self.r_],[0,2,3,1]),(1,self.Ic,self.Jc,1),(1,1,1,1),(1,1,1,1),"VALID"),(self.I,self.Ic,self.Jc,self.Ic,self.Jc,self.J)),[0,5,1,2,3,4])
        a1      = tf.einsum('ijabcd,ijabk->kabcd',rS5[:,:,:,:,::-1,::-1],0.5*rescalor/self.sigmas2_patch_)
        a2      = tf.einsum('ijabcd,ijcdk->kabcd',rS5[:,:,::-1,::-1],0.5*rescalor/self.sigmas2_patch_)
        A       = tf.reshape(a1+a2,[self.C,self.Ic*self.Jc,self.Ic*self.Jc])*(1-tf.expand_dims(tf.eye(self.Ic*self.Jc),0))+tf.matrix_diag(tf.reshape(DM2V2,[self.C,-1])+rescalor*self.sparsity_prior)
        with(tf.device('/device:CPU:0')):
            W      = tf.reshape(tf.matrix_solve(A*tf.square(self.gamma_[self.k_]),self.gamma_[self.k_]*tf.expand_dims(B,-1)),[self.C,self.Ic,self.Jc])
        return tf.scatter_nd_update(self.W_,[[self.k_,self.r_]],[tf.transpose(W,[1,2,0])])
    def update_pi(self): return tf.assign(self.pi,self.Sp/tf.reduce_sum(self.Sp,axis=1,keepdims=True))
    def update_BV(self):
        return []
        if(self.update_b == False): return []
        value = tf.transpose(self.Sm,[1,2,0])-tf.gradients(self.sigmas2_patch_,self.sigmas2_,tf.reduce_sum(myexpand(self.SMp,[-1,-1,-1])*myexpand(self.W,[1,1]),[0,3]))[0]
        if(self.update_b=='local'): return tf.assign(self.b_,value)
        rescalor = tf.reduce_min(self.sigmas2_)
        value1 = value*rescalor/self.sigmas2_
        value2 = rescalor/self.sigmas2_
        if(self.update_b=='channel'): return tf.assign(self.b_,tf.ones_like(self.b_)*tf.reduce_sum(value1,[0,1],keepdims=True)/tf.reduce_sum(value2,[0,1],keepdims=True))
        return tf.assign(self.b_,tf.ones_like(self.b_)*tf.reduce_sum(value1,keepdims=True)/tf.reduce_sum(value2,keepdims=True))
    def update_sigma(self,get_value=False):
#        return []
        k1  = tf.transpose(self.Sm2+self.Sv2-2*tf.transpose(self.b_,[2,0,1])*self.Sm,[1,2,0])+tf.square(self.b_)
        k20 = tf.reduce_sum(myexpand(self.SM2V2,[-1,-1,-1])*myexpand(tf.square(self.W),[1,1]),[0,3])
        k21 = -2*tf.reduce_sum(self.SmMp*myexpand(self.W,[1,1]),[0,3])
        k2  = tf.gradients(self.sigmas2_patch_,self.sigmas2_,k20+k21)[0]
        k3  = tf.gradients(self.sigmas2_patch_,self.sigmas2_,tf.reduce_sum(myexpand(self.SMp,[-1,-1,-1])*myexpand(self.W,[1,1]),[0,3]))[0]*self.b_*2
        u   = tf.zeros_like(self.sigmas2_)
        pu  = self.extract_patch(tf.expand_dims(u,0),with_n=0) # ( I J a b c)
        def helper(r1,r2):
            k1 = tf.reshape(myexpand(self.W[:,r1],[1,1])*pu,[self.K*self.I*self.J,self.Ic,self.Jc,self.C]) # (K I J a b c)->(KIJ a b c)
            k2 = tf.pad(k1,[[0,0],[self.Ic-1,self.Ic-1],[self.Jc-1,self.Jc-1],[0,0]]) # (KIJ 3a-2 3b-2 c)
            k3 = tf.reshape(tf.extract_image_patches(k2,(1,self.Ic,self.Jc,1),(1,1,1,1),(1,1,1,1),"VALID"),(self.K*self.I*self.J,2*self.Ic-1,2*self.Jc-1,self.Ic,self.Jc,self.C)) # (KIJ 2a-1 2b-1 a b c)
            k4 = tf.reduce_sum(tf.expand_dims(k3,1)*myexpand(self.W[:,r2],[0,-4,-4]),[4,5,6]) # (KIJ K 2a-1 2b-1)
            return tf.reshape(k4,[self.K,self.I,self.J,self.K,2*self.Ic-1,2*self.Jc-1])
        MpMp   = tf.transpose(tf.reshape(tf.stack([helper(r1,r2)*(1-self.E) for r1,r2 in itertools.product(range(self.R),range(self.R))],-1),[self.K,self.I,self.J,self.K,self.Ic*2-1,self.Jc*2-1,self.R,self.R]),[0,1,2,6,3,4,5,7])
        scalar = tf.reduce_sum(self.SMpMp*MpMp)#(K I J K' I' J')
        k4     = tf.gradients(scalar,u)[0]
        value  = k1+k2+k3+k4
        if(get_value): return value
        if(self.sigma_opt=='local'):     return tf.assign(self.sigmas2_,value)
        elif(self.sigma_opt=='channel'): return tf.assign(self.sigmas2_,tf.reduce_sum(value,[0,1],keepdims=True)*tf.ones([self.Iin,self.Jin,1])/(self.Iin*self.Jin))
	elif(self.sigma_opt=='global'):  return tf.assign(self.sigmas2_,tf.fill([self.Iin,self.Jin,self.C],tf.reduce_sum(value)/(self.Iin*self.Jin*self.C)))








class PoolLayer(Layer):
    def __init__(self,input_layer,Ic,Jc,Dc=1,sigma='local',alpha=0.5):
        self.alpha = tf.Variable(alpha)
	self.sigma_opt         = sigma
        self.input_layer       = input_layer
        input_layer.next_layer = self
        self.bs,self.Iin,self.Jin,self.C  = input_layer.output_shape 
        self.Ic,self.Jc,self.K,self.Dc = Ic,Jc,input_layer.output_shape[-1]/Dc,Dc
        K=self.K
        R,self.R = Ic*Jc,Ic*Jc
        self.input             = input_layer.m
        self.input_shape       = input_layer.output_shape
        self.output_shape      = (self.bs,self.input_shape[-3]/self.Ic,self.input_shape[-2]/self.Jc,K)
	self.D_in              = prod(self.input_shape[1:])
        self.I,self.J          = self.output_shape[1],self.output_shape[2]
        self.input_patch       = self.extract_patch(self.input,with_n=1)
	# WE DEFINE THE PARAMETERS
        self.pi      = tf.Variable(tf.ones((Ic,Jc))/float32(R))
	self.sigmas2_= tf.Variable(tf.ones((self.Iin,self.Jin,self.C)))
	self.sigmas2 = tf.expand_dims(self.sigmas2_,0)
        self.sigmas2_patch_= self.extract_patch(self.sigmas2,with_n=0)
	self.sigmas2_patch = tf.expand_dims(self.sigmas2_patch_,0)
	# SOME OTHER VARIABLES
	self.b_      = tf.Variable(tf.zeros_like(self.sigmas2_))
	self.b       = tf.expand_dims(self.b_,0)
	self.m_      = tf.Variable(tf.zeros((K,self.I,self.J,self.bs))) # (K I J N)
        self.m       = tf.transpose(self.m_,[3,1,2,0])   # (N I J K)
	self.p_      = tf.Variable(tf.zeros((K,self.I,self.J,self.Ic,self.Jc,self.Dc,self.bs)))# (K,I,J,a,b,N)
        self.p       = tf.transpose(self.p_,[6,1,2,3,4,0,5]) # (N I J a b K dc)
        self.v2_     = tf.Variable(tf.zeros((self.K,self.I,self.J,self.bs))) # (K I J N)
        self.v2      = tf.transpose(self.v2_,[3,1,2,0]) # (N I J K)
        # STATISTICS
        self.Sm2    = tf.Variable(tf.zeros((self.Iin,self.Jin,self.C)))
        self.Sv2    = tf.Variable(tf.zeros((self.Iin,self.Jin,self.C)))
        self.SmMp   = tf.Variable(tf.zeros((self.K,self.I,self.J,self.Ic,self.Jc,Dc)))
        self.SM2v2p = tf.Variable(tf.zeros((self.K,self.I,self.J,self.Ic,self.Jc,Dc)))
        self.SMp    = tf.Variable(tf.zeros((self.K,self.I,self.J,self.Ic,self.Jc,Dc)))
        self.Sm     = tf.Variable(tf.zeros((self.Iin,self.Jin,self.C)))
	#
        self.apodization = 1
        input_layer.next_layer = self
    def update_S(self):
        Sm2    = tf.reduce_sum(tf.square(self.input_layer.m),0)/self.bs
        Sv2    = tf.reduce_sum(self.input_layer.v2,0)/self.bs
        SmMp   = tf.transpose(tf.reduce_sum(self.input_patch*self.p*myexpand(self.m,[3,3,-1]),0),[4,0,1,2,3,5])/self.bs
        SM2v2p = tf.reduce_sum(myexpand(tf.square(self.m_)+self.v2_,[3,3,3])*self.p_,6)/self.bs
        SMp    = tf.reduce_sum(myexpand(self.m_,[3,3,3])*self.p_,6)/self.bs
        Sm     = tf.reduce_sum(self.input_layer.m,0)/self.bs
        return tf.group(tf.assign(self.Sm2,self.alpha*Sm2+(1-self.alpha)*self.Sm2),tf.assign(self.Sv2,self.alpha*Sv2+(1-self.alpha)*self.Sv2),
                        tf.assign(self.SmMp,self.alpha*SmMp+(1-self.alpha)*self.SmMp),tf.assign(self.SM2v2p,self.alpha*SM2v2p+(1-self.alpha)*self.SM2v2p),
                        tf.assign(self.SMp,self.alpha*SMp+(1-self.alpha)*self.SMp),tf.assign(self.Sm,self.alpha*Sm+(1-self.alpha)*self.Sm))
    def extract_patch(self,u,with_n=1,with_reshape=1):
	patches = tf.extract_image_patches(u,(1,self.Ic,self.Jc,1),(1,self.Ic,self.Jc,1),(1,1,1,1),"VALID")
	if(with_reshape):
	    if(with_n): return tf.reshape(patches,(self.bs,self.output_shape[1],self.output_shape[2],self.Ic,self.Jc,self.C/self.Dc,self.Dc))
	    else:       return tf.reshape(patches,(self.output_shape[1],self.output_shape[2],self.Ic,self.Jc,self.C/self.Dc,self.Dc))
	return patches
#                                           ---- BACKWARD OPERATOR ---- 
    def deconv(self,input=None,masked_m=0,masked_w=0,m=None,p=None):
	if(m is None):m=self.m
	if(p is None):p=self.p
        value = myexpand(m,[3,3,-1])*p# (N I J a b c)
        return tf.gradients(self.input_patch,self.input,value)[0]
    def sample(self,M,K=None,sigma=1):
	#multinomial returns [K,n_samples] with integer value 0,...,R-1
	if(isinstance(self.input_layer,InputLayer)):sigma=0
	noise      = sigma*tf.random_normal(self.input_shape)*tf.sqrt(self.sigmas2)
        sigma_hot  = tf.reshape(tf.one_hot(tf.reshape(tf.multinomial(tf.expand_dims(tf.log(tf.reshape(self.pi,[self.R])),0),self.bs*self.I*self.J*self.K*self.Dc),(self.K,self.Dc,self.I,self.J,self.bs)),self.R),[self.K,self.Dc,self.I,self.J,self.bs,self.Ic,self.Jc]) # (K dc I J N Ic Jc)
        return self.deconv(m=M,p=tf.transpose(sigma_hot,[4,2,3,5,6,0,1]))+noise
    def evidence(self): return 0
    def likelihood(self,batch=0,pretraining=False):
        if(batch==0):
            if(pretraining==False):  extra_k = 0
            else:                    extra_k = -0.5*tf.reduce_sum((tf.square(self.m_)+self.v2_)/self.bs)
            a1  = -tf.reduce_sum((tf.square(self.input)+self.input_layer.v2)/(2*self.sigmas2*self.bs))
            a2  = tf.reduce_sum(self.input*self.deconv()/self.sigmas2)/self.bs
            a3  = -tf.reduce_sum((tf.square(self.m)+self.v2)*tf.reduce_sum(self.p/self.sigmas2_patch,[3,4,6]))*0.5/self.bs
            k1  = -float32(self.D_in/2.0)*tf.log(2*PI_CONST)-tf.reduce_sum(tf.log(self.sigmas2_+eps))/2
            return k1+a1+a2+a3+extra_k
        else:
            k1  = -float32(self.D_in/2.0)*tf.log(2*PI_CONST)-tf.reduce_sum(tf.log(self.sigmas2_+eps))/2
            a1  = tf.reduce_sum((-self.Sm2-self.Sv2)/self.sigmas2_)*0.5
            a2  = tf.reduce_sum(tf.transpose(self.SmMp,[1,2,3,4,0,5])/self.sigmas2_patch_)
            a3 = -tf.reduce_sum(tf.transpose(self.SM2v2p,[1,2,3,4,0,5])/self.sigmas2_patch_)*0.5
            return k1+a1+a2+a3
    def KL(self,pretraining=False):
            return self.likelihood(0,pretraining)+(-tf.reduce_sum(self.p_*tf.log(self.p_+eps))+float32(0.5)*tf.reduce_sum(tf.log(self.v2_+eps)))/float32(self.bs)+float32(self.D_in/2.0)*tf.log(2*PI_CONST)
    def update_v2(self,pretraining=False):# DONE
        if(isinstance(self.next_layer,ConvLayer)): v_value = self.next_layer.sigmas2 # (N I J K)
        else:                                      v_value = tf.expand_dims(tf.reshape(self.next_layer.sigmas2_,self.output_shape[1:]),0) # (N I J K)
        if(pretraining):     next_sigmas = tf.ones_like(v_value)*1
        else:                next_sigmas = v_value
        rescalor = tf.reduce_min(self.sigmas2_)
        a4       = tf.transpose(tf.reduce_sum(self.p*rescalor/self.sigmas2_patch,[3,4,6]),[3,1,2,0])# (N I J K) -> (K I J N)
        return tf.assign(self.v2_,rescalor/(a4+tf.transpose(rescalor/next_sigmas,[3,1,2,0])))
    def update_m(self,mp_opt=0,pretraining=False):
        rescalor = tf.reduce_min(self.sigmas2_)
        forward  = tf.reduce_sum(self.extract_patch(self.input*rescalor/self.sigmas2)*self.p,[3,4,6])  # (N I J K)
	if(isinstance(self.next_layer,ConvLayer)):
            back = (self.next_layer.deconv()+self.next_layer.b)
            back_sigma = self.next_layer.sigmas2
	else:                                      
            back = self.next_layer.vector2tensor(self.next_layer.backward()+self.next_layer.b)
            back_sigma = tf.expand_dims(tf.reshape(self.next_layer.sigmas2_,self.output_shape[1:]),0)
        if(pretraining):                           
            back = tf.zeros_like(back)
            back_sigma = ones_like(back_sigma)
	return tf.assign(self.m_,tf.transpose((forward+back)/(rescalor/back_sigma+tf.reduce_sum(self.p*rescalor*self.sigmas2_patch,[3,4,6])),[3,1,2,0]))
    def update_p(self):
        K = tf.transpose(self.extract_patch(self.input/self.sigmas2)*myexpand(self.m,[3,3,-1])-0.5*myexpand(tf.square(self.m)+self.v2,[3,3,-1])/self.sigmas2_patch,[5,1,2,3,4,6,0]) # (N I J a b K dc) -> (K I J a b dc N)
	return tf.assign(self.p_,mysoftmax(K,axis=[3,4,5]))
    def update_pi(self): return []#tf.assign(self.pi,self.Sp/tf.reduce_sum(self.Sp))
    def update_sigma(self,get_value=False):
        value = self.Sm2+self.Sv2+tf.gradients(self.sigmas2_patch_,self.sigmas2_,tf.transpose(self.SM2v2p-2*self.SmMp,[1,2,3,4,0,5]))[0]
        if(get_value): return value
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
    def __init__(self,input_shape,X,X_mask=None,**kwargs):
	# input_shape is (N I J C) or (N D)	   
	print kwargs 
        self.kwargs_init(kwargs)
        self.input_shape,self.output_shape  = input_shape,input_shape
	self.N                              = X.shape[0]
	# THETAQ PARAMETERS
	self.v2           = tf.Variable(tf.zeros(input_shape))
        self.m            = tf.Variable(tf.zeros(input_shape))
	self.mask         = tf.Variable(tf.zeros(input_shape))
        # THETAQ DATA for all the dataset, as opposed to the variables which have the size of the batch
        # only need for m as v2 is the same across data (n invariant)
        self.mask_data         = X_mask if X_mask is not None else ones_like(X).astype('float32')
	self.mask_placeholder  = tf.placeholder(tf.float32,self.input_shape)
        self.m_data            = X.astype('float32')
        self.m_placeholder     = tf.placeholder(tf.float32,self.input_shape)
        self.v2_data           = self.init_v2_data(X.shape)
        self.v2_placeholder    = tf.placeholder(tf.float32,self.input_shape)
        self.thetaq_assign_op  = tf.group(tf.assign(self.m,self.m_placeholder),tf.assign(self.v2,self.v2_placeholder),tf.assign(self.mask,self.mask_placeholder))
    def set_batch(self,session,indices):
        if(len(indices)==self.N): return
        session.run(self.thetaq_assign_op,feed_dict={self.m_placeholder:self.m_data[indices],self.v2_placeholder:self.v2_data[indices],self.mask_placeholder:self.mask_data[indices]})
    def save_batch(self,session,indices):
        if(len(indices)==self.N): return
        self.m_data[indices] =session.run(self.m)
        self.v2_data[indices]=session.run(self.v2)
    def init_thetaq(self):
	return tf.assign(self.v2,self.mask)
    def KL(self,pretraining=False):
	return float32(0.5)*tf.reduce_sum(tf.log(self.v2+eps)*self.mask)/float32(self.input_shape[0])
    def update_v2(self,pretraining=False):
        a40 = tf.expand_dims(tf.reshape(self.next_layer.sigmas2_,self.input_shape[1:]),0)
        return tf.assign(self.v2,self.mask/a40)
    def update_m(self,opt=0,pretraining=False):
	priorm  = self.next_layer.backward()
        return tf.assign(self.m,priorm*self.mask+self.m*(1-self.mask))
    def evidence(self): 
	return float32(0)



class CategoricalLastLayer(Layer):
    def __init__(self,input_layer,R,Y=None,Y_mask=None,**kwargs):
	self.kwargs_init(kwargs)
	self.R                 = R
	self.N                 = input_layer.N
        self.input_layer       = input_layer
	self.input_shape       = input_layer.output_shape
        input_layer.next_layer = self
        # RESHAPE IF NEEDED (acting as a flatten layer)
        if(len(self.input_shape)>2):
            self.is_flat     = False
            self.D           = prod(self.input_shape[1:])
            self.input_shape = (self.input_shape[0],self.D)#potentially different if flattened
            self.input       = tf.reshape(self.input_layer.m,self.input_shape)
        else:
            self.is_flat     = True
            self.D           = self.input_shape[-1]
            self.input_shape = (self.input_shape[0],self.D)#potentially different if flattened
            self.input       = self.input
        self.mask              = tf.Variable(tf.zeros(input_layer.output_shape[0]))
	# THETA PARAMETERS
        self.sigmas2 = tf.Variable(tf.ones(self.D))
	self.W       = tf.Variable(self.init_W((R,self.D)))
	self.pi      = tf.Variable(tf.ones(R)/R)
        self.b       = tf.Variable(self.init_b(self.D))
	# THETAQ PARAMETERS
        self.p       = tf.Variable(tf.zeros((self.input_shape[0],R))) # (N R)
	self.mask    = tf.Variable(tf.zeros((self.input_shape[0],R))) # (N R)
        # THETAQ DATA for all the dataset, as opposed to the variables which have the size of the batch
        # only need for m as v2 is the same across data (n invariant)
	self.mask_data        = Y_mask if Y_mask is not None else ones_like(Y).astype('float32')
        self.mask_placeholder = tf.placeholder(tf.float32,(self.input_shape[0],R))
        self.p_data           = self.init_p_data((self.N,R)).astype('float32') if Y is None else self.init_p_data((self.N,R)).astype('float32')*self.p_mask+Y*(1-self.p_mask)
        self.p_placeholder    = tf.placeholder(tf.float32,(self.input_shape[0],R))
        self.thetaq_assign_op = tf.group(tf.assign(self.p,self.p_placeholder),tf.assign(self.mask,self.mask_placeholder))
        # STATISTICS
	self.alpha = tf.Variable(tf.zeros(1))
        self.Spm   = tf.Variable(tf.zeros((self.R,self.D)))
        self.Sv2   = tf.Variable(tf.zeros(self.D))
        self.Sm    = tf.Variable(tf.zeros(self.D))
        self.Sp    = tf.Variable(tf.zeros(self.R))
        self.Sm2   = tf.Variable(tf.zeros((self.D)))
	# empty variable for consistancy
	self.k_        = tf.placeholder(tf.int32)
        self.W_indices = asarray([0])
    def set_batch(self,session,indices):
        if(len(indices)==self.N): return
        session.run(self.thetaq_assign_op,feed_dict={self.p_placeholder:self.p_data[indices],self.mask_placeholder:self.mask[indices]})
    def save_batch(self,session,indices):
        if(len(indices)==self.N): return
        self.p[indices]=session.run(self.p)
    def update_S(self):
        Spm = tf.matmul(self.p_,self.input_,transpose_a=True)/self.bs
        if(self.is_flat): Sv2 = tf.reduce_sum(self.input_layer.v2_,1)/self.bs
        else:             Sv2 = tf.reshape(tf.reduce_sum(self.input_layer.v2,0),[self.D_in])/self.bs
        Sp   = tf.reduce_sum(self.p_,0)/self.bs
        Sm2  = tf.reduce_sum(tf.square(self.input_),0)/self.bs
        Sm   = tf.reduce_sum(self.input_,0)/self.bs
        return tf.group(tf.assign(self.Spm,self.alpha*Spm+(1-self.alpha)*self.Spm),tf.assign(self.Sv2,self.alpha*Sv2+(1-self.alpha)*self.Sv2),
                        tf.assign(self.Sp,self.alpha*Sp+(1-self.alpha)*self.Sp),tf.assign(self.Sm2,self.alpha*Sm2+(1-self.alpha)*self.Sm2),
                        tf.assign(self.Sm,self.alpha*Sm+(1-self.alpha)*self.Sm))
    def backward(self,flat=0):
	output = tf.matmul(self.p,self.W)+tf.expand_dims(self.b,0)
	if(flat or self.is_flat):  return output
	else:      return tf.reshape(output,self.input_layer.output_shape)
    def sample(self,K=None,sigma_multiplier=1):
        """ K must be a pre imposed region used for generation
        if not given it is generated according to pi, its shape 
        must be (N R) with a one hot vector on the last dimension"""
        noise = sigma*tf.random_normal(self.input_shape)*tf.expand_dims(tf.sqrt(self.sigmas2),0)
        if(K is None): K = tf.one_hot(tf.multinomial(tf.log(tf.expand_dims(self.pi,0)),self.bs)[0],self.R)
        samples = tf.matmul(K,self.W)+noise+self.b
	if(self.is_flat): return samples
	else:             return tf.reshape(samples,self.input_layer.output_shape)
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
    def likelihood(self,E_step=True):
        if(E_step):
            k1  = -tf.reduce_sum(tf.log(self.sigmas2_))*float32(0.5)-tf.log(2*PI_CONST)*float32(self.D_in*0.5)+tf.reduce_sum(tf.reduce_sum(self.p_,0)*tf.log(self.pi))/self.bs
            if(self.is_flat): k2 = -tf.reduce_sum(tf.reduce_sum(self.input_layer.v2,0)/self.sigmas2_)*0.5/self.bs
            else:             k2 = -tf.reduce_sum(tf.reshape(tf.reduce_sum(self.input_layer.v2,0),[self.D_in])/self.sigmas2_)*0.5/self.bs
            k3  = -tf.reduce_sum(tf.reduce_sum(tf.square(tf.expand_dims(self.input_-self.b,1)-tf.expand_dims(self.W,0))/tf.expand_dims(self.sigmas2,0),2)*self.p_)*0.5/self.bs
            return k1+k2+k3-self.sparsity_prior*tf.reduce_sum(tf.square(self.W))/2
        else:
            k1 = -tf.reduce_sum(tf.log(self.sigmas2_))*float32(0.5)-tf.log(2*PI_CONST)*float32(self.D_in*0.5)+tf.reduce_sum(self.Sp*tf.log(self.pi))
            k2 = tf.reduce_sum(self.Spm*self.W/self.sigmas2)
            k3 = -0.5*tf.reduce_sum((tf.square(self.b_)+self.Sm2+self.Sv2-2*self.Sm*self.b_)/self.sigmas2_)
            k4 = -0.5*tf.reduce_sum(self.Sp*tf.reduce_sum((tf.square(self.W)+2*self.W*self.b)/self.sigmas2,1))
            return k1+k2+k3+k4-self.sparsity_prior*tf.reduce_sum(tf.square(self.W))/2
    def KL(self,pretraining=False): return self.likelihood(0,pretraining)-tf.reduce_sum(self.p_*tf.log(self.p_+eps))/float32(self.bs)
                                        ################### E STEP UPDATES
    def update_p(self,ii=0,pretraining=False):
	proj    = -tf.reduce_sum(tf.square(tf.expand_dims(self.input_-self.b,1)-tf.expand_dims(self.W,0))/tf.expand_dims(self.sigmas2,0),2)*0.5 # ( N R)
        prior   = tf.expand_dims(self.pi,0)
        V       = tf.nn.softmax(proj+tf.log(prior))
        return tf.assign(self.p_,V*tf.expand_dims(self.mask,-1)+self.p_*(1-tf.expand_dims(self.mask,-1)))
        ################### M STEP UPDATES
    def update_sigma(self,get_value=False):
        k1 = tf.square(self.b_)+self.Sm2+self.Sv2+tf.reduce_sum(tf.expand_dims(self.Sp,-1)*tf.square(self.W),0)
        k2 = -2*tf.reduce_sum(self.Spm*self.W,0)+(-self.Sm+tf.reduce_sum(tf.expand_dims(self.Sp,-1)*self.W,0))*self.b_
        value_ = k1+k2
        if(get_value):                  return value_
        if(self.sigma_opt=='local'):    return tf.assign(self.sigmas2_,value_)
        elif(self.sigma_opt=='global'): return tf.assign(self.sigmas2_,tf.fill([self.D_in],tf.reduce_sum(value_)/self.D_in))
        elif(self.sigma_opt=='none'):   return []
    def update_pi(self): 
	return tf.assign(self.pi,self.Sp/tf.reduce_sum(self.Sp))
    def update_Wk(self): 
	return tf.assign(self.W,self.Spm/(tf.expand_dims(self.Sp,-1)+self.sparsity_prior)-self.b_/(1+self.sparsity_prior))
    def update_BV(self):
        if(self.update_b): return tf.assign(self.b_,self.Sm-tf.reduce_sum(self.W*tf.expand_dims(self.Sp,1),0))
        else:              return []



class ContinuousLastLayer(Layer):
    def __init__(self,input_layer,K,sigma_opt='local',sparsity_prior=0,init_W=tf.orthogonal_initializer(1),update_b=True):
        self.alpha             = tf.Variable(float32(0))
        self.update_b = update_b
        self.sigma_opt        = sigma_opt
        self.input_shape       = input_layer.output_shape
        self.bs,self.D_in,self.K=input_layer.bs,prod(self.input_shape[1:]),K
        self.input             = input_layer.m
        self.input_layer       = input_layer
        input_layer.next_layer = self
        self.sparsity_prior    = sparsity_prior
        # MODEL PARAMETERS
        self.sigmas2_= tf.Variable(tf.ones(self.D_in))
        self.sigmas2 = tf.expand_dims(self.sigmas2_,0)
        self.W       = tf.Variable(init_W((K,self.D_in))/sqrt(K+prod(self.input_shape[1:])))
        self.b_      = tf.Variable(tf.zeros(self.D_in))
        self.b       = tf.expand_dims(self.b_,0)
        # VI PARAMETERS
        self.m_      = tf.Variable(tf.zeros((self.bs,K)))# (N K)
        self.v2_     = tf.Variable(tf.zeros((K,)))# (N K)
        # SUFFICIENT STATISTICS
        self.SmM = tf.Variable(tf.zeros((self.D_in,K)))
        self.Sv2 = tf.Variable(tf.zeros(self.D_in))
        self.Sm  = tf.Variable(tf.zeros(self.D_in))
        self.SV2 = tf.Variable(tf.zeros(self.K))
        self.SM  = tf.Variable(tf.zeros(self.K))
        self.Sm2 = tf.Variable(tf.zeros((self.D_in)))
        self.SMM = tf.Variable(tf.zeros((self.K,self.K)))
        self.SM2 = tf.Variable(tf.zeros(self.K))
        # BN PARAMETERS
        self.mu_ = tf.Variable(tf.zeros(self.D_in))
        self.mu  = tf.expand_dims(self.mu_,0)
        self.gamma_ = tf.Variable(tf.ones(self.D_in))
        self.gamma  = tf.expand_dims(self.gamma_,0)
        #
        self.k_  = tf.placeholder(tf.int32)
        if(len(self.input_shape)>2):
            self.is_flat = False
            self.input_  = tf.reshape(self.input,(self.bs,self.D_in))
        else:
            self.is_flat = True
            self.input_  = self.input
        self.W_indices = asarray(range(self.K))
    def update_BN(self):
        return []
        mu = self.alpha*tf.reduce_sum(self.input_,0)/self.bs+(1-self.alpha)*self.mu_
        return tf.group(tf.assign(self.mu_,mu),tf.assign(self.gamma_,self.alpha*tf.sqrt(tf.reduce_sum(tf.square(self.input_-mu),0)/self.bs)+(1-self.alpha)*self.gamma_))
    def update_S(self):
        SmM   = tf.matmul(self.input_/self.bs,self.m_,transpose_a=True)#(n,Din),(n,k) -> (Din,k)
        if(self.is_flat): Sv2 = tf.reduce_sum(self.input_layer.v2_/self.bs,1)
        else:             Sv2 = tf.reshape(tf.reduce_sum(self.input_layer.v2/self.bs,0),[self.D_in])
        Sm   = tf.reduce_sum(self.input_/self.bs,0)
        SV2  = self.v2_ 
        SM   = tf.reduce_sum(self.m_/self.bs,0)
        Sm2  = tf.reduce_sum(tf.square(self.input_)/self.bs,0)
        SMM  = tf.matmul(self.m_,self.m_/self.bs,transpose_a=True)#(n,k)x(n,k)->(k,k)
        SM2  = tf.reduce_sum(tf.square(self.m_),0)/self.bs
        return tf.group(tf.assign(self.SmM,self.alpha*SmM+(1-self.alpha)*self.SmM),tf.assign(self.Sv2,self.alpha*Sv2+(1-self.alpha)*self.Sv2),
                        tf.assign(self.Sm,self.alpha*Sm+(1-self.alpha)*self.Sm),tf.assign(self.SV2,self.alpha*SV2+(1-self.alpha)*self.SV2),
                        tf.assign(self.SM,self.alpha*SM+(1-self.alpha)*self.SM),tf.assign(self.Sm2,self.alpha*Sm2+(1-self.alpha)*self.Sm2),
                        tf.assign(self.SMM,self.alpha*SMM+(1-self.alpha)*self.SMM),tf.assign(self.SM2,self.alpha*SM2+(1-self.alpha)*self.SM2))
    def vector2tensor(self,u):
        return tf.reshape(u,self.input_shape)
    def backward(self,flat=1,resi=None): 
        value = tf.matmul(self.m_,self.W) #->(n,Din)
        if(flat): return value
        else:     return tf.reshape(value,self.input_shape)
    def sample(self,M=None,K=None,sigma=1,deterministic=False):
        if(deterministic==True): return self.backward(self.is_flat)
        return tf.reshape((tf.matmul(tf.random_normal((self.bs,self.K)),self.W)+self.b+sigma*tf.random_normal((self.bs,self.D_in))*tf.sqrt(self.sigmas2_))*self.gamma+self.mu,self.input_shape)
    def likelihood(self,batch=False,pretraining=False):
        if(batch==False):
            k1  = -tf.reduce_sum(tf.log(self.sigmas2_))*float32(0.5)-tf.log(2*PI_CONST)*float32(self.D_in*0.5)     #(1)
            k2  = -tf.reduce_sum((tf.reshape(self.input_layer.v2,[self.bs,-1]))/(tf.square(self.gamma)*self.bs*self.sigmas2))*0.5
            k3  = -tf.reduce_sum(tf.square(((self.input_-self.mu)/self.gamma)-self.b_-tf.matmul(self.m_,self.W))/self.sigmas2)*0.5/self.bs
            k4  = -tf.reduce_sum(tf.expand_dims(self.v2_,1)*tf.square(self.W)/self.sigmas2)*float32(0.5)            #(1)
            EXTRA = -tf.reduce_sum(tf.square(self.m_)/self.bs)*0.5-tf.reduce_sum(self.v2_)*0.5
            return k1+k2+k3+k4+EXTRA-0.5*tf.reduce_sum(tf.square(self.W))*self.sparsity_prior
        else:
            k1  = -tf.reduce_sum(tf.log(self.sigmas2_))*float32(0.5)-tf.log(2*PI_CONST)*float32(self.D_in*0.5)     #(1)
            k2  = -tf.reduce_sum((self.Sv2/tf.square(self.gamma)+self.Sm2/tf.square(self.gamma)+tf.square(self.mu_)/tf.square(self.gamma_)-2*self.Sm*self.mu_/tf.square(self.gamma_)+tf.square(self.b_)-2*(self.Sm-self.mu_)/self.gamma_*self.b_)/self.sigmas2_)*0.5
            k3  = -tf.reduce_sum(tf.matmul(self.W,self.W/self.sigmas2,transpose_b=True)*self.SMM)*0.5
            k4  = -tf.reduce_sum(tf.reduce_sum(self.W*self.b/self.sigmas2,1)*self.SM)
            k5  = tf.reduce_sum(tf.transpose(self.SmM)/self.gamma*self.W/self.sigmas2)-tf.reduce_sum(self.mu_/self.gamma_*tf.reduce_sum(self.W*tf.expand_dims(self.SM,1),0)/self.sigmas2_)-tf.reduce_sum(tf.expand_dims(self.SV2,-1)*tf.square(self.W)/tf.expand_dims(self.sigmas2_,0))*float32(0.5)
            EXTRA = -0.5*tf.reduce_sum(self.SM2+self.SV2)
            return k1+k2+k3+k4+k5+EXTRA-0.5*tf.reduce_sum(tf.square(self.W))*self.sparsity_prior
    def update_Wk(self):
        numerator   = -self.b_*self.SM[self.k_]+self.SmM[:,self.k_]/self.gamma_-self.mu_/self.gamma_*self.SM[self.k_]-tf.reduce_sum(self.W*tf.expand_dims(self.SMM[self.k_]*(1-tf.one_hot(self.k_,self.K)),-1),0)
        denominator = self.SV2[self.k_]+self.SMM[self.k_,self.k_]
        return tf.scatter_update(self.W,[self.k_],[numerator/(denominator+self.sparsity_prior)])
    def update_b(self):
        if(self.update_b): return tf.assign(self.b_,(self.Sm-self.mu)/self.gamma-tf.einsum('kd,k->',self.W,self.M))
        else: return []
    def update_sigma(self,get_value=False):
        k2 = (self.Sv2+self.Sm2+tf.square(self.mu)-2*self.Sm*self.mu)/tf.square(self.gamma)+tf.square(self.b_)+tf.reduce_sum(tf.matmul(self.SMM,self.W)*self.W,0)
        k3 = self.b_*tf.reduce_sum(self.W*tf.expand_dims(self.SM,-1),0)-tf.reduce_sum(self.SmM*tf.transpose(self.W/self.gamma),1)+tf.reduce_sum(self.mu_/self.gamma_*tf.reduce_sum(self.W*tf.expand_dims(self.SM,1),0))-self.Sm*self.b_/self.gamma_
        k4 = tf.reduce_sum(tf.expand_dims(self.SV2,-1)*tf.square(self.W),0)
        value = k2+k3*2+k4
        if(get_value): return value
        if(self.sigma_opt=='local'):    return tf.assign(self.sigmas2_,value)
        elif(self.sigma_opt=='global'): return tf.assign(self.sigmas2_,tf.ones_like(self.sigmas2_)*tf.reduce_sum(value)/(prod(self.input_shape[1:])))
    def evidence(self,b=0):
        return 0.
    def update_v2(self,b=0):
        rescalor = tf.reduce_min(self.sigmas2_)
        return tf.assign(self.v2_,rescalor/(rescalor+tf.einsum('kd,d->k',tf.square(self.W),rescalor/self.sigmas2_)))
    def update_m(self,b=0,bb=0):
        b = tf.matmul(((self.input_-self.mu)/self.gamma-self.b)/self.sigmas2,self.W,transpose_b=True)
        A = tf.matmul(self.W,self.W/self.sigmas2,transpose_b=True)+tf.eye(self.K)
        with tf.device('/device:CPU:0'):
            m = tf.transpose(tf.matrix_solve(A,tf.transpose(b)))
        return tf.assign(self.m_,m)
    def KL(self,pretraining=False): return self.likelihood(0)+float32(0.5)*tf.reduce_sum(tf.log(self.v2_+eps))#*self.mask)/float32(self.input_shape[0])



