from pylab import *
import tensorflow as tf
from sklearn.datasets import make_moons
eps = 0.000000000000000001
from sklearn.datasets import load_digits

tf.get_collection('latent')
tf.get_collection('params')


class DenseLayer:
	def __init__(self,input_shape,K,R):
		#INPUT SHAPE (batch_size,C)
		self.input_shape = input_shape
		if(len(input_shape)>2): self.is_flat = False
		else: self.is_flat = True
                self.C       = int32(prod(input_shape[1:]))
		self.D       = self.C
		self.bs      = int32(input_shape[0])
                self.output_shape = [self.bs,K]
		self.K       = K
		self.R       = R
		self.W       = tf.Variable(tf.random_uniform((K,R,self.C),-0.1,0.1))
		self.pi_     = tf.Variable(tf.random_uniform((K,R),-0.1,0.1))
		self.pi      = tf.nn.softmax(self.pi_)+eps
		self.sigmas2_= tf.Variable(tf.ones(1))
		self.sigmas2 = tf.pow(self.sigmas2_,2)+eps
		self.m       = tf.Variable(tf.random_uniform((self.bs,self.K,R),-0.1,0.1))
		self.p_      = tf.Variable(tf.random_uniform((self.bs,self.K,R),-0.1,0.1))
		self.p       = tf.nn.softmax(self.p_)
#		self.v2_     = tf.Variable(tf.random_uniform((1,n_out,r)))
#		self.v2      = tf.pow(self.v2_,2)#random_uniform((batch_size,n_out,r)))#tf.stop_gradient()+eps
		self.M       = tf.reduce_sum(self.m*self.p,axis=2)
                tf.add_to_collection('latent',self.p_)
                tf.add_to_collection('latent',self.m)
#                tf.add_to_collection('latent',self.v2_)
                tf.add_to_collection('params',self.pi_)
                tf.add_to_collection('params',self.W)
                tf.add_to_collection('params',self.sigmas2_)
#	def forward(self,x):
#		return tf.reduce_sum(tf.tensordot(x,self.W,[[1],[2]])*self.p*self.m,axis=2)
	def backward(self):
#		mask = tf.concat([tf.ones((self.C,1,self.n_in)),tf.zeros((self.D,1,self.n_in))],axis=1)
		if(self.is_flat):
                        return tf.tensordot(self.p*self.m,self.W,[[1,2],[0,1]])
		else:
			return tf.reshape(tf.tensordot(self.p*self.m,self.W,[[1,2],[0,1]]),self.input_shape)
	def sample(self,M):
		noise = tf.random_normal((self.bs,self.C))*tf.sqrt(self.sigmas2[0]) 
		K = tf.one_hot(tf.transpose(tf.multinomial(self.pi,self.bs)),self.R)
		if(self.is_flat):
			return tf.tensordot(tf.expand_dims(M,-1)*K,self.W,[[1,2],[0,1]])+noise
		else:
                        return tf.reshape(tf.tensordot(tf.expand_dims(M,-1)*K,self.W,[[1,2],[0,1]])+noise,self.input_shape)





class ConvLayer:
	def __init__(self,input_shape,I,J,K,R,stride=1):
		#INPUT_SHAPE : (batch_size,input_filter,M,N)
		M = input_shape[-2]
		N = input_shape[-1]
		self.output_shape = (input_shape[0],K,input_shape[-2],input_shape[-1])
		self.C       = input_shape[1]
		self.D       = prod(input_shape[1:])
		self.R       = R
		self.bs = input_shape[0]
		self.s       = stride
		self.K       = K
		self.W       = tf.Variable(tf.random_uniform((R,I,J,self.C,K),-0.1,0.1))
		self.pi_     = tf.Variable(tf.random_uniform((K,R),-0.1,0.1))
		self.pi      = tf.nn.softmax(self.pi_)+eps
		self.sigmas2_= tf.Variable(tf.ones(1))
		self.sigmas2 = tf.pow(self.sigmas2_,2)+eps
		self.m       = tf.Variable(tf.random_uniform((self.bs,K,R,M,N),-0.1,0.1))
		self.p_      = tf.Variable(tf.random_uniform((self.bs,K,R,M,N),-0.1,0.1))
		self.p       = tf.nn.softmax(self.p_,dim=2)
#		self.v2_     = tf.Variable(tf.random_uniform((1,n_out,r)))
#		self.v2      = tf.pow(self.v2_,2)#random_uniform((batch_size,n_out,r)))#tf.stop_gradient()+eps
		self.M       = tf.reduce_sum(self.m*self.p,axis=2)
                tf.add_to_collection('latent',self.p_)
                tf.add_to_collection('latent',self.m)
#                tf.add_to_collection('latent',self.v2_)
                tf.add_to_collection('params',self.pi_)
                tf.add_to_collection('params',self.W)
                tf.add_to_collection('params',self.sigmas2_)
#	def forward(self,x):
#		return tf.nn.conv3d(tf.expand_dims(x,1),self.W,self.stride,"SAME")#tf.reduce_sum(tf.tensordot(x,self.W,[[1],[2]])*self.p*self.m,axis=2)
	def backward(self):
		deconv = tf.nn.conv3d_transpose(self.m*self.p,self.W,padding="SAME",strides=[1,1,1,self.s,self.s],data_format="NCDHW")#BS,input_filter,R,N,M
		return tf.reduce_sum(deconv,axis=2)
	def sample(self,M):
		#multinomial returns [K,n_samples] with integer value 0,...,R-1
		noise = tf.random_normal(self.output_shape)*tf.sqrt(self.sigmas2[0])
		mp    = tf.one_hot(tf.reshape(tf.multinomial(self.pi,self.batch_size*self.M*self.N),(self.batch_size,self.K,self.M,self.N)),self.R)#(self.batch_size,self.K,self.M,self.N,self.R)
		deconv = tf.nn.conv3d_transpose(mp,self.W,padding="SAME",strides=[1,1,1,self.s,self.s],data_format="NCDHW")#BS,input_filter,R,N,M
                return tf.reduce_sum(deconv,axis=2)


		





class InputLayer:
        def __init__(self,input_shape):
		self.input_shape = input_shape
		self.output_shape = input_shape
                self.m       = tf.Variable(tf.random_normal(self.input_shape))
		self.M       = self.m


class UnsupFinalLayer:
        def __init__(self,input_shape,R):
                self.input_shape = input_shape
                if(len(input_shape)>2): self.is_flat = False
                else: self.is_flat = True
                self.C       = input_shape[1]
		self.D       = self.C
                self.bs      = input_shape[0]
                self.R       = R
                self.W       = tf.Variable(tf.random_uniform((1,R,self.C),-0.1,0.1))
                self.pi_     = tf.Variable(tf.random_uniform((1,R),-0.1,0.1))
		self.pi      = tf.nn.softmax(self.pi_)
                self.sigmas2_= tf.Variable(tf.ones(1))
		self.sigmas2 = tf.pow(self.sigmas2_,2)+eps
                self.p_      = tf.Variable(tf.random_uniform((self.bs,1,R),-0.1,0.1))
		self.p       = tf.nn.softmax(self.p_)
		tf.add_to_collection('latent',self.p_)
		tf.add_to_collection('params',self.pi_)
                tf.add_to_collection('params',self.W)
                tf.add_to_collection('params',self.sigmas2_)
#        def forward(self,x):
#                return tf.reduce_sum(tf.tensordot(x,self.W,[[1],[2]])*self.p,axis=2)
        def backward(self,e=0):
		if(self.is_flat):
	                return tf.tensordot(self.p,self.W,[[1,2],[0,1]])
		else:
                        return tf.reshape(tf.tensordot(self.p,self.W,[[1,2],[0,1]]),self.input_shape)
        def sample(self,i=0):
                noise = tf.random_normal((self.bs,self.D))*tf.sqrt(self.sigmas2[0])
                K = tf.one_hot(tf.transpose(tf.multinomial(self.pi,self.bs)),self.R)
		if(self.is_flat):
	                return tf.tensordot(K,self.W,[[1,2],[0,1]])+noise
		else:
                        return tf.reshape(tf.tensordot(K,self.W,[[1,2],[0,1]])+noise,self.input_shape)





def init_latent(x,layers):
	new_p = []
	new_m = []
	new_v = []
        M     = []
	for i in xrange(len(layers)):
		if(isinstance(layers[i],InputLayer)):
                        new_m.append(tf.assign(layers[i].m,x))
#		if(isinstance(layers[i],DenseLayer)):
#			new_m.append(tf.assign(layers[i].m,tf.zeros_like(layers[i].m)))
#			layers[i].v2=tf.stop_gradient(1/tf.expand_dims((tf.reduce_sum(layers[i].W*layers[i].W,axis=2)/layers[i].sigmas2+1/layers[i+1].sigmas2),0))
	return new_p,new_m#,new_v



def init_v2(layers):
        for i in xrange(len(layers)):
                if(isinstance(layers[i],DenseLayer)):
                        layers[i].v2=tf.stop_gradient(1/tf.expand_dims((tf.reduce_sum(layers[i].W*layers[i].W,axis=2)/layers[i].sigmas2[0]+1/layers[i+1].sigmas2[0]),0))
		elif(isinstance(layers[i],ConvLayer)):
			value = tf.reshape(tf.transpose(tf.reduce_sum(layers[i].W*layers[i].W,axis=[1,2,3])),(1,layers[i].K,layers[i].R,1,1))
                        layers[i].v2=tf.stop_gradient(1/(value/layers[i].sigmas2[0]+1/layers[i+1].sigmas2[0]))





def sample(layers):
	s=float32(1)
        for i in xrange(1,len(layers)):
		s = layers[-i].sample(s)
	return s



#def sample(layers):
#        s=float32(1)
#        return layers[1].backward(s)


def SSE(x,y):
	return tf.reduce_sum(tf.pow(x-y,2))
	
		


def likelihood(layers):
	# FIRST LAYER
	like=0# a1+a2+a3+a4+a5
	for l in xrange(1,len(layers)-1):
	        a1 = -SSE(layers[l-1].M,layers[l].backward())/(2*layers[l].sigmas2[0])
		if(isinstance(layers[l],DenseLayer)):
	                k  = layers[l].bs*layers[l].D*(tf.log(layers[l].sigmas2[0]+eps)/2+tf.log(2*3.14159)/2)+tf.reduce_sum(layers[l].p*tf.expand_dims(tf.log(layers[l].pi+eps),0))
			a2 = tf.reduce_sum(layers[l].W*layers[l].W,axis=2)/(2*layers[l].sigmas2)+1/(2*layers[l+1].sigmas2)
		else:
	                k  = layers[l].bs*layers[l].D*(tf.log(layers[l].sigmas2[0]+eps)/2+tf.log(2*3.14159)/2)+tf.reduce_sum(layers[l].p*tf.reshape(tf.log(layers[l].pi+eps),(layers[l].bs,layers[l].K,layers[l].R,1,1)))
			a2 = tf.reshape(tf.transpose(tf.reduce_sum(layers[l].W*layers[l].W,axis=[1,2,3])),(1,layers[l].K,layers[l].R,1,1))
		a3 = tf.reduce_sum(a2*tf.reduce_sum((tf.pow(layers[l].m,2))*(tf.pow(layers[l].p,2)-layers[l].p),axis=0)/(2*layers[l].sigmas2[0]))
		like+=a1+a3+k#-tf.reduce_sum(layers[l].p)##tf.reduce_sum(tf.expand_dims(a2,0)*layers[l].p*layers[l].v2)
	l+=1
	# LAST LAYER
        k  = layers[l].bs*layers[l].D*(tf.log(layers[l].sigmas2+eps)/2+tf.log(2*3.14159)/2)+tf.reduce_sum(layers[l].p*tf.expand_dims(tf.log(layers[l].pi+eps),0))
        a1 = -tf.reduce_sum(tf.reduce_sum(tf.pow(tf.expand_dims(layers[l-1].M,1)-tf.expand_dims(layers[l].W[0],0),2),axis=2)*layers[-1].p[:,0,:])/(2*layers[l].sigmas2)
#        a2 = tf.reduce_sum(layers[l].W*layers[l].W,axis=2)/(2*layers[l].sigmas2)
#        a3 = tf.reduce_sum(a2*tf.reduce_sum(tf.pow(layers[l].p,2)-layers[l].p,axis=0)/(2*layers[l].sigmas2))
        like+=a1+k
	return like
	

def KL(layers):
	kl = 0
        for l in xrange(1,len(layers)-1):
	        kl += tf.reduce_sum(layers[l].p*(tf.log(layers[l].p+eps)-tf.log(layers[l].v2+eps)/2))
	kl += tf.reduce_sum(layers[-1].p*(tf.log(layers[-1].p+eps)))
	return likelihood(layers)-kl







def get_p(layers):
	p=[]
	for l in layers[1:]:
		p.append(l.p)
	return p

def get_m(layers):
        p=[]
        for l in layers[1:]:
                p.append(l.M)
        return p

def get_v(layers):
        p=[]
        for l in layers[1:]:
                p.append(l.v2)
        return p



XX = load_digits(10)['data']#make_moons(batch_size,noise=0.051)

XX = XX.reshape((XX.shape[0],1,8,8))

input_shape = XX.shape

layers = [InputLayer(input_shape)]
layers.append(ConvLayer(layers[-1].output_shape,K=10,I=3,J=3,R=2))
layers.append(UnsupFinalLayer(layers[-1].output_shape,10))

init_v2(layers)

x       = tf.placeholder(tf.float32,shape=layers[0].input_shape)

opti = tf.train.AdamOptimizer(0.810005)
train_op1 = opti.minimize(-KL(layers),var_list=tf.get_collection('latent'))
train_op2 = opti.minimize(-likelihood(layers),var_list=tf.get_collection('params'))

session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)




XX-=XX.mean(1,keepdims=True)
XX/=XX.max(1,keepdims=True)/2


session.run(init_latent(x,layers),feed_dict={x:XX})



U=session.run(sample(layers))
figure()
for i in xrange(25):
    subplot(5,5,1+i)
    imshow(U[i].reshape((8,8)),aspect='auto')
    colorbar()


for k in xrange(6):
	print tf.get_collection('latent')
	for i in xrange(16):
		session.run(train_op1)
		print "KL",session.run(KL(layers))
	for i in xrange(16):
	        session.run(train_op2)
	        print session.run(likelihood(layers))


U=session.run(sample(layers))
figure()
for i in xrange(25):
    subplot(5,5,1+i)
    imshow(U[i].reshape((8,8)),aspect='auto')
    colorbar()

show()






