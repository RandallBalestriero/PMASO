from pylab import *
import tensorflow as tf
from sklearn.datasets import make_moons
eps = 0.000000000000000001
from sklearn.datasets import load_digits

tf.get_collection('latent')
tf.get_collection('params')


def compute_WpmWpm(layers,l):
	Wmp=tf.reduce_sum(tf.expand_dims(layers[l].W,0)*tf.expand_dims(layers[l].m*layers[l].p,-1),axis=2)# (N,K,D)
	masked = tf.expand_dims(Wmp,1)*(1-tf.reshape(tf.eye(layers[l].D),(1,layers[l].D,layers[l].D,1)))#(N,K,K,D)
	return tf.reduce_sum(tf.expand_dims(Wmp,1)*masked)


class DenseLayer:
	def __init__(self,batch_size,n_in,n_out,r):
		self.n_in = n_in
		self.batch_size = batch_size
		self.D       = n_out
		self.R       = r
		self.W       = tf.Variable(tf.random_uniform((n_out,r,n_in),-0.1,0.1))
		self.pi_      = tf.Variable(tf.fill([n_out,r],1.0/r))
		self.pi      = tf.nn.softmax(self.pi_)+eps
		self.sigmas2_ = tf.Variable(tf.ones(1)*0.32-4)
		self.sigmas2 = tf.nn.softplus(self.sigmas2_)+eps
		self.m       = tf.Variable(tf.random_uniform((batch_size,n_out,r)))
		self.p_       = tf.Variable(tf.random_uniform((batch_size,n_out,r)))
		self.p       = tf.nn.softmax(self.p_)
		self.v2_     = tf.Variable(tf.random_uniform((1,n_out,r)))
		self.v2      = tf.nn.softplus(self.v2_)
		self.M       = tf.reduce_sum(self.m*self.p,axis=2)
                tf.add_to_collection('latent',self.p_)
                tf.add_to_collection('latent',self.m)
                tf.add_to_collection('latent',self.v2_)
                tf.add_to_collection('params',self.pi_)
                tf.add_to_collection('params',self.W)
                tf.add_to_collection('params',self.sigmas2_)
	def forward(self,x):
		return tf.reduce_sum(tf.tensordot(x,self.W,[[1],[2]])*self.p*self.m,axis=2)
	def backward(self,e):
		return tf.tensordot(tf.expand_dims(e,-1)*self.p*self.m,self.W,[[1,2],[0,1]])
	def sample(self,i):
		noise = tf.random_normal((self.batch_size,self.n_in))*tf.sqrt(self.sigmas2[0])
		K = tf.one_hot(tf.transpose(tf.multinomial(self.pi,self.batch_size)),self.R)
		return tf.tensordot(tf.expand_dims(i,-1)*K,self.W,[[1,2],[0,1]])+noise

class InputLayer:
        def __init__(self,batch_size,n_in):
                self.batch_size = batch_size
                self.D       = n_in
                self.m       = tf.Variable(tf.random_normal((batch_size,n_in)))
		self.M       = self.m


class UnsupFinalLayer:
        def __init__(self,batch_size,n_in,r):
                self.n_in = n_in
                self.batch_size = batch_size
                self.D       = 1
                self.R       = r
                self.W       = tf.Variable(tf.random_uniform((1,r,n_in),-1,1))
                self.pi_     = tf.Variable(tf.fill([1,r],1.0/r))
		self.pi      = tf.nn.softmax(self.pi_)
                self.sigmas2_= tf.Variable(tf.ones(1)*0.32-6)
		self.sigmas2 = tf.nn.softplus(self.sigmas2_)+eps
                self.p_      = tf.Variable(tf.random_normal((batch_size,1,r)))
		self.p       = tf.nn.softmax(self.p_)
		tf.add_to_collection('latent',self.p_)
		tf.add_to_collection('params',self.pi_)
                tf.add_to_collection('params',self.W)
                tf.add_to_collection('params',self.sigmas2_)
        def forward(self,x):
                return tf.reduce_sum(tf.tensordot(x,self.W,[[1],[2]])*self.p,axis=2)
        def backward(self,e=0):
                return tf.tensordot(self.p,self.W,[[1,2],[0,1]])
        def sample(self,i=0):
                noise = tf.random_normal((self.batch_size,self.n_in))*tf.sqrt(self.sigmas2[0])
                K = tf.one_hot(tf.transpose(tf.multinomial(self.pi,self.batch_size)),self.R)
                return tf.tensordot(K,self.W,[[1,2],[0,1]])+noise





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
	return new_p,new_m


def sample(layers):
	s=0
        for i in xrange(1,len(layers)):
		s = layers[-i].sample(s)
	return s
		
	
		


def likelihood(layers):
	# FIRST LAYER
	a1=-layers[0].batch_size*layers[0].D*(tf.log(layers[1].sigmas2+eps)/2+tf.log(2*3.14159)/2)+tf.reduce_sum(layers[1].p*tf.expand_dims(tf.log(layers[1].pi+eps),0))
	a2=-tf.reduce_sum(tf.pow(layers[0].m,2))/(2*layers[1].sigmas2[0])
	a3=tf.reduce_sum(layers[0].m*layers[1].backward(float32(1)))/layers[1].sigmas2[0]
	a4=-tf.reduce_sum(tf.reduce_sum(layers[1].W*layers[1].W,axis=2)*tf.reduce_sum(layers[1].p*(tf.pow(layers[1].m,2)+layers[1].v2),axis=0))/(2*layers[1].sigmas2)
	a51= tf.reduce_sum(tf.expand_dims(layers[1].W,0)*tf.expand_dims(layers[1].p*layers[1].m,-1),2)
	a5=0#-compute_WpmWpm(layers,1)/(2*layers[1].sigmas2[0])
	like= a1+a2+a3+a4+a5
	for l in xrange(2,len(layers)-1):
	        a1=-layers[0].batch_size*layers[l-1].D*tf.log(layers[l].sigmas2+eps)/2+tf.reduce_sum(layers[l].p*tf.expand_dims(tf.log(layers[l].pi),0))
	        a2=-tf.reduce_sum((tf.pow(layers[l-1].m,2)+layers[l-1].v2)*layers[l-1].p)/(2*layers[l].sigmas2[0])
	        a3=tf.reduce_sum(layers[l-1].M*layers[l].backward(float32(1)))/layers[l].sigmas2[0]
	        a4=-tf.reduce_sum(tf.reduce_sum(layers[l].W*layers[l].W,axis=2)*tf.reduce_sum(layers[l].p*(tf.pow(layers[l].m,2)+layers[l].v2),axis=0))/(2*layers[l].sigmas2)
		a5= -compute_WpmWpm(layers,l)/(2*layers[l].sigmas2[0])
	        like+= a1+a2+a3+a4+a5
	# LAST LAYER
#	like=0
        a1=-layers[0].batch_size*layers[-2].D*(tf.log(layers[-2].sigmas2+eps)/2+tf.log(2*3.14159)/2)+tf.reduce_sum(layers[-1].p*tf.expand_dims(tf.log(layers[-1].pi),0))
        a2=-tf.reduce_sum((tf.pow(layers[-2].m,2)+layers[-2].v2)*layers[-2].p)/(2*layers[-1].sigmas2[0])
        a3=tf.reduce_sum(layers[-2].M*layers[-1].backward(float32(1)))/layers[-1].sigmas2[0]
        a4=-tf.reduce_sum(tf.reduce_sum(layers[-1].W*layers[-1].W,axis=2)*tf.reduce_sum(layers[-1].p,axis=0))/(2*layers[-1].sigmas2)
        like+= a1+a2+a3+a4
	return like
	

def KL(layers):
        v11 = tf.reduce_sum(layers[1].p*(tf.log(layers[1].p+eps)-tf.log(layers[1].v2+eps)/2))
	v12 = tf.reduce_sum(layers[-1].p*(tf.log(layers[-1].p+eps)))
	return -likelihood(layers)+v11+v12







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



XX = load_digits(3)['data']#make_moons(batch_size,noise=0.051)


batch_size = XX.shape[0]

layers = [InputLayer(batch_size,64),DenseLayer(batch_size,64,10,1),UnsupFinalLayer(batch_size,10,3)]

x       = tf.placeholder(tf.float32,shape=[batch_size,layers[0].D])



opti = tf.train.AdamOptimizer(0.02)
train_op1 = opti.minimize(KL(layers),var_list=tf.get_collection('latent'))
train_op2 = opti.minimize(-likelihood(layers),var_list=tf.get_collection('params'))



session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)




XX-=XX.mean(1,keepdims=True)
XX/=XX.max(1,keepdims=True)


session.run(init_latent(x,layers),feed_dict={x:XX})



U=session.run(sample(layers))
figure()
for i in xrange(25):
    subplot(5,5,1+i)
    imshow(U[i].reshape((8,8)),aspect='auto')
    colorbar()


for k in xrange(4):
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






