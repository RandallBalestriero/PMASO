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


def compute_lmek(layers,l):
        Wmp=tf.reduce_sum(tf.expand_dims(layers[l].W,0)*tf.expand_dims(layers[l].m*layers[l].p,-1),axis=2)# (N,K,D)
        masked = tf.expand_dims(Wmp,1)*(1-tf.reshape(tf.eye(layers[l].D),(1,layers[l].D,layers[l].D,1)))#(N,K,K,D)
	return tf.reduce_sum(masked,axis=1)






class DenseLayer:
	def __init__(self,batch_size,n_in,n_out,r):
		self.n_in = n_in
		self.batch_size = batch_size
		self.D       = n_out
		self.R       = r
		self.W       = tf.Variable(tf.random_uniform((n_out,r,n_in),-0.7,0.7))
		self.pi      = tf.Variable(tf.fill([n_out,r],1.0/r))
#		self.pi      = tf.nn.softmax(tf.clip_by_value(self.pi_,-5,5))+eps
		self.sigmas2 = tf.Variable(tf.ones(1))
#		self.sigmas2 = tf.nn.softplus(tf.clip_by_value(self.sigmas2_,-8,2))+eps
		self.m       = tf.Variable(tf.random_uniform((batch_size,n_out,r)))
		self.p       = tf.Variable(tf.random_uniform((batch_size,n_out,r)))
#		self.p       = tf.nn.softmax(tf.clip_by_value(self.p_,-1,1))
		self.v2     = tf.Variable(tf.random_uniform((1,n_out,r)))
#		self.v2      = tf.nn.softplus(tf.clip_by_value(self.v2_,-8,2))
		self.M       = tf.reduce_sum(self.m*self.p,axis=2)
#                tf.add_to_collection('latent',self.p_)
#                tf.add_to_collection('latent',self.m)
#                tf.add_to_collection('latent',self.v2_)
#                tf.add_to_collection('params',self.pi_)
#                tf.add_to_collection('params',self.W)
#                tf.add_to_collection('params',self.sigmas2_)
	def forward(self,x):
		return tf.reduce_sum(tf.tensordot(x,self.W,[[1],[2]])*self.p*self.m,axis=2)
	def backward(self,e):
		return tf.tensordot(tf.expand_dims(e,-1)*self.p*self.m,self.W,[[1,2],[0,1]])
	def sample(self,i):
		noise = tf.random_normal((self.batch_size,self.n_in))*0.0001#tf.sqrt(self.sigmas2[0])
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
                self.pi     = tf.Variable(tf.fill([1,r],1.0/r))
#		self.pi      = tf.nn.softmax(self.pi_)
                self.sigmas2= tf.Variable(tf.ones(1))
#		self.sigmas2 = tf.nn.softplus(self.sigmas2_)+eps
                self.p      = tf.Variable(tf.random_normal((batch_size,1,r)))
#		self.p       = tf.nn.softmax(self.p_)
#		tf.add_to_collection('latent',self.p_)
#		tf.add_to_collection('params',self.pi_)
# #               tf.add_to_collection('params',self.W)
#                tf.add_to_collection('params',self.sigmas2_)
        def forward(self,x):
                return tf.reduce_sum(tf.tensordot(x,self.W,[[1],[2]])*self.p,axis=2)
        def backward(self,e=0):
                return tf.tensordot(self.p,self.W,[[1,2],[0,1]])
        def sample(self,i=0):
                noise = tf.random_normal((self.batch_size,self.n_in))*0.0001#tf.sqrt(self.sigmas2[0])
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
		if(isinstance(layers[i],DenseLayer)):
			new_m.append(tf.assign(layers[i].m,tf.zeros_like(layers[i].m)))
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
	a5=-compute_WpmWpm(layers,1)/(2*layers[1].sigmas2[0])
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






def update_m(layers,l,k):
       if(isinstance(layers[l],DenseLayer)):
                scaling = tf.reduce_sum(layers[l].W[k]*layers[l].W[k],axis=1)
                if(l<len(layers)-1):
                        scaling  += layers[l].sigmas2[0]/layers[l+1].sigmas2[0]
                        prior     = layers[l+1].backward(tf.constant(1,dtype=tf.float32,shape=(1,layers[l+1].D)))[:,k]
                else:
                        prior     = tf.ones(layers[l].batch_size)#tf.constant(0,dtype=tf.float32,shape=(layers[l].batch_size))
                rec_error = layers[l-1].M-layers[l].backward(1-tf.one_hot(k,layers[l].D))
                value     = (1/scaling)*(tf.reduce_sum(tf.expand_dims(layers[l].W[k],0)*tf.expand_dims(rec_error,1),axis=2)+tf.expand_dims(prior,-1))
                indices   = tf.transpose(tf.stack([tf.range(layers[0].batch_size),tf.fill([layers[0].batch_size],k)]))
                m         = tf.scatter_nd_update(layers[l].m,indices,value)
		return m



def approx_m(layers,l):
       if(isinstance(layers[l],DenseLayer)):
                scaling = tf.reduce_sum(layers[l].W*layers[l].W,axis=2)
		ratio     = layers[l].sigmas2[0]/layers[l+1].sigmas2[0]
                scaling  += ratio
                prior     = layers[l+1].backward(float32(1))*ratio#N K
#		A  = tf.expand_dims(layers[l].W,0)*tf.expand_dims(layers[l].m*layers[l].p,-1)#(N,K,R,D)
#		Wp = tf.expand_dims(tf.reduce_sum(A,2),1)*(1-tf.reshape(tf.eye(layers[l].D),(1,layers[l].D,layers[l].D,1)))#(N,K,K,D)
		retro     = 0#tf.reduce_sum(tf.expand_dims(compute_lmek(layers,l),2)*tf.expand_dims(layers[l].W,0),axis=3)#tf.tensordot(Wp,layers[l].W,[[2,3],[0,2]])
                proj      = (tf.tensordot(layers[l-1].M,layers[l].W,[[1],[2]])-retro+tf.expand_dims(prior,-1))/2
                value     = proj/tf.expand_dims(scaling+eps,0)
                m         = tf.assign(layers[l].m,value)
		return m










def update_v(layers,l):
       if(isinstance(layers[l],DenseLayer)):
		scaling = tf.reduce_sum(layers[l].W*layers[l].W,axis=2)
		scaling+=1#layers[l].sigmas2[0]/layers[l+1].sigmas2[0]
                v2      = tf.assign(layers[l].v2,1/tf.expand_dims(eps+scaling,0))#tf.assign(layers[l].v2,layers[l].sigmas2[0]/tf.expand_dims(eps+scaling,0))
		return v2


def update_p(layers,l,k):
       if(isinstance(layers[l],DenseLayer)):
                scaling = tf.reduce_sum(layers[l].W[k]*layers[l].W[k],axis=1)
                if(l<len(layers)-1):
                        scaling  += layers[l].sigmas2[0]/layers[l+1].sigmas2[0]
                        prior     = layers[l+1].backward(tf.constant(1,dtype=tf.float32,shape=(1,layers[l+1].D)))[:,k]/layers[l+1].sigmas2[0]
                else:
			prior     = tf.zeros(layers[l].batch_size)
                rec_error = (layers[l-1].M-layers[l].backward(1-tf.one_hot(k,layers[l].D)))/layers[l].sigmas2[0]
                value     = (tf.reduce_sum(tf.expand_dims(layers[l].W[k],0)*tf.expand_dims(rec_error,1),axis=2)+tf.expand_dims(prior,-1))*layers[l].m[:,k,:]-(tf.pow(layers[l].m[:,k,:],2)+layers[l].v2[:,k,:])*scaling/2+tf.log(tf.expand_dims(layers[l].pi[k]+eps,0))
                expvalue=tf.exp(value)
                indices   = tf.transpose(tf.stack([tf.range(layers[0].batch_size),tf.fill([layers[0].batch_size],k)]))
                p         = tf.scatter_nd_update(layers[l].p,indices,expvalue/tf.reduce_sum(expvalue,axis=1,keepdims=True))
                return p



def approx_p(layers,l):
	if(isinstance(layers[l],DenseLayer)):
        	expvalue=tf.exp(tf.pow(layers[l].m,2)/(2*layers[l].v2)+tf.expand_dims(tf.log(layers[l].pi+eps),0))
                p         = tf.assign(layers[l].p,expvalue/(eps+tf.reduce_sum(expvalue,axis=2,keep_dims=True)))
                return p
	elif(isinstance(layers[l],UnsupFinalLayer)):
		expvalue=tf.exp((tf.tensordot(layers[l-1].M,layers[l].W,[[1],[2]])-tf.expand_dims(tf.reduce_sum(layers[l].W*layers[l].W,axis=2),0))/layers[l].sigmas2[0]+tf.expand_dims(tf.log(layers[l].pi+eps),0))
                p         = tf.assign(layers[l].p,expvalue/(eps+tf.reduce_sum(expvalue,axis=2,keep_dims=True)))
                return p








def update_W(layers,l,k):
       if(isinstance(layers[l],DenseLayer)):
		rec     = tf.expand_dims((layers[l-1].M-layers[l].backward(1-tf.one_hot(k,layers[l].D))),1)*tf.expand_dims(layers[l].m[:,k,:],-1)
		scaling = tf.pow(layers[l].m[:,k,:],2)+layers[l].v2[:,k,:]
                w       = tf.scatter_update(layers[l].W,tf.constant(k),tf.reduce_sum(rec,axis=0)/tf.expand_dims(tf.reduce_sum(scaling,axis=0),-1))
                return w




def approx_W(layers,l):
	if(isinstance(layers[l],DenseLayer)):
#                rec     = tf.expand_dims(tf.expand_dims(layers[l-1].M,1)-compute_lmek(layers,l),2)*tf.expand_dims(layers[l].m*layers[l].p,-1)
		rec     = tf.expand_dims(tf.expand_dims(layers[l-1].M,1),1)*tf.expand_dims(layers[l].m*layers[l].p,-1)
                scaling = layers[l].p*(tf.pow(layers[l].m,2)+layers[l].v2)
                w       = tf.assign(layers[l].W,tf.reduce_sum(rec,axis=0)/tf.expand_dims(tf.reduce_sum(eps+scaling,axis=0),-1))
                return w
	if(isinstance(layers[l],UnsupFinalLayer)):
		rec     = tf.reshape(layers[l-1].M,(layers[l].batch_size,1,1,layers[l-1].D))*tf.expand_dims(layers[l].p,-1)
		w       = tf.assign(layers[l].W,tf.reduce_sum(rec,axis=0)/tf.expand_dims(tf.reduce_sum(eps+layers[l].p,axis=0),-1))
                return w



def approx_sigmas(layers,l):
        if(isinstance(layers[l],DenseLayer)):
		if(l==1):
                        a1 = -tf.reduce_sum(tf.pow(layers[l-1].m,2))
		else:
			a1 = -tf.reduce_sum((tf.pow(layers[l-1].m,2)+layers[l-1].v2)*layers[l-1].p)
		a2 = tf.reduce_sum(layers[l-1].M*layers[l].backward(float32(1)))
		a3 = -tf.reduce_sum(tf.expand_dims(tf.reduce_sum(layers[l].W*layers[l].W,axis=2),0)*layers[l].p*(tf.pow(layers[l].m,2)+layers[l].v2))
#                P = tf.reduce_sum(tf.expand_dims(layers[l].W,0)*tf.expand_dims(layers[l].p*layers[l].m,-1),axis=2)#N K D
#                reco = tf.expand_dims(P,1)*(1-tf.reshape(tf.eye(layers[l].D),(1,layers[l].D,layers[l].D,1)))# N K K D
		a4 = -compute_WpmWpm(layers,l)#tf.reduce_sum(tf.tensordot(reco,P,[[0,1,3],[0,1,2]]))
		value = (a1+2*a2+a3+a4)/(layers[l].batch_size*layers[l-1].D)
                w       = tf.assign(layers[l].sigmas2,tf.stack([-value]))
                return w
	else:
                if(l==1):
                        a1 = -tf.reduce_sum(tf.pow(layers[l-1].m,2))/2
                else:
                        a1 = -tf.reduce_sum((tf.pow(layers[l-1].m,2)+layers[l-1].v2)*layers[l-1].p)
                a2 = tf.reduce_sum(layers[l-1].M*layers[l].backward(float32(1)))
                a3 = -tf.reduce_sum(tf.reduce_sum(layers[l].W*layers[l].W,axis=2)*tf.reduce_sum(layers[l].p,0))
#                P = tf.reduce_sum(tf.expand_dims(layers[l].W,0)*tf.expand_dims(layers[l].p*layers[l].m,-1),axis=2)#N K D
#                reco = tf.expand_dims(P,1)*(1-tf.reshape(tf.eye(layers[l].D),(1,layers[l].D,layers[l].D,1)))# N K K D
#                a4 = compute_WpmWpm(layers,l)#tf.reduce_sum(tf.tensordot(reco,P,[[0,1,3],[0,1,2]]))
                value = (a1+2*a2+a3)/(layers[l].batch_size*layers[l-1].D)
                w       = tf.assign(layers[l].sigmas2,tf.stack([-value]))
                return w












def update_pi(layers,l):
       if(isinstance(layers[l],DenseLayer) or isinstance(layers[l],UnsupFinalLayer)):
		p = tf.reduce_mean(layers[l].p+eps,axis=0)
                w = tf.assign(layers[l].pi,p/tf.reduce_sum(p,axis=1,keep_dims=True))
                return w








def plot_p(l):
        figure()
        for n in xrange(5):
                for d in xrange(layers[l].D):
                        subplot(5,layers[l].D,n*layers[l].D+1+d)
                        title('Example '+str(n+1)+' Neuron'+str(d+1))
                        for r in xrange(layers[l].R):
                                plot(all_p[l][:,n,d,r],linewidth=3)

        suptitle('LAYER '+str(l))


def plot_m(l):
        figure()
        for n in xrange(5):
                for d in xrange(layers[l].D):
                        subplot(5,layers[l].D,n*layers[l].D+1+d)
                        title('Example '+str(n+1)+' Neuron'+str(d+1))
                        plot(all_m[l][:,n,d],'r',linewidth=3)

        suptitle('LAYER '+str(l))





def Estep(x_batch,ite):
#        session.run(init_latent(x,layers),feed_dict={x:x_batch})
        all_p = dict()
        all_m = dict()
#	for i in xrange(len(layers)):
#        	all_p[i+1]=[]
#		all_m[i+1]=[]
	layers_i = arange(1,len(layers))#[::-1]
        for i in xrange(ite):
		print "KL",session.run(KL(layers))
#		layers_i=layers_i[::-1]
                for l in layers_i:
#                        all_p[l].append(session.run(get_p(layers))[l-1])
#                        all_m[l].append(session.run(get_m(layers))[l-1])
#			print "\tm",l,session.run(KL(layers))
			if(l<len(layers)-1):
				session.run(approx_m(layers,l))
#			print "\t\t",session.run(KL(layers))
#                        print "\tv",l,session.run(KL(layers))
	                        session.run(update_v(layers,l))
#                        print "\t\t",session.run(KL(layers))
#                        print "\tp",l,session.run(KL(layers))
                        session.run(approx_p(layers,l))
#                        print "\t\t",session.run(KL(layers))
	print session.run(layers[-2].m)#[:10]
#        pp=session.run(layers[-2].p)#[:10]
#	mm = session.run(layers[-2].m)[:10]
#	for k in xrange(layers[-2].D):
#		subplot(1,layers[-2].D,1+k)
#		for xx,i in zip(x_batch,pp[:,k,:]):
#			plot(xx[0],xx[1],'x',color=(i[0],0,i[1]))

#	show()
#	show()

	print "KL",session.run(KL(layers))
#                        session.run(update_v(layers,l))
        return all_p,all_m





def Mstep(ite):
        layers_i = arange(1,len(layers))#[::-1]
        for i in xrange(ite):
#                layers_i=layers_i[::-1]
		LIKELIHOOD.append(session.run(likelihood(layers)))
		print "ML",LIKELIHOOD[-1]
                for l in xrange(1,len(layers)):
                        session.run(update_pi(layers,l))
                        session.run(approx_W(layers,l))
#			session.run(approx_sigmas(layers,l))
#			print "pi",session.run(layers[l].pi)
#                        print "W",session.run(layers[l].W)
#                        print "sigma",session.run(layers[l].sigmas2)








batch_size = 1797

layers = [InputLayer(batch_size,64),DenseLayer(batch_size,64,10,2),UnsupFinalLayer(batch_size,10,10)]

x       = tf.placeholder(tf.float32,shape=[batch_size,layers[0].D])



#opti = tf.train.AdamOptimizer(0.1)
#train_op1 = opti.minimize(KL(layers),var_list=tf.get_collection('latent'))
#train_op2 = opti.minimize(-likelihood(layers),var_list=tf.get_collection('params'))



session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)

XX = load_digits()['data']#make_moons(batch_size,noise=0.051)
#XX[batch_size/4:]+=3
#XX[batch_size/4:batch_size/2,1]-=1

XX-=XX.mean(1,keepdims=True)
XX/=XX.max(1,keepdims=True)

#XX/=XX.std(0,keepdims=True)
#XX*=2


LIKELIHOOD = []

session.run(init_latent(x,layers),feed_dict={x:XX})



U=session.run(sample(layers))
#subplot(1,10,1)
#imshow(XX[0].reshape((8,8)),aspect='auto')
figure()
for i in xrange(25):
    subplot(5,5,1+i)
    imshow(U[i].reshape((8,8)),aspect='auto')



LIKELIHOOD = []


for i in xrange(10):
        all_p,all_m=Estep(XX,2)
        Mstep(2)










U=session.run(sample(layers))
#pp = session.run(layers[-1].p)
figure()
for i in xrange(25):
    subplot(5,5,1+i)
    imshow(U[i].reshape((8,8)),aspect='auto')


show()






