from pylab import *
import tensorflow as tf

eps = 0.0001

class DenseLayer:
	def __init__(self,batch_size,n_in,n_out,r):
		self.batch_size = batch_size
		self.D       = n_out
		self.R       = r
		self.W       = tf.Variable(tf.random_normal((n_out,r,n_in))/(n_out*n_in))
		self.pi      = tf.Variable(tf.fill([n_out,r],1.0/r))
		self.sigmas2 = tf.Variable(tf.random_uniform([1]))
		self.m       = tf.Variable(tf.random_normal((batch_size,n_out,r)))
		self.p       = tf.Variable(tf.random_normal((batch_size,n_out,r)))
		self.v2      = tf.Variable(tf.random_uniform((batch_size,n_out,r)))
		self.M       = tf.reduce_sum(self.m*self.p,axis=2)
	def forward(self,x):
		return tf.reduce_sum(tf.tensordot(x,self.W,[[1],[2]])*self.p,axis=2)
	def backward(self,e):
		return tf.tensordot(tf.expand_dims(e,-1)*self.p*self.m,self.W,[[1,2],[0,1]])

class InputLayer:
        def __init__(self,batch_size,n_in):
                self.batch_size = batch_size
                self.D       = n_in
                self.m       = tf.Variable(tf.random_normal((batch_size,n_in)))
		self.M       = self.m


def init_latent(x,layers):
	new_p = []
	new_m = [x]
	new_v = []
	for i in xrange(len(layers)):
		if(isinstance(layers[i],InputLayer)):
                        new_m.append(tf.assign(layers[i].m,x))
		if(isinstance(layers[i],DenseLayer)):
			proj       = tf.tensordot(new_m[-1],layers[i].W,[[1],[2]])
			max_values = tf.reduce_max(proj,axis=2,keep_dims=True)
			mask       = tf.cast(tf.greater(proj,max_values-eps),tf.float32)
			renorm_mask= mask/tf.reduce_sum(mask,axis=2,keep_dims=True) 
			new_p.append(tf.assign(layers[i].p,renorm_mask))
			new_m.append(tf.assign(layers[i].m,proj))
			value      = tf.reduce_sum(tf.expand_dims(tf.reduce_sum(layers[i].W*layers[i].W,axis=2),0)*new_p[-1],axis=2)
			if(i<len(layers)-1):
				new_v.append(tf.assign(layers[i].v2,1/(1/layers[i+1].sigmas2+value/layers[i].sigmas2)))
			else:
                                new_v.append(tf.assign(layers[i].v2,1/(value/layers[i].sigmas2)))
	return new_p,new_m


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
                scaling = tf.reduce_sum(layers[l].W[k]*layer[l].W[k],axis=1)
                if(l<len(layers)-1):
                        scaling  += layers[l].sigmas2[0]/layers[l+1].sigmas2[0]
                        prior     = layers[l+1].backward(1)[:,k]
                else:
                        prior     = 0
                rec_error = layers[l-1].M-layers[l].backward(1-tf.one_hot(k,layers[l].D))
                value     = (1/scaling)*(tf.reduce_sum(tf.expand_dims(layers[l].W[k],0)*tf.expand_dims(rec_error,1),axis=2)+prior)
                indices   = tf.transpose(tf.stack([tf.range(layers[0].batch_size),tf.fill([layers[0].batch_size],k)]))
                m         = tf.scatter_nd_update(layers[l].m,indices,value)
		return m


def update_v(layers,l):
       if(isinstance(layers[l],DenseLayer)):
		scaling = tf.reduce_sum(layers[l].W*layer[l].W,axis=2)/layers[l].sigmas2[0]
                if(l<len(layers)-1):
			scaling+=1/layers[l+1].sigmas2[0]
                v2      = tf.assign(layers[l].v2,1/scaling)
		return v2


def update_p(layers,l,k):
       if(isinstance(layers[l],DenseLayer)):
                scaling = tf.reduce_sum(layers[l].W[k]*layer[l].W[k],axis=1)
                if(l<len(layers)-1):
                        scaling  += layers[l].sigmas2[0]/layers[l+1].sigmas2[0]
                        prior     = layers[l+1].backward(1)[:,k]/layers[l+1].sigmas2[0]
                else:
			prior     = 0
                rec_error = (layers[l-1].M-layers[l].backward(1-tf.one_hot(k,layers[l].D)))/layers[l].sigmas2[0]
                value     = (tf.reduce_sum(tf.expand_dims(layers[l].W[k],0)*tf.expand_dims(rec_error,1),axis=2)+prior)*layers[l].m[:,k,:]
                indices   = tf.transpose(tf.stack([tf.range(layers[0].batch_size),tf.fill([layers[0].batch_size],k)]))
                p         = tf.scatter_nd_update(layers[l].m,indices,value-(tf.pow(layers[l].m[:,k,:],2)+layers[l].v2[:,k,:])*scaling/2)



def update_W(layers,l,k):
       if(isinstance(layers[l],DenseLayer)):
		rec     = (layers[l-1].M-layers[l].backward(1-tf.one_hot(k,layers[l].D)))*layers[l].m[:,k,:]
		scaling = tf.pow(layers[l].m[:,k,:],2)+layers[l].v2[:,k,:]
                w       = tf.scatter_update(layers[l].W,tf.constant(k),tf.reduce_sum(rec,axis=0)/tf.reduce_sum(scaling,axis=0))
                return w



def update_pi(layers,l):
       if(isinstance(layers[l],DenseLayer)):
		p = tf.reduce_sum(layers[l].p,axis=0)
                w = tf.assign(layers[l].pi,p/tf.reduce_sum(p,axis=1))
                return w








def plot_p(l):
        figure()
        for n in xrange(batch_size):
                for d in xrange(layers[l].D):
                        subplot(batch_size,layers[l].D,n*layers[l].D+1+d)
                        title('Example '+str(n+1)+' Neuron'+str(d+1))
                        for r in xrange(layers[l].R):
                                plot(all_p[l][:,n,d,r],linewidth=3)

        suptitle('LAYER '+str(l))


def plot_m(l):
        figure()
        for n in xrange(batch_size):
                for d in xrange(layers[l].D):
                        subplot(batch_size,layers[l].D,n*layers[l].D+1+d)
                        title('Example '+str(n+1)+' Neuron'+str(d+1))
                        plot(all_m[l][:,n,d],'r',linewidth=3)

        suptitle('LAYER '+str(l))





def Estep(x_batch,ite):
        session.run(init_latent(x,layers),feed_dict={x:x_batch})

        all_p = dict()
        all_p[1]=[]
        all_p[2]=[]

        all_m = dict()
        all_m[1]=[]
        all_m[2]=[]

        for i in xrange(ite):
                for l in xrange(1,len(layers)):
                        all_p[l].append(session.run(get_p(layers))[l-1])
                        all_m[l].append(session.run(get_m(layers))[l-1])
                        for k in xrange(layers[l].D):
                                print l,k
                                session.run(update_p(layers,l,k))
                                session.run(update_m(layers,l,k))
                                session.run(update_v(layers,l,k))
        return all_p,all_m





def Mstep(ite):
	for l in xrange(1,len(layers)):
	        session.run(update_pi(layers,l))
        for i in xrange(ite):
                for l in xrange(1,len(layers)):
                        for k in xrange(layers[l].D):
                                print l,k
                                session.run(update_W(layers,l,k))








batch_size = 5

layers = [InputLayer(batch_size,3),DenseLayer(batch_size,3,3,2),DenseLayer(batch_size,3,1,2)]

x       = tf.placeholder(tf.float32,shape=[batch_size,3])

session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)



all_p,all_m=Estep(randn(batch_size,3).astype('float32'),3)
Mstep(3)
#all_p,all_m=Estep(randn(batch_size,3).astype('float32'),3)



print shape(all_p[1])
all_p[1]=asarray(all_p[1])
all_p[2]=asarray(all_p[2])
all_m[1]=asarray(all_m[1])
all_m[2]=asarray(all_m[2])


plot_p(1)
plot_p(2)

plot_m(1)
plot_m(2)





show()






