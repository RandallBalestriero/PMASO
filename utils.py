import tensorflow as tf
import time
from pylab import *
import layers as layers_







#############################################################################################
#
#
#                       UPDATE and SAMPLE HELPER
#
#
#############################################################################################


def update_sigma(layers,local=1):
    if(local):
        v2 = []
        for l in layers:
            if(not isinstance(l,layers_.InputLayer)):
                v2+=l.update_sigma(local)
        return v2
    else:
        v2   = 0.
        accu = 0.
        for l in layers:
            if(not isinstance(l,layers_.InputLayer)):
                v2+=l.update_sigma(local)
                accu+=prod(l.input_shape)
        update_ops = []
        for l in layers:
            if(not isinstance(l,layers_.InputLayer)):
                update_ops.append(tf.assign(l.sigmas2,[tf.clip_by_value(v2/accu,0.0005,10)]))
        return update_ops


def sample(layers,Ks=None,sigma=1):
    """Ks is used if one wants to pre imposed some 
    t variables at different layers, one can provide
    a specific set of t for any layer but must be
    of the same shape as the inner layer variable"""
    if(Ks == None):
        Ks = [None]*len(layers)
    samples=0 # variables that carries the per layer samples representation going from layer to layer
    for i in xrange(len(layers)-1,0,-1):
	samples = layers[i].sample(samples,Ks[i],sigma)
    return samples

def sampleclass(layers,K,sigma=1):
    """Ks is used if one wants to pre imposed some 
    t variables at different layers, one can provide
    a specific set of t for any layer but must be
    of the same shape as the inner layer variable"""
    Ks  = [None]*(len(layers)-1)
    pp = zeros((layers[-1].input_shape[0],1,layers[-1].R),dtype='float32')
    pp[:,:,K]=1
    Ks.append(pp)
    samples=0 # variables that carries the per layer samples representation going from layer to layer
    for i in xrange(len(layers)-1,0,-1):
	samples = layers[i].sample(samples,Ks[i],sigma)
    return samples


def sampletrue(layers):
    s=float32(1)
    try:
        return layers[1].deconv()
    except:
        return layers[1].backward(0)


def SSE(x,y):
    return tf.reduce_sum(tf.square(x-y))
	
		


def likelihood(layers):
    """ gather all the per layer likelihoods
    and add them together as derived in the paper"""
    like = 0
    for l in layers:
        like+=l.likelihood()
    return like
	

def KL(layers):
    """gather the KL divergence by
    summing the per layers one as derived"""
    kl = 0
    for l in layers:
        kl+=l.KL()
    return kl



class model:
    def __init__(self,layers,local_sigma=1):
        self.layers    = layers
        self.local_sigma = local_sigma
        self.L         = len(layers)
        self.x         = tf.placeholder(tf.float32,shape=layers[0].input_shape)
        self.y         = tf.placeholder(tf.int32,shape=[layers[0].input_shape[0]]) # MIGHT NOT BE USED DEPENDING ON SETTING
        self.sigma     = tf.placeholder(tf.float32)
        # CREATE DN LOSS FOR INIT
        self.dn_loss    = tf.losses.softmax_cross_entropy(tf.one_hot(self.y,10),self.layers[-1].dn_p_logits[:,0,:]) # TAKE THE LAST ONE AS ONE NEURON
        learner         = tf.train.AdamOptimizer(0.0001)
        self.dn_updates = learner.minimize(self.dn_loss)
        # INIT SESSION
        session_config = tf.ConfigProto(allow_soft_placement=False,log_device_placement=True)
        session_config.gpu_options.allow_growth=True
        session        = tf.Session(config=session_config)
        init           = tf.global_variables_initializer()
        session.run(init)
        self.session=session
        ## INITIALIZATION
        self.initx              = tf.assign(self.layers[0].m,self.x)
        self.inity              = tf.assign(self.layers[-1].p_,tf.expand_dims(tf.one_hot(self.y,self.layers[-1].R),0))
        self.initop_W_random       = [l.init_W(1) for l in layers[1:]]
        self.initop_W              = [l.init_W(0) for l in layers[1:]]
        self.initop_thetaq_random  = [l.init_thetaq(1) for l in layers[1:]]
        self.initop_thetaq         = [l.init_thetaq(0) for l in layers[1:]]
        ### GATHER UPDATES
        self.updates_vmpk    = [l.update_vmpk() for l in layers[1:]]
        self.updates_sigma_global = update_sigma(layers,0)
        self.updates_sigma_local  = update_sigma(layers,1)
        self.updates_Wk      = [l.update_Wk() for l in layers[1:]]
        self.updates_pi      = [l.update_pi() for l in layers[1:]]
        self.updates_b       = [l.update_b() for l in layers[1:]]
        ## GATHER LOSSES
        self.KL              = KL(layers)
        self.like            = likelihood(layers)
        # GATHER SAMPLING
        self.samplesclass    = [sampleclass(layers,k,sigma=self.sigma) for k in xrange(layers[-1].R)]
        self.samples         = sample(layers,sigma=self.sigma)
        self.samplet         = sampletrue(layers)
        # CREATE INDICES FOR CYCLING THE E STEP
        indices = []
        for l,l_ in zip(layers[1:],range(self.L-1)):
            indices.append([])
            if(isinstance(l,layers_.PoolLayer) or isinstance(l,layers_.FinalLayer)):
                indices[-1].append([l_])
            elif(isinstance(l,layers_.DenseLayer)):
                for k in xrange(l.K):
                    indices[-1].append([l_,k])
            elif(isinstance(l,layers_.ConvLayer)):
                for i in xrange(l.I):
                    for j in xrange(l.J):
                        for k in xrange(l.K):
                            indices[-1].append([l_,i,j,k])
        self.indices = indices#[i for j in indices for i in j]
    def E_step(self,random=1,fineloss=1):
        loss = []
#        if(random):
#            perm = permutation(len(self.indices))
#        else:
#            perm = range(len(self.indices))
#        for i in perm:
#            t=time.time()
#            if(len(self.indices[i])==1):   # POOL AND LAST LAYER
#                self.session.run(self.updates_vmpk[self.indices[i][0]])
#            elif(len(self.indices[i])==2): # FULLY CONNECTED
#                self.session.run(self.updates_vmpk[self.indices[i][0]],feed_dict={self.layers[self.indices[i][0]+1].k_:int32(self.indices[i][1])})
#            else:                     # CONV
#                self.session.run(self.updates_vmpk[self.indices[i][0]],feed_dict={self.layers[self.indices[i][0]+1].i_:int32(self.indices[i][1]),self.layers[self.indices[i][0]+1].j_:int32(self.indices[i][2]),self.layers[self.indices[i][0]+1].k_:int32(self.indices[i][3])})
#            newtime = time.time()-t
#            if(fineloss):
#                t=time.time()
#                L = self.session.run(self.KL)
#                loss.append(L)
#                print "E AFTER ",self.indices[i],L,'KL',time.time()-t,'update',newtime
#        if(fineloss==0):
#            L = self.session.run(self.KL)
#            loss.append(L)
#        return loss
        loss = []
        for indices in self.indices[:len(self.indices)-self.hold_last_p]:      # EACH ITEM IS A LIST OF INDICES FOR EACH LAYER remove the last layer if y was given
            if(random):
                perm = permutation(len(indices))
            else:
                perm = range(len(indices))
            for i in perm:
                t=time.time()
                if(len(indices[i])==1):   # POOL AND LAST LAYER
                    self.session.run(self.updates_vmpk[indices[i][0]])
                elif(len(indices[i])==2): # FULLY CONNECTED
                    self.session.run(self.updates_vmpk[indices[i][0]],feed_dict={self.layers[indices[i][0]+1].k_:int32(indices[i][1])})
                else:                     # CONV
                    self.session.run(self.updates_vmpk[indices[i][0]],feed_dict={self.layers[indices[i][0]+1].i_:int32(indices[i][1]),self.layers[indices[i][0]+1].j_:int32(indices[i][2]),self.layers[indices[i][0]+1].k_:int32(indices[i][3])})
                newtime = time.time()-t
                if(fineloss):
                    t=time.time()
                    L = self.session.run(self.KL)
                    loss.append(L)
                    print "E AFTER ",indices[i],L,'KL',time.time()-t,'update',newtime
        if(fineloss==0):
            L = self.session.run(self.KL)
            loss.append(L)
        return loss
    def M_step(self,random=1,fineloss=1,local=0):
        loss = []
        self.session.run(self.updates_pi)
        if(fineloss):
            L = self.session.run(self.like)
            loss.append(L)
        for l in xrange(self.L-2):
            if(isinstance(self.layers[l+1],layers_.PoolLayer)):
                pass
            else:
                if(random):
                    perm = permutation(self.layers[l+1].K).astype('int32')
                else:
                    perm = range(self.layers[l+1].K)
                for kk in perm:
                    t=time.time()
                    self.session.run(self.updates_Wk[l],feed_dict={self.layers[l+1].k_:int32(kk)})
                    newtime = time.time()-t
                    if(fineloss):
                        t=time.time()
                        L = self.session.run(self.like)
                        loss.append(L)
                        print "M AFTER W",l,kk,L,'likelihood',time.time()-t,'update',newtime
        self.session.run(self.updates_Wk[-1])
        if(fineloss):
            L = self.session.run(self.like)
            loss.append(L)
            print "M AFTER LAST W",L
        self.session.run(self.updates_b)
        if(fineloss):
            L = self.session.run(self.like)
            loss.append(L)
            print "M AFTER B",L
	if(local==0):
        	self.session.run(self.updates_sigma_global)
        	L = self.session.run(self.like)
        	loss.append(L)
        	print "M AFTER S",L,self.session.run([l.sigmas2 for l in self.layers[1:]])
	else:
	        self.session.run(self.updates_sigma_local)
        	L = self.session.run(self.like)
        	loss.append(L)
        	print "M AFTER S",L,self.session.run([l.sigmas2 for l in self.layers[1:]])
        L = self.session.run(self.like)
        loss.append(L)
        return loss
    def init_dataset(self,x,y=None):
        """this function initializes the variable of the
        first layer with x and possibly the last layer
        with y. Depending if y is given or not, an extra variable
        is updated that will impact the E-step. If no y is given the E step
        is done also on the last layer, to infer the p values"""
        self.session.run(self.initx,feed_dict={self.x:x})
        if(y is not None):
            self.session.run(self.inity,feed_dict={self.y:y})
            self.hold_last_p = 1
        else:
            self.hold_last_p = 0
    def init_thetaq(self,random=1):
        """this function is used alone when for example testing
        the model on a new dataset (using init_dataset),
        then the parameters of the model are kept as the 
        trained ones but one aims at correct initialization 
        of the Q fistribution parameters. 
        For initialization of the whole model see the below fun"""
        if(random):
            for op in self.initop_thetaq_random:
                if(op is not None):
                    self.session.run(op)
        else:
            for op in self.initop_thetaq:
                if(op is not None):
                    self.session.run(op)
    def train_dn(self,n,y):
        loss = []
        for i in xrange(n):
            self.session.run(self.dn_updates,feed_dict={self.y:y})
            loss.append(self.session.run(self.dn_loss,feed_dict={self.y:y}))
        return loss
    def sample(self,sigma):
        return self.session.run(self.samples,feed_dict={self.sigma:float32(sigma)})
    def sampleclass(self,sigma,k):
        return self.session.run(self.samplesclass[k],feed_dict={self.sigma:float32(sigma)})
    def reconstruct(self):
        return self.session.run(self.samplet)
    def predict(self):
        return self.session.run(self.layers[-1].p_)[0]


def train_model(model,rcoeff,CPT,random=0,fineloss=1,return_time=0):
    global_global_L = 0
    cpt             = 0
    LOSSES          = []
    L               = rcoeff*2
    inittime=time.time()
    timings = []
    while((L-global_global_L)>rcoeff and cpt<CPT):
        print cpt
        newtime = time.time()
        cpt  += 1
        lcpt  = 0
        global_global_L=model.session.run(model.like)
        ########################################### E STEP
        L        = rcoeff*2
        global_L = 0
        while((L-global_L)>rcoeff and lcpt<CPT/3):
            LOSSES.append([[],'KL'])
            global_L  = model.session.run(model.KL)
            LOSSES[-1][0].append(model.E_step(random=random,fineloss=fineloss))
            L = LOSSES[-1][0][-1][-1] # TAKE THE LAST KL VALUE
            print "E",(L-global_L),">",rcoeff
        ########################################### M STEP
        L        = rcoeff*2
        global_L = 0
        lcpt     = 0
        while((L-global_L)>rcoeff and lcpt<CPT):
            LOSSES.append([[],'LIKE'])
            lcpt+=1
            global_L = model.session.run(model.like)
            LOSSES[-1][0].append(model.M_step(random=random,fineloss=fineloss,local=0))
            L = LOSSES[-1][0][-1][-1] # TAKE THE LAST LIKE VALUE
            print "M",(L-global_L),">",rcoeff
        print "GLOBAL",(L-global_global_L),">",rcoeff
        timings.append(time.time()-newtime)
    print "TRAINING DONE IN ",time.time()-inittime
    if(return_time):
        return LOSSES,timings
    return LOSSES





###################################################################
#
#
#                       UTILITY FOR CIFAR10 & MNIST
#
#
###################################################################



import cPickle
import glob
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split



def load_data(DATASET):
    if(DATASET=='MNIST'):
        batch_size = 50
        mnist         = fetch_mldata('MNIST original')
        x             = mnist.data.reshape(70000,1,28,28).astype('float32')
        y             = mnist.target.astype('int32')
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=10000,stratify=y)
        input_shape   = (batch_size,28,28,1)
#	x_train = transpose(x_train,[0,2,3,1])
#	x_test  = transpose(x_test,[0,2,3,1])
	c = 10
        n_epochs = 150

    elif(DATASET == 'CIFAR'):
        batch_size = 50
        TRAIN,TEST = load_cifar(3)
        x_train,y_train = TRAIN
        x_test,y_test     = TEST
        input_shape       = (batch_size,32,32,3)
        x_train = transpose(x_train,[0,2,3,1])
        x_test  = transpose(x_test,[0,2,3,1])
	c=10
        n_epochs = 150

    elif(DATASET == 'CIFAR100'):
	batch_size = 100
        TRAIN,TEST = load_cifar100(3)
        x_train,y_train = TRAIN
        x_test,y_test     = TEST
        input_shape       = (batch_size,32,32,3)
        x_train = transpose(x_train,[0,2,3,1])
        x_test  = transpose(x_test,[0,2,3,1])
        c=100
        n_epochs = 200

    elif(DATASET=='IMAGE'):
	batch_size=200
        x,y           = load_imagenet()
	x = x.astype('float32')
	y = y.astype('int32')
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=20000,stratify=y)
        input_shape   = (batch_size,64,64,3)
	c=200
        n_epochs = 200

    else:
        batch_size = 50
        TRAIN,TEST = load_svhn()
        x_train,y_train = TRAIN
        x_test,y_test     = TEST
        input_shape       = (batch_size,32,32,3)
        x_train = transpose(x_train,[0,2,3,1])
        x_test  = transpose(x_test,[0,2,3,1])
	c=10
        n_epochs = 150
  
#    x_train          -= x_train.mean((1,2,3),keepdims=True)
    x_train          /= abs(x_train).max((1,2,3),keepdims=True)#/10
#    x_test           -= x_test.mean((1,2,3),keepdims=True)
    x_test           /= abs(x_test).max((1,2,3),keepdims=True)
    x_train           = x_train.astype('float32')
    x_test            = x_test.astype('float32')
    y_train           = array(y_train).astype('int32')
    y_test            = array(y_test).astype('int32')
    return x_train,y_train,x_test,y_test



def principal_components(x):
    x = x.transpose(0, 2, 3, 1)
    flatx = numpy.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
    sigma = numpy.dot(flatx.T, flatx) / flatx.shape[1]
    U, S, V = numpy.linalg.svd(sigma)
    eps = 0.0001
    return numpy.dot(numpy.dot(U, numpy.diag(1. / numpy.sqrt(S + eps))), U.T)


def zca_whitening(x, principal_components):
#    x = x.transpose(1,2,0)
    flatx = numpy.reshape(x, (x.size))
    whitex = numpy.dot(flatx, principal_components)
    x = numpy.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
    return x

def load_imagenet():
        import scipy.misc
        classes = glob.glob('../../DATASET/tiny-imagenet-200/train/*')
        x_train,y_train = [],[]
        cpt=0
        for c,name in zip(range(200),classes):
                print name
                files = glob.glob(name+'/images/*.JPEG')
                for f in files:
                        x_train.append(scipy.misc.imread(f, flatten=False, mode='RGB'))
                        y_train.append(c)
	return asarray(x_train),asarray(y_train)



def load_svhn():
        import scipy.io as sio
        train_data = sio.loadmat('../../DATASET/train_32x32.mat')
        x_train = train_data['X'].transpose([3,2,0,1]).astype('float32')
        y_train = concatenate(train_data['y']).astype('int32')-1
        test_data = sio.loadmat('../../DATASET/test_32x32.mat')
        x_test = test_data['X'].transpose([3,2,0,1]).astype('float32')
        y_test = concatenate(test_data['y']).astype('int32')-1
        print y_test
        return [x_train,y_train],[x_test,y_test]



def unpickle100(file,labels,channels):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    if(channels==1):
        p=dict['data'][:,:1024]*0.299+dict['data'][:,1024:2048]*0.587+dict['data'][:,2048:]*0.114
        p = p.reshape((-1,1,32,32))#dict['data'].reshape((-1,3,32,32))
    else:
        p=dict['data']
        p = p.reshape((-1,channels,32,32)).astype('float64')#dict['data'].reshape((-1,3,32,32))
    if(labels == 0 ):
        return p
    else:
        return asarray(p),asarray(dict['fine_labels'])



def unpickle(file,labels,channels):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    if(channels==1):
        p=dict['data'][:,:1024]*0.299+dict['data'][:,1024:2048]*0.587+dict['data'][:,2048:]*0.114
        p = p.reshape((-1,1,32,32))#dict['data'].reshape((-1,3,32,32))
    else:
        p=dict['data']
        p = p.reshape((-1,channels,32,32)).astype('float64')#dict['data'].reshape((-1,3,32,32))
    if(labels == 0 ):
        return p
    else:
        return asarray(p),asarray(dict['labels'])





def load_mnist():
        mndata = file('../DATASET/MNIST.pkl','rb')
        data=cPickle.load(mndata)
        mndata.close()
        return [concatenate([data[0][0],data[1][0]]).reshape(60000,1,28,28),concatenate([data[0][1],data[1][1]])],[data[2][0].reshape(10000,1,28,28),data[2][1]]

def load_cifar(channels=1):
        path = '../../DATASET/cifar-10-batches-py/'
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for i in ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']:
                PP = unpickle(path+i,1,channels)
                x_train.append(PP[0])
                y_train.append(PP[1])
        x_test,y_test = unpickle(path+'test_batch',1,channels)
        x_train = concatenate(x_train)
        y_train = concatenate(y_train)
        return [x_train,y_train],[x_test,y_test]



def load_cifar100(channels=1):
        path = '../../DATASET/cifar-100-python/'
        PP = unpickle100(path+'train',1,channels)
        x_train = PP[0]
        y_train = PP[1]
        PP = unpickle100(path+'test',1,channels)
        x_test = PP[0]
        y_test = PP[1]
        return [x_train,y_train],[x_test,y_test]











