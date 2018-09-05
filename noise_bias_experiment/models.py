execfile('lasagne_tf.py')
execfile('utils.py')
import random

def onehot(n,k):
        z=zeros(n,dtype='float32')
        z[k]=1
        return z

class DNNClassifier(object):
	def __init__(self,input_shape,model_class,lr=0.0001,optimizer = adam,std=0):
		tf.reset_default_graph()
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.n_classes=  model_class.n_classes
		config.log_device_placement=True
		self.session = tf.Session(config=config)
		self.batch_size = input_shape[0]
		self.lr = lr
		with tf.device('/device:GPU:0'):
			self.learning_rate = tf.placeholder(tf.float32,name='learning_rate')
			optimizer = optimizer(self.learning_rate)
        		self.x             = tf.placeholder(tf.float32, shape=input_shape,name='x')
        	        self.y_            = tf.placeholder(tf.int32, shape=[input_shape[0]],name='y')
        	        self.test_phase    = tf.placeholder(tf.bool,name='phase')
        	        self.layers = model_class.get_layers(self.x+tf.random_normal(input_shape)*std,input_shape,test=self.test_phase)
			count_number_of_params()
        	        self.prediction    = self.layers[-1].output
                        self.crossentropy_loss = tf.reduce_mean(categorical_crossentropy(self.prediction,self.y_))
			self.loss          = self.crossentropy_loss
        	        self.variables     = tf.trainable_variables()
                        self.apply_updates = optimizer.apply(self.loss,self.variables)
        	        self.accuracy      = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(self.prediction,1),tf.int32), self.y_),tf.float32))
		self.session.run(tf.global_variables_initializer())
	def get_params(self):
		return self.session.run(self.variables)
	def _fit(self,X,y,indices,update_time=10):
		self.e+=1
                if(self.e==25 or self.e==50 or self.e==80):
	                self.lr/=10
        	n_train    = X.shape[0]/self.batch_size
        	train_loss = []
        	for i in xrange(n_train):
			if(self.batch_size<self.n_classes):
				here = [random.sample(k,1) for k in indices]
				here = [here[i] for i in permutation(self.n_classes)[:self.batch_size]]
			else:
				here = [random.sample(k,self.batch_size/self.n_classes) for k in indices]
			here = concatenate(here)
                        self.session.run(self.apply_updates,feed_dict={self.x:X[here],self.y_:y[here],self.test_phase:True,self.learning_rate:float32(self.lr)})#float32(self.lr/sqrt(self.e))})
#			if(i%update_time==0):
#                                train_loss.append(self.session.run(self.loss,feed_dict={self.x:X[here],self.y_:y[here],self.test_phase:True}))
#                        if(i%100 ==0):
#                            print i,n_train,train_loss[-1]
        	return train_loss
        def fit(self,X,y,X_test,y_test,n_epochs=5,return_train_accu=0):
		if(n_epochs==0):
			return [0],[0],[]
		train_accu = []
		test_accu  = []
		self.e     = 0
                n_test     = X_test.shape[0]/self.batch_size
                indices    = [find(y==k) for k in xrange(self.n_classes)]
		for i in xrange(n_epochs):
			print "epoch",i
			self._fit(X,y,indices)
			# NOW COMPUTE TEST SET ACCURACY
                	acc1 = 0.0
                	for j in xrange(n_test):
                	        acc1+=self.session.run(self.accuracy,feed_dict={self.x:X_test[self.batch_size*j:self.batch_size*(j+1)],
						self.y_:y_test[self.batch_size*j:self.batch_size*(j+1)],self.test_phase:False})
                	test_accu.append(acc1/n_test)
			if(1):#return_train_accu):
		                n_train    = X.shape[0]/self.batch_size
	                        acc1 = 0.0
	                        for j in xrange(n_train):
	                                acc1+=self.session.run(self.accuracy,feed_dict={self.x:X[self.batch_size*j:self.batch_size*(j+1)],
	                                                self.y_:y[self.batch_size*j:self.batch_size*(j+1)],self.test_phase:False})
	                        train_accu.append(acc1/n_train)
				print train_accu[-1]
			# SAVE LAST W FOR STATISTIC COMPUTATION
                	print 'test accu',test_accu[-1]
        	return asarray(train_accu),asarray(test_accu)





class densedouble:
        def __init__(self,bn=0,n_classes=10,bias=1,nonlinearity=tf.nn.relu,Ns=[4,4,4]):
                self.nonlinearity = nonlinearity
                self.bn = bn
                self.Ns = Ns
                self.bias=bias
                self.n_classes = n_classes
        def get_layers(self,input_variable,input_shape,test):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(DenseLayer(layers[-1],self.Ns[0],test,nonlinearity=self.nonlinearity,bias=self.bias,bn=self.bn))
                layers.append(DenseLayer(layers[-1],self.Ns[1],test,nonlinearity=self.nonlinearity,bias=self.bias,bn=self.bn))
                layers.append(DenseLayer(layers[-1],self.n_classes,test,bn=0,nonlinearity=lambda x:x,bias=0))
                return layers







class smallCNN:
        def __init__(self,bn=1,n_classes=10,bias=1,nonlinearity=tf.nn.relu):
                self.nonlinearity = nonlinearity
                self.bn = bn
                self.bias=bias
                self.n_classes = n_classes
        def get_layers(self,input_variable,input_shape,test):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(Conv2DLayer(layers[-1],32,3,test=test,bn=self.bn,bias=self.bias,nonlinearity=self.nonlinearity))
                layers.append(Pool2DLayer(layers[-1],2))
                layers.append(Conv2DLayer(layers[-1],64,3,test=test,bn=self.bn,bias=self.bias,nonlinearity=self.nonlinearity))
                layers.append(Pool2DLayer(layers[-1],2))
                layers.append(Conv2DLayer(layers[-1],128,1,test=test,bn=self.bn,bias=self.bias,nonlinearity=self.nonlinearity))
                layers.append(GlobalPoolLayer(layers[-1]))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=test,bn=0,nonlinearity=lambda x:x,bias=0))
                return layers


class smallCNNnopool:
        def __init__(self,bn=1,n_classes=10,augmentation=0,p=0,bias=1,nonlinearity=tf.nn.relu):
                self.nonlinearity = nonlinearity
                self.bn = bn
                self.augmentation = augmentation
                self.p = p
                self.bias=bias
                self.n_classes = n_classes
        def get_layers(self,input_variable,input_shape,test):
                layers = [InputLayer(input_shape,input_variable)]
                masks  = []
                if(self.augmentation):
                    layers.append(Generator(layers[-1],test=test,p=self.p))
                layers.append(Conv2DLayer(layers[-1],32,3,test=test,bn=self.bn,bias=self.bias,nonlinearity=self.nonlinearity))
                masks.append(tf.greater(layers[-1].output,0))
                layers.append(Conv2DLayer(layers[-1],64,3,test=test,bn=self.bn,bias=self.bias,nonlinearity=self.nonlinearity))
                masks.append(tf.greater(layers[-1].output,0))
                layers.append(Conv2DLayer(layers[-1],128,1,test=test,bn=self.bn,bias=self.bias,nonlinearity=self.nonlinearity))
                masks.append(tf.greater(layers[-1].output,0))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=test,bn=0,nonlinearity=lambda x:x,bias=0))
                return layers,masks












class CNNresnet:
        def __init__(self,bn=1,n_classes=10,augmentation=0,p=0,bias=1,nonlinearity=tf.nn.relu):
                self.bn = bn
                self.augmentation = augmentation
                self.p = p
		self.nonlinearity=nonlinearity
                self.bias = bias
                self.n_classes = n_classes
        def get_layers(self,input_variable,input_shape,test):
                depth = 10
                k = 1
                layers = [InputLayer(input_shape,input_variable)]
                if(self.augmentation):
                    layers.append(Generator(layers[-1],test=test,p=self.p))
                layers.append(Conv2DLayer(layers[-1],16*k,3,test=test,bn=0,pad='same',nonlinearity= lambda x:x,bias=self.bias))
                for i in xrange(depth):
                    layers.append(NNConv2DLayer(layers[-1],16*k,3,pad='same',test=test,bias=self.bias,nonlinearity=self.nonlinearity))
                layers.append(NNConv2DLayer(layers[-1],16*k*2,3,pad='same',test=test,bias=self.bias,nonlinearity=self.nonlinearity))
                layers.append(Pool2DLayer(layers[-1],2))
                for i in xrange(depth-1):
                    layers.append(NNConv2DLayer(layers[-1],16*k*2,3,pad='same',test=test,bias=self.bias,nonlinearity=self.nonlinearity))
                layers.append(NNConv2DLayer(layers[-1],16*k*4,3,pad='same',test=test,bias=self.bias,nonlinearity=self.nonlinearity))
                layers.append(Pool2DLayer(layers[-1],2))
                for i in xrange(depth-1):
                    layers.append(NNConv2DLayer(layers[-1],16*k*4,3,pad='same',test=test,bias=self.bias,nonlinearity=self.nonlinearity))
                layers.append(GlobalPoolLayer(layers[-1]))
                layers.append(DenseLayer(layers[-1],self.n_classes,nonlinearity=lambda x:x,bias=0))
                return layers,0





class resnet_large:
        def __init__(self,bn=1,n_classes=10,augmentation=0,p=0,bias=1):
                self.bn = bn
                self.augmentation = augmentation
                self.p = p
		self.bias = bias
                self.n_classes = n_classes
        def get_layers(self,input_variable,input_shape,test):
                depth = 12
                k = 2
                layers = [InputLayer(input_shape,input_variable)]
                if(self.augmentation):
                    layers.append(Generator(layers[-1],test=test,p=self.p))
                layers.append(Conv2DLayer(layers[-1],16*k,3,test=test,bn=0,pad='same',nonlinearity= lambda x:x,bias=self.bias))
                for i in xrange(depth):
                    layers.append(Block(layers[-1],16*k,1,test=test,bias=self.bias))
                layers.append(Block(layers[-1],16*k*2,2,test=test,bias=self.bias))
                for i in xrange(depth-1):
                    layers.append(Block(layers[-1],16*k*2,1,test=test,bias=self.bias))
                layers.append(Block(layers[-1],16*k*4,2,test=test,bias=self.bias))
                for i in xrange(depth-1):
                    layers.append(Block(layers[-1],16*k*4,1,test=test,bias=self.bias))
                layers.append(GlobalPoolLayer(layers[-1]))
                layers.append(DenseLayer(layers[-1],self.n_classes,nonlinearity=lambda x:x,bias=0))
                return layers



class smallRESNET:
        def __init__(self,bn=1,n_classes=10,augmentation=0,p=0,bias=1,nonlinearity=tf.nn.relu):
                self.bn = bn
                self.augmentation = augmentation
                self.p = p
		self.nonlinearity = nonlinearity
		self.bias = bias
                self.n_classes = n_classes
        def get_layers(self,input_variable,input_shape,test):
                depth = 2
                k = 1
                layers = [InputLayer(input_shape,input_variable)]
                if(self.augmentation):
                    layers.append(Generator(layers[-1],test=test,p=self.p))
                layers.append(Conv2DLayer(layers[-1],16*k,3,test=test,bn=self.bn,pad='same',nonlinearity= lambda x:x))
                for i in xrange(depth):
                    layers.append(Block(layers[-1],16*k,1,test=test,bias=self.bias,bn=self.bn,nonlinearity=self.nonlinearity))# Resnet 4-4 straightened bottleneck
                layers.append(Block(layers[-1],16*k*2,2,test=test,bias=self.bias,bn=self.bn,nonlinearity=self.nonlinearity))
                for i in xrange(depth-1):
                    layers.append(Block(layers[-1],16*k*2,1,test=test,bias=self.bias,bn=self.bn,nonlinearity=self.nonlinearity))
                layers.append(Block(layers[-1],16*k*4,2,test=test,bias=self.bias,bn=self.bn,nonlinearity=self.nonlinearity))
                for i in xrange(depth-1):
                    layers.append(Block(layers[-1],16*k*4,1,test=test,bias=self.bias,bn=self.bn,nonlinearity=self.nonlinearity))
                layers.append(GlobalPoolLayer(layers[-1]))
                layers.append(DenseLayer(layers[-1],self.n_classes,test,nonlinearity=lambda x:x,bias=0))
                return layers,0



class largeCNN:
        def __init__(self,bn=1,n_classes=10,augmentation=0,p=0,bias=1,nonlinearity=tf.nn.relu):
                self.bn = bn
		self.nonlinearity = nonlinearity
                self.p = p
		self.bias = bias
                self.augmentation = augmentation
		self.n_classes = n_classes
        def get_layers(self,input_variable,input_shape,test):
                layers = [InputLayer(input_shape,input_variable)]
                masks  = []
                if(self.augmentation):
                    layers.append(Generator(layers[-1],test=test,p=self.p))
                layers.append(Conv2DLayer(layers[-1],96,3,pad='same',test=test,bn=self.bn,bias=self.bias,nonlinearity=self.nonlinearity))
                masks.append(tf.greater(layers[-1].output,0))
                layers.append(Conv2DLayer(layers[-1],96,3,pad='full',test=test,bn=self.bn,bias=self.bias,nonlinearity=self.nonlinearity))
                masks.append(tf.greater(layers[-1].output,0))
                layers.append(Conv2DLayer(layers[-1],96,3,pad='full',test=test,bn=self.bn,bias=self.bias,nonlinearity=self.nonlinearity))
                masks.append(tf.greater(layers[-1].output,0))
                layers.append(Pool2DLayer(layers[-1],2))
                masks.append(tf.gradients(layers[-1].output,layers[-2].output)[0])
                layers.append(Conv2DLayer(layers[-1],192,3,test=test,bn=self.bn,bias=self.bias,nonlinearity=self.nonlinearity))
                masks.append(tf.greater(layers[-1].output,0))
                layers.append(Conv2DLayer(layers[-1],192,3,pad='full',test=test,bn=self.bn,bias=self.bias,nonlinearity=self.nonlinearity))
                masks.append(tf.greater(layers[-1].output,0))
                layers.append(Conv2DLayer(layers[-1],192,3,test=test,bn=self.bn,bias=self.bias,nonlinearity=self.nonlinearity))
                masks.append(tf.greater(layers[-1].output,0))
                layers.append(Pool2DLayer(layers[-1],2))
                masks.append(tf.gradients(layers[-1].output,layers[-2].output)[0])
                layers.append(Conv2DLayer(layers[-1],192,3,test=test,bn=self.bn,bias=self.bias,nonlinearity=self.nonlinearity))
                masks.append(tf.greater(layers[-1].output,0))
                layers.append(Conv2DLayer(layers[-1],192,1,test=test,bias=self.bias,nonlinearity=self.nonlinearity))
                masks.append(tf.greater(layers[-1].output,0))
                layers.append(GlobalPoolLayer(layers[-1]))
#                masks.append(tf.gradients(layers[-1].output,layers[-2].output)[0])
                layers.append(DenseLayer(layers[-1],self.n_classes,training=test,nonlinearity=lambda x:x,bias=0,bn=0))
                return layers,masks

class NNlargeCNN:
        def __init__(self,bn=1,n_classes=10,augmentation=0,p=0,bias=1,nonlinearity=tf.nn.relu):
                self.bn = bn
                self.nonlinearity = nonlinearity
                self.p = p
                self.bias = bias
                self.augmentation = augmentation
                self.n_classes = n_classes
        def get_layers(self,input_variable,input_shape,test):
                layers = [InputLayer(input_shape,input_variable)]
                masks  = []
                if(self.augmentation):
                    layers.append(Generator(layers[-1],test=test,p=self.p))
                layers.append(NNConv2DLayer(layers[-1],96,3,pad='same',test=test,bn=self.bn,bias=self.bias,nonlinearity=self.nonlinearity))
                masks.append(tf.greater(layers[-1].output,0))
                layers.append(NNConv2DLayer(layers[-1],96,3,pad='full',test=test,bn=self.bn,bias=self.bias,nonlinearity=self.nonlinearity))
                masks.append(tf.greater(layers[-1].output,0))
                layers.append(NNConv2DLayer(layers[-1],96,3,pad='full',test=test,bn=self.bn,bias=self.bias,nonlinearity=self.nonlinearity))
                masks.append(tf.greater(layers[-1].output,0))
                layers.append(Pool2DLayer(layers[-1],2))
                masks.append(tf.gradients(layers[-1].output,layers[-2].output)[0])
                layers.append(NNConv2DLayer(layers[-1],192,3,test=test,bn=self.bn,bias=self.bias,nonlinearity=self.nonlinearity))
                masks.append(tf.greater(layers[-1].output,0))
                layers.append(NNConv2DLayer(layers[-1],192,3,pad='full',test=test,bn=self.bn,bias=self.bias,nonlinearity=self.nonlinearity))
                masks.append(tf.greater(layers[-1].output,0))
                layers.append(NNConv2DLayer(layers[-1],192,3,test=test,bn=self.bn,bias=self.bias,nonlinearity=self.nonlinearity))
                masks.append(tf.greater(layers[-1].output,0))
                layers.append(Pool2DLayer(layers[-1],2))
                masks.append(tf.gradients(layers[-1].output,layers[-2].output)[0])
                layers.append(NNConv2DLayer(layers[-1],192,3,test=test,bn=self.bn,bias=self.bias,nonlinearity=self.nonlinearity))
                masks.append(tf.greater(layers[-1].output,0))
                layers.append(NNConv2DLayer(layers[-1],192,1,test=test,bias=self.bias,nonlinearity=self.nonlinearity))
                masks.append(tf.greater(layers[-1].output,0))
                layers.append(Pool2DLayer(layers[-1],2))
                masks.append(tf.gradients(layers[-1].output,layers[-2].output)[0])
                layers.append(NNDenseLayer(layers[-1],self.n_classes,nonlinearity=lambda x:x,bias=0))
                return layers,masks






