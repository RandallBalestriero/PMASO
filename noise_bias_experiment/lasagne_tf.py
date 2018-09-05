import tensorflow as tf

############################################################################################
#
# OPTIMIZER and LOSSES
#
#
############################################################################################


def resnetnonlinearity(x):
	return tf.maximum(x,2*x)


def ortho_loss2(W):
    return tf.reduce_mean(tf.pow(tf.matmul(W,W,transpose_a=True)-tf.matrix_diag(tf.reduce_mean(W*W,axis=0)),2))

def ortho_loss4(W):
    return tf.reduce_mean(tf.pow(tf.tensordot(W,W,[[0,1,2],[0,1,2]])-tf.matrix_diag(tf.reduce_mean(W*W,axis=[0,1,2])),2))

def categorical_crossentropy(logits, labels):
	labels = tf.cast(labels, tf.int32)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy')
	return cross_entropy


def l1_penaly():
	Ws = tf.get_collection("regularizable")
	cost = tf.add_n([tf.norm(v,ord=1) for v in Ws])/float32(len(Ws))
	return cost


def l2_penaly():
        Ws = tf.get_collection("regularizable")
        cost = tf.add_n([tf.norm(v,ord=2) for v in Ws])/float32(len(Ws))
        return cost



def count_number_of_params():
	print np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])


class adam:
	def __init__(self,alpha=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8):
                self.alpha = alpha
                self.beta1 = beta1
                self.beta2 = beta2
                self.epsilon = epsilon
	def apply(self,loss_or_grads,variables):
		self.m  = dict()
		self.u  = dict()
		updates = dict()
		# If loss generate the gradients else setup the gradients
		if(isinstance(loss_or_grads,list)):
			gradients = loss_or_grads
		else:
			gradients = tf.gradients(loss_or_grads,variables)
		# INIT THE Variables and Update Rules
		self.t = tf.Variable(0.0, trainable=False)
		t      = self.t.assign_add(1.0)
		updates[self.t]= self.t.assign_add(1.0)
		for g,v in zip(gradients,variables):
			self.m[v] = tf.Variable(tf.zeros(tf.shape(v.initial_value)), 'm')
			self.u[v] = tf.Variable(tf.zeros(tf.shape(v.initial_value)), 'u')
               	        updates[self.m[v]] = self.m[v].assign(self.beta1*self.m[v] + (1-self.beta1)*g)
               	        updates[self.u[v]] = self.u[v].assign(self.beta2*self.u[v] + (1-self.beta2)*g*g)
			updates[v]         = v.assign_sub(self.alpha*updates[self.m[v]]/(tf.sqrt(updates[self.u[v]])+self.epsilon))
		print tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		final = tf.get_collection(tf.GraphKeys.UPDATE_OPS)+updates.values()
		return tf.group(*final)


class meta_sgd:
        def __init__(self,alpha,beta,gamma):
                self.gamma   = gamma
		self.alpha   = alpha
		self.beta    = beta
        def apply(self,loss_or_grads,variables):
                self.gradl   = dict()
                self.gradll  = dict()
		self.alphas  = dict()
		self.betas   = dict()
                updates      = dict()
                # If loss generate the gradients else setup the gradients
                if(isinstance(loss_or_grads,list)):
                        gradients = loss_or_grads
                else:
                        gradients = tf.gradients(loss_or_grads,variables)
                for g,v in zip(gradients,variables):
			self.alphas[v] = tf.Variable(self.alpha*tf.ones(tf.shape(v.initial_value)), 'alphas')
                        self.betas[v]  = tf.Variable(self.beta*tf.ones(tf.shape(v.initial_value)), 'betas')
                        self.gradl[v]  = tf.Variable(tf.zeros(tf.shape(v.initial_value)), 'm')
                        self.gradll[v] = tf.Variable(tf.zeros(tf.shape(v.initial_value)), 'u')
			updates[self.betas[v]]  = self.betas[v].assign_sub(self.gamma*g*self.gradl[v]**2*self.gradll[v])
                        updates[self.alphas[v]] = self.alphas[v].assign_add(updates[self.betas[v]]*g*self.gradl[v])
                        updates[v]              = v.assign_sub(updates[self.alphas[v]]*g)
                        updates[self.gradll[v]] = self.gradll[v].assign(self.gradl[v])
			with tf.control_dependencies([updates[self.gradll[v]]]):
                        	updates[self.gradl[v]] = self.gradl[v].assign(g)
                print tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                final = tf.get_collection(tf.GraphKeys.UPDATE_OPS)+updates.values()
                return tf.group(*final)





class Momentum:
        def __init__(self,alpha=0.001,nu=0.9):
                self.alpha = alpha
		self.nu    = nu
        def apply(self,loss_or_grads,variables):
                self.v  = dict()
                updates = dict()
                # If loss generate the gradients else setup the gradients
                if(isinstance(loss_or_grads,list)):
                        gradients = loss_or_grads
                else:   
                        gradients = tf.gradients(loss_or_grads,variables)
                # INIT THE Variables and Update Rules
                for g,v in zip(gradients,variables):
                        self.v[v] = tf.Variable(tf.zeros(tf.shape(v.initial_value)), 'm')
                        updates[self.v[v]] = self.v[v].assign(self.v[v]*self.nu -self.alpha*g)
                        updates[v]         = v.assign_add(updates[self.v[v]])
                print tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                final = tf.get_collection(tf.GraphKeys.UPDATE_OPS)+updates.values()
                return tf.group(*final)

###########################################################################################
#
#
#		Layers
#
#
###########################################################################################



class Pool2DLayer:
	def __init__(self,incoming,window,pool_type='MAX',stride=None):
		self.output = tf.nn.pool(incoming.output,(window,window),pool_type,padding='VALID',strides=(window,window))
		self.output_shape = (incoming.output_shape[0],incoming.output_shape[1]/window,incoming.output_shape[2]/window,incoming.output_shape[3])
		print self.output_shape



def random_crop(value, p):
    offset = tf.random_uniform([2],dtype=tf.int32,maxval=p+1)
    return tf.slice(value,[offset[0],offset[1],0],[value.get_shape()[0]-p,value.get_shape()[1]-p,3])

class Generator:
        def __init__(self,incoming,test,p=2):
            offset = tf.random_uniform([2,incoming.output_shape[0]],minval=0,maxval=p*2+1,dtype=tf.int32)
            self.output_shape= incoming.output_shape
            output_crop      = lambda : tf.map_fn(lambda img: random_crop(img,2*p),tf.pad(incoming.output,[[0,0],[p,p],[p,p],[0,0]],mode='REFLECT'))
            output_crop_flip = lambda : tf.map_fn(lambda img: random_crop(img,2*p),tf.pad(incoming.output,[[0,0],[p,p],[p,p],[0,0]],mode='REFLECT'))[:,:,::-1,:]
            output           = lambda : incoming.output
            z                = tf.random_uniform([1])[0]
            output_augmented = lambda :tf.cond(tf.less(z,0.5),output_crop,output_crop_flip)
            print test
            self.output      = tf.cond(test,output_augmented,output)






class InputLayer:
	def __init__(self,input_shape,x):
		self.output = x
		self.output_shape = input_shape





class DenseLayer:
	def __init__(self,incoming,n_output,training,nonlinearity=tf.nn.relu,bias=1,bn=0):
		if(len(incoming.output_shape)>2):
			inputf = tf.layers.flatten(incoming.output)
			in_dim = prod(incoming.output_shape[1:])
		else:
			inputf = incoming.output
			in_dim = incoming.output_shape[1]
		init   = tf.contrib.layers.xavier_initializer(uniform=True)
                self.W = tf.Variable(init((in_dim,n_output)),name='W_dense',trainable=True)
                if(bias==0 and bn==0):
                        self.b = tf.Variable(tf.zeros((1,n_output)),name='b_dense',trainable=False)
		if(bias==1 and bn==0):
                	self.b = tf.Variable(tf.zeros((1,n_output)),name='b_dense',trainable=True)
                if(bias=='norm' and bn==0):
                        self.b = -0.5*tf.reduce_sum(self.W*self.W,axis=0,keep_dims=True)
                tf.add_to_collection("regularizable",self.W)
		self.output_shape = (incoming.output_shape[0],n_output)
		if(bn==0):
                        input_ = tf.matmul(inputf,self.W)+self.b
		else:
			input_ = tf.layers.batch_normalization(tf.matmul(inputf,self.W),training=training,fused=True)
                self.mask         = tf.greater(input_,0)
                self.output       = nonlinearity(input_)





class NNDenseLayer:
        def __init__(self,incoming,n_output,nonlinearity=tf.nn.relu,bias=1):
                if(len(incoming.output_shape)>2):
                        inputf = tf.layers.flatten(incoming.output)
                        in_dim = prod(incoming.output_shape[1:])
                else:
                        inputf = incoming.output
                        in_dim = incoming.output_shape[1]
                init   = tf.contrib.layers.xavier_initializer(uniform=True)
                if(bias):
                        self.b = tf.Variable(tf.zeros((1,n_output)),name='b_dense',trainable=True)
                self.W = tf.Variable(init((in_dim,n_output)),name='W_dense',trainable=True)
                tf.add_to_collection("regularizable",self.W)
                self.output_shape = (incoming.output_shape[0],n_output)
                if(bias):
                        self.output       = nonlinearity(tf.matmul(inputf,self.W)+self.b)
                else:
                        self.output       = nonlinearity(tf.matmul(inputf,self.W))



class QDenseLayer:
        def __init__(self,incoming,n_output,nonlinearity=tf.nn.relu):
                if(len(incoming.output_shape)>2):
                        inputf = tf.layers.flatten(incoming.output)
                        in_dim = prod(incoming.output_shape[1:])
                else:
                        inputf = incoming.output
                        in_dim = incoming.output_shape[1]
                init   = tf.contrib.layers.xavier_initializer(uniform=True)
                self.b = tf.Variable(tf.zeros((1,n_output)),name='b_dense',trainable=True)
                self.W = tf.Variable(init((in_dim,n_output)),name='W_dense',trainable=True)
                self.b2 = tf.Variable(tf.zeros((1,in_dim)),name='b2_dense',trainable=True)
                self.W2= tf.Variable(init((in_dim,in_dim,n_output)),name='W2_dense',trainable=True)
                tf.add_to_collection("regularizable",self.W)
                tf.add_to_collection("regularizable",self.W2)
                self.output_shape = (incoming.output_shape[0],n_output)
		output2 = tf.tensordot(tf.expand_dims(inputf+self.b2,-1)*tf.expand_dims(inputf+self.b2,-2),[[1,2],[0,1]])
                self.output       = nonlinearity(tf.matmul(inputf,self.W)+output2+self.b)




class GlobalPoolLayer:
        def __init__(self,incoming):
                self.output = tf.reduce_mean(incoming.output,[1,2],keep_dims=True)
                self.output_shape = [incoming.output_shape[0],1,1,incoming.output_shape[3]]


class QConv2DLayer:
        def __init__(self,incoming,n_filters,filter_shape,test,stride=1,pad='valid',option="diagonal",mode='CONSTANT',nonlinearity=tf.nn.relu,bn=0):
                if(pad=='valid' or filter_shape==1):
                        padded_input = incoming.output
                        self.output_shape = (incoming.output_shape[0],(incoming.output_shape[1]-filter_shape+1)/stride,(incoming.output_shape[1]-filter_shape+1)/stride,n_filters)
                elif(pad=='same'):
                        assert(filter_shape%2 ==1)
                        p = (filter_shape-1)/2
                        padded_input = tf.pad(incoming.output,[[0,0],[p,p],[p,p],[0,0]],mode=mode)
                        self.output_shape = (incoming.output_shape[0],incoming.output_shape[1]/stride,incoming.output_shape[2]/stride,n_filters)
                else:
                        p = filter_shape-1
                        padded_input = tf.pad(incoming.output,[[0,0],[p,p],[p,p],[0,0]],mode=mode)
                        self.output_shape = (incoming.output_shape[0],(incoming.output_shape[1]+filter_shape-1)/stride,(incoming.output_shape[1]+filter_shape-1)/stride,n_filters)
                init       = tf.contrib.layers.xavier_initializer(uniform=True)
                self.W     = tf.Variable(init((filter_shape,filter_shape,incoming.output_shape[3],n_filters)),name='W_qconv2d',trainable=True)
		output1    = tf.nn.conv2d(padded_input,self.W,strides=[1,stride,stride,1],padding='VALID')
		if(option=='none'):
			if(bn==0):
	                        self.b  = tf.Variable(tf.zeros((1,1,1,n_filters)),name='b_qconv',trainable=True)
				self.output = nonlinearity(output1+self.b)
			else:
	                        self.output= nonlinearity(tf.layers.batch_normalization(output1,training=test))
		elif(option=="diagonal"):
                        self.b2 = tf.Variable(tf.zeros((1,1,1,incoming.output_shape[3])),name='b2_qconv',trainable=True)
			self.W2 = tf.Variable(tf.zeros((filter_shape,filter_shape,incoming.output_shape[3],n_filters)),name='A_qconv2d',trainable=True)
			output2 = tf.nn.conv2d((padded_input+self.b2)*(padded_input+self.b2),self.W2,strides=[1,stride,stride,1],padding='VALID')
#			self.W3 = tf.Variable(tf.zeros((1,1,n_filters,n_filters*2)),name='A_qconv2d',trainable=True)
#                        self.W4 = tf.Variable(tf.zeros((1,1,n_filters,n_filters*2)),name='A_qconv2d',trainable=True)
                        if(bn==0):
##                                self.b1 = tf.Variable(tf.zeros((1,1,1,n_filters)),name='b_qconv',trainable=True)
                                self.b3 = tf.Variable(tf.zeros((1,1,1,n_filters)),name='b_qconv',trainable=True)
#				output3 = tf.nn.conv2d(output1,self.W3,strides=[1,1,1,1],padding='VALID')+tf.nn.conv2d(output2,self.W4,strides=[1,1,1,1],padding='VALID')+self.b
                                self.output = nonlinearity(output1+output2+self.b3)
                        else:
                                output3 = tf.nn.conv2d(output1,self.W3,strides=[1,1,1,1],padding='VALID')+tf.nn.conv2d(output2,self.W4,strides=[1,1,1,1],padding='VALID')
                                self.output= nonlinearity(tf.layers.batch_normalization(output3,training=test))
                        tf.add_to_collection("regularizable",self.W2)
                tf.add_to_collection("regularizable",self.W)



class Conv2DLayer:
        def __init__(self,incoming,n_filters,filter_shape,test,stride=1,pad='valid',mode='CONSTANT',nonlinearity=tf.nn.relu,bn=0,bias=1):
		print incoming.output_shape
                if(pad=='valid' or filter_shape==1):
                        padded_input = incoming.output
                        self.output_shape = (incoming.output_shape[0],(incoming.output_shape[1]-filter_shape+1)/stride,(incoming.output_shape[1]-filter_shape+1)/stride,n_filters)
                elif(pad=='same'):
                        assert(filter_shape%2 ==1)
                        p = (filter_shape-1)/2
                        padded_input = tf.pad(incoming.output,[[0,0],[p,p],[p,p],[0,0]],mode=mode)
                        self.output_shape = (incoming.output_shape[0],incoming.output_shape[1]/stride,incoming.output_shape[2]/stride,n_filters)
                else:
                        p = filter_shape-1
                        padded_input = tf.pad(incoming.output,[[0,0],[p,p],[p,p],[0,0]],mode=mode)
                        self.output_shape = (incoming.output_shape[0],(incoming.output_shape[1]+filter_shape-1)/stride,(incoming.output_shape[1]+filter_shape-1)/stride,n_filters)
                init       = tf.contrib.layers.xavier_initializer(uniform=True)
                self.W     = tf.Variable(init((filter_shape,filter_shape,incoming.output_shape[3],n_filters)),name='W_conv2d',trainable=True)
                output1    = tf.nn.conv2d(padded_input,self.W,strides=[1,stride,stride,1],padding='VALID')
                if(bn==0):
	                if(bias==0):
	                        self.b = tf.Variable(tf.zeros((1,1,1,n_filters)),name='b_dense',trainable=False)
	                if(bias==1):
	                        self.b = tf.Variable(tf.zeros((1,1,1,n_filters)),name='b_dense',trainable=True)
	                if(bias=='norm'):
	                        self.b = -0.5*tf.reduce_sum(self.W*self.W,axis=[0,1,2],keep_dims=True)
                       	self.output = nonlinearity(output1+self.b)
                else:
                        self.output = nonlinearity(tf.layers.batch_normalization(output1,training=test,fused=True))#bias
#			if(bias==0):
#				A=tf.get_collection_ref(tf.GraphKeys.UPDATE_OPS)
#				A.pop(-2)
                tf.add_to_collection("regularizable",self.W)
		print self.output_shape



def NN(x):
	return tf.nn.softplus(x)

class NNConv2DLayer:
        def __init__(self,incoming,n_filters,filter_shape,test,stride=1,pad='valid',mode='CONSTANT',nonlinearity=tf.nn.relu,bn=0,bias=1):
                print incoming.output_shape
                if(pad=='valid' or filter_shape==1):
                        padded_input = incoming.output
                        self.output_shape = (incoming.output_shape[0],(incoming.output_shape[1]-filter_shape+1)/stride,(incoming.output_shape[1]-filter_shape+1)/stride,n_filters)
                elif(pad=='same'):
                        assert(filter_shape%2 ==1)
                        p = (filter_shape-1)/2
                        padded_input = tf.pad(incoming.output,[[0,0],[p,p],[p,p],[0,0]],mode=mode)
                        self.output_shape = (incoming.output_shape[0],incoming.output_shape[1]/stride,incoming.output_shape[2]/stride,n_filters)
                else:
                        p = filter_shape-1
                        padded_input = tf.pad(incoming.output,[[0,0],[p,p],[p,p],[0,0]],mode=mode)
                        self.output_shape = (incoming.output_shape[0],(incoming.output_shape[1]+filter_shape-1)/stride,(incoming.output_shape[1]+filter_shape-1)/stride,n_filters)
                init       = tf.contrib.layers.xavier_initializer(uniform=True)
                self.W     = tf.Variable(init((filter_shape,filter_shape,incoming.output_shape[3],n_filters)),name='W_conv2d',trainable=True)
                output1    = tf.nn.conv2d(tf.layers.batch_normalization(padded_input,training=test,fused=True,center=bias),self.W,strides=[1,stride,stride,1],padding='VALID')
                if(bn==0):
                        if(bias):
                                self.b      = tf.Variable(tf.zeros((1,1,1,n_filters)),name='b_conv',trainable=True)
                                self.output = nonlinearity(output1+self.b)
                        else:
                                self.output = nonlinearity(output1)
                else:
                        self.output = nonlinearity(output1)#nonlinearity(tf.layers.batch_normalization(output1,training=test,fused=True,center=bias))
                        if(bias==0):
                                A=tf.get_collection_ref(tf.GraphKeys.UPDATE_OPS)
                                A.pop(-2)
                tf.add_to_collection("regularizable",self.W)
                print self.output_shape




class Block:
	def __init__(self,incoming,n_filters1,stride,test,bias,bn=1,nonlinearity=tf.nn.relu):
		input_shape = incoming.output_shape
		input= incoming.output
                self.output_shape = (incoming.output_shape[0],incoming.output_shape[1]/stride,incoming.output_shape[2]/stride,n_filters1)
                init        = tf.contrib.layers.xavier_initializer(uniform=True)
                self.W1     = tf.Variable(init((3,3,incoming.output_shape[3],n_filters1)),name='W1_',trainable=True)
                self.W11    = tf.Variable(init((3,3,n_filters1,n_filters1)),name='W11_',trainable=True)
		if(bn):
			input1 = nonlinearity(tf.layers.batch_normalization(tf.nn.conv2d(input,self.W1,strides=[1,1,1,1],padding='SAME'),training=test,fused=True))
                        input2 = tf.layers.batch_normalization(tf.nn.conv2d(input1,self.W11,strides=[1,1,1,1],padding='SAME'),training=test,fused=True)
#			input2 = tf.nn.conv2d(tf.nn.relu(tf.layers.batch_normalization(input1,training=test,fused=True)),self.W11,strides=[1,1,1,1],padding='SAME')
		else:
			self.b1 = tf.Variable(init((1,1,1,n_filters1)),name='b1_',trainable=True)
#                        self.b11 = tf.Variable(init((1,1,1,n_filters1)),name='b1_',trainable=True)
                        input1 = nonlinearity(tf.nn.conv2d(input,self.W1,strides=[1,1,1,1],padding='SAME')+self.b1)
                        input2 = tf.nn.conv2d(input1,self.W11,strides=[1,1,1,1],padding='SAME')
                if(stride==2):
                    self.W0     = tf.Variable(init((1,1,incoming.output_shape[3],n_filters1)),name='W1_',trainable=True)
                    input_output = tf.nn.conv2d(input,self.W0,strides=[1,1,1,1],padding='VALID')
                    self.output = input2+input_output
                    self.output = tf.nn.avg_pool(self.output,[1,2,2,1],[1,2,2,1],'VALID')
                else:
                    self.output = input2+input
		print self.output_shape,self.output.get_shape()


class Conv3DLayer:
        def __init__(self,incoming,n_filters,filter_shape,test,stride=1,pad='valid',mode='CONSTANT',nonlinearity=tf.nn.relu,bn=0):
                if(pad=='valid' or filter_shape==1):
                        padded_input = incoming.output
                        self.output_shape = (incoming.output_shape[0],(incoming.output_shape[1]-filter_shape+1)/stride,(incoming.output_shape[1]-filter_shape+1)/stride,n_filters)
                elif(pad=='same'):
                        assert(filter_shape%2 ==1)
                        p = (filter_shape-1)/2
                        padded_input = tf.pad(incoming.output,[[0,0],[p,p],[p,p],[0,0]],mode=mode)
                        self.output_shape = (incoming.output_shape[0],incoming.output_shape[1]/stride,incoming.output_shape[2]/stride,n_filters)
                else:
                        p = filter_shape-1
                        padded_input = tf.pad(incoming.output,[[0,0],[p,p],[p,p],[0,0]],mode=mode)
                        self.output_shape = (incoming.output_shape[0],(incoming.output_shape[1]+filter_shape-1)/stride,(incoming.output_shape[1]+filter_shape-1)/stride,n_filters)
		padded_input = tf.expand_dims(tf.transpose(padded_input,[0,3,1,2]),-1)
#		padded_input = tf.concat([padded_input,padded_input],1)
                init         = tf.contrib.layers.xavier_initializer(uniform=True)
                self.W       = tf.Variable(init((max(incoming.output_shape[3]-3,1),filter_shape,filter_shape,1,n_filters)),name='W_conv3d',trainable=True)
                padded_input = tf.nn.conv3d(padded_input,self.W,strides=[1,stride,stride,1,1],padding='VALID')
		padded_input = tf.reduce_max(padded_input,axis=1)
                if(bn==0):
                        self.b      = tf.Variable(tf.zeros((1,1,1,n_filters)),name='b_conv',trainable=True)
                        self.output = nonlinearity(padded_input+self.b)
                else:
                        self.output = nonlinearity(tf.layers.batch_normalization(padded_input,training=test))
                tf.add_to_collection("regularizable",self.W)







 



class NConv2DLayer:
        def __init__(self,incoming,n_filters,filter_shape,test,stride=1,pad='valid',mode='CONSTANT',nonlinearity=tf.nn.relu,bn=1):
                if(pad=='valid' or filter_shape==1):
                        padded_input = incoming.output
                        self.output_shape = (incoming.output_shape[0],(incoming.output_shape[1]-filter_shape+1)/stride,(incoming.output_shape[1]-filter_shape+1)/stride,n_filters)
                elif(pad=='same'):
                        assert(filter_shape%2 ==1)
                        p = (filter_shape-1)/2
                        padded_input = tf.pad(incoming.output,[[0,0],[p,p],[p,p],[0,0]],mode=mode)
                        self.output_shape = (incoming.output_shape[0],incoming.output_shape[1]/stride,incoming.output_shape[2]/stride,n_filters)
                else:
                        p = filter_shape-1
                        padded_input = tf.pad(incoming.output,[[0,0],[p,p],[p,p],[0,0]],mode=mode)
                        self.output_shape = (incoming.output_shape[0],(incoming.output_shape[1]+filter_shape-1)/stride,(incoming.output_shape[1]+filter_shape-1)/stride,n_filters)
                init       = tf.contrib.layers.xavier_initializer(uniform=True)
                self.W     = tf.Variable(init((filter_shape,filter_shape,incoming.output_shape[3],n_filters)),name='W_conv3d',trainable=True)
#                self.W2    = tf.Variable(init((1,1,n_filters*3,n_filters)),name='W_conv3d',trainable=True)
#		apodization= hamming(filter_shape)
#		apodization= apodization.reshape((-1,1))*apodization.reshape((1,-1))
#		apodization/=apodization.sum()
#		apodization = tf.Variable(apodization.astype('float32').reshape((filter_shape,filter_shape,1,1)),trainable=False)
                output_conv= tf.nn.conv2d(padded_input,self.W,strides=[1,stride,stride,1],padding='VALID')
#		norms      = tf.nn.conv2d(padded_input*padded_input,tf.ones((filter_shape,filter_shape,incoming.output_shape[3],n_filters)),strides=[1,stride,stride,1],padding='VALID')
                if(bn==0):
                        self.b      = tf.Variable(tf.zeros((1,1,1,n_filters)),name='b_conv',trainable=True)
                        self.output = nonlinearity(output_conv+self.b)
                else:   
                        self.b      = tf.Variable(tf.zeros((1,1,1,n_filters)),name='b_conv',trainable=True)
			caca=nonlinearity(tf.abs(tf.layers.batch_normalization(output_conv,training=test,center=False))+self.b)
                        self.output = caca





class MConv2DLayer:
        def __init__(self,incoming,n_filters,filter_shape,test,stride=1,pad='valid',mode='CONSTANT',nonlinearity=tf.nn.relu,bn=1):
                if(pad=='valid' or filter_shape==1):
                        padded_input = incoming.output
                        self.output_shape = (incoming.output_shape[0],(incoming.output_shape[1]-filter_shape+1)/stride,(incoming.output_shape[1]-filter_shape+1)/stride,n_filters)
                elif(pad=='same'):
                        assert(filter_shape%2 ==1)
                        p = (filter_shape-1)/2
                        padded_input = tf.pad(incoming.output,[[0,0],[p,p],[p,p],[0,0]],mode=mode)
                        self.output_shape = (incoming.output_shape[0],incoming.output_shape[1]/stride,incoming.output_shape[2]/stride,n_filters)
                else:
                        p = filter_shape-1
                        padded_input = tf.pad(incoming.output,[[0,0],[p,p],[p,p],[0,0]],mode=mode)
                        self.output_shape = (incoming.output_shape[0],(incoming.output_shape[1]+filter_shape-1)/stride,(incoming.output_shape[1]+filter_shape-1)/stride,n_filters)
                init       = tf.contrib.layers.xavier_initializer(uniform=True)
                self.W     = tf.Variable(init((filter_shape,filter_shape,incoming.output_shape[3],n_filters)),name='W_conv2d',trainable=True)
                self.b     = tf.Variable(init((1,1,1,n_filters)),name='b',trainable=True)
                output_conv= tf.nn.conv2d(padded_input,self.W,strides=[1,stride,stride,1],padding='VALID')
		mask       = tf.greater((output_conv+self.b)*(2*output_conv-tf.reduce_sum(self.W*self.W,axis=[0,1,2],keep_dims=True)*(output_conv+self.b)),0)
		output     = output_conv*tf.cast(mask,tf.float32)+self.b
                self.output= tf.layers.batch_normalization(output,training=test)









