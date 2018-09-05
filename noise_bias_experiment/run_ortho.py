from pylab import *
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

execfile('utils.py')
execfile('models.py')
execfile('lasagne_tf.py')

DATASET = sys.argv[-1]
lr      = 0.001

B = int(sys.argv[-3])

if(int(sys.argv[-2])==0):
	m = smallCNN
	m_name = 'smallCNN'
elif(int(sys.argv[-2])==1):
	m = largeCNN
	m_name = 'largeCNN'
elif(int(sys.argv[-2])==2):
        m = resnet_large
        m_name = 'densedouble'


x_train,x_test,y_train,y_test,c,n_epochs,input_shape=load_utility(DATASET)

for kk in xrange(10):
	for std in linspace(0,1,5):
        	name = DATASET+'_'+m_name+'_lr'+str(lr)+'_run'+str(kk)+'_std'+str(std)+'_b'+str(B)
		model1  = DNNClassifier(input_shape,m(bn=0,n_classes=c,bias=B),lr=lr,std=std)
		train_accu,test_accu = model1.fit(x_train,y_train,x_test,y_test,n_epochs=n_epochs)
		f = open('./'+name,'wb')
		cPickle.dump([train_accu,test_accu],f)
		f.close()




