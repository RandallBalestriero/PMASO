import cPickle
from pylab import *
import glob
import matplotlib as mpl


import os
SAVE_DIR = os.environ['SAVE_DIR']

label_size = 13
mpl.rcParams['xtick.labelsize'] = label_size+10
mpl.rcParams['ytick.labelsize'] = label_size

fs=15

def normalize(x):
        return (x-x.min())/(x.max()-x.min())


def process(f_name):
    f= open(f_name,'rb')
    loss,reconstruction,x,X,y,pred = cPickle.load(f)
    f.close()
    print "accuracy",[mean((y==argmax(p,1)).astype('float32')) for p in [pred]]
    loss = unique(concatenate(loss,axis=0))
    reconstruction = asarray(reconstruction)
    reconstruction_loss = ((reconstruction-X.reshape((1,1000,28,28,1)))**2).sum((1,2,3,4))/1000
    return loss,reconstruction_loss,reconstruction,x,X



def plot_(DATASET,CLASS,MODEL_TYPE,OCLUSION_TYPE,OCLUSION_SPEC,KNOWN_Y,pl=0):
    f= open(SAVE_DIR+'exp_oclusion_'+DATASET+'_'+str(CLASS)+'_'+MODEL_TYPE+'_'+OCLUSION_TYPE+'_'+str(OCLUSION_SPEC)+'_'+str(KNOWN_Y)+'.pkl','rb')
    loss,reconstruction,x,X,Y,preds = cPickle.load(f)
    f.close()
    LLL = []
    for i in xrange(len(X)):
	LLL.append(((reconstruction[-1][i]-X[i])**2).sum())
    if(pl==0): return mean(LLL),std(LLL),loss[-1][-1]
    for i in xrange(130):
	if(DATASET=='MNIST'):
            figure(figsize=(2,2))
	    imshow(reconstruction[-1][i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])
            tight_layout()
	    savefig('../BASE_EXP/oclusion/reconstruction_'+DATASET+'_'+str(CLASS)+'_'+MODEL_TYPE+'_'+OCLUSION_TYPE+'_'+str(OCLUSION_SPEC)+'_'+str(KNOWN_Y)+'_'+str(i)+'.png')
	    close()
            figure(figsize=(2,2))
            imshow(x[i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])
            tight_layout()
            savefig('../BASE_EXP/oclusion/x_'+DATASET+'_'+str(CLASS)+'_'+MODEL_TYPE+'_'+OCLUSION_TYPE+'_'+str(OCLUSION_SPEC)+'_'+str(KNOWN_Y)+'_'+str(i)+'.png')
	    close()
            figure(figsize=(2,2))
            imshow(X[i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])
            tight_layout()
            savefig('../BASE_EXP/oclusion/X_'+DATASET+'_'+str(CLASS)+'_'+MODEL_TYPE+'_'+OCLUSION_TYPE+'_'+str(OCLUSION_SPEC)+'_'+str(KNOWN_Y)+'_'+str(i)+'.png')
	    close()
	else:
            figure(figsize=(2,2))
            imshow(normalize(reconstruction[-1][i]),aspect='auto',interpolation='nearest')
            xticks([])
            yticks([])
            tight_layout()
            savefig('../BASE_EXP/oclusion/reconstruction_'+DATASET+'_'+str(CLASS)+'_'+MODEL_TYPE+'_'+OCLUSION_TYPE+'_'+str(OCLUSION_SPEC)+'_'+str(KNOWN_Y)+'_'+str(i)+'.png')
            close()
            figure(figsize=(2,2))
            imshow(normalize(x[i]),aspect='auto',interpolation='nearest')
            xticks([])
            yticks([])
            tight_layout()
            savefig('../BASE_EXP/oclusion/x_'+DATASET+'_'+str(CLASS)+'_'+MODEL_TYPE+'_'+OCLUSION_TYPE+'_'+str(OCLUSION_SPEC)+'_'+str(KNOWN_Y)+'_'+str(i)+'.png')
            close()
            figure(figsize=(2,2))
            imshow(normalize(X[i]),aspect='auto',interpolation='nearest')
            xticks([])
            yticks([])
            tight_layout()
            savefig('../BASE_EXP/oclusion/X_'+DATASET+'_'+str(CLASS)+'_'+MODEL_TYPE+'_'+OCLUSION_TYPE+'_'+str(OCLUSION_SPEC)+'_'+str(KNOWN_Y)+'_'+str(i)+'.png')
            close()

    return mean(LLL),std(LLL),loss[-1][-1]


for known_y in ['0']:
	for TYPE in ['pixel','box']:
		if(TYPE=='pixel'):
			1
#			for SPEC in ['0.05']:
#				print 'MLP MNIST knowny',known_y,'pixel ',SPEC,plot_('CIFAR','4','MLP',TYPE,SPEC,known_y,1)
		else:
			for SPEC in ['0','1','2','3','4','5']:
                                print 'MLP MNIST knowny',known_y,'box ',SPEC,plot_('CIFAR',SPEC,'MLP',TYPE,'20.0',known_y,1)

