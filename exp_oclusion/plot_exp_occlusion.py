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


def process(f_name):
    f= open(f_name,'rb')
    loss,reconstruction,x,X,y,pred = cPickle.load(f)
    f.close()
    print "accuracy",[mean((y==argmax(p,1)).astype('float32')) for p in [pred]]
    loss = unique(concatenate(loss,axis=0))
    reconstruction = asarray(reconstruction)
    reconstruction_loss = ((reconstruction-X.reshape((1,1000,28,28,1)))**2).sum((1,2,3,4))/1000
    return loss,reconstruction_loss,reconstruction,x,X



def plot_(loss,reconstruction_loss,reconstruction,x,X,name,K=10):
    subplot(121)
    plot(loss)
    subplot(122)
    plot(reconstruction_loss)
    tight_layout()
    savefig(name+'_loss.png')
    close()
    for i in xrange(len(reconstruction)):
	figure(figsize=(2*K,2))
	for k in xrange(K):
	    subplot(3,K,1+k+K*2)
	    imshow(reconstruction[i][k,:,:,0])
	    xticks([])
	    yticks([])
	    subplot(3,K,1+k+K)
	    imshow(x[k,:,:,0])
	    xticks([])
	    yticks([])
	    subplot(3,K,1+k)
	    imshow(x[k,:,:,0])
	    xticks([])
	    yticks([])
	tight_layout()
	savefig(name+'_'+str(i)+'.png')
	close()
	


files = glob.glob(SAVE_DIR+'exp_oclusion2_0.1_global_2.pkl')
for f in files:
    print f
    loss,reconstruction_loss,reconstruction,x,X=process(f)
    plot_(loss,reconstruction_loss,reconstruction,x,X,'./BASE_EXP/occlusion/'+f.split('/')[-1][:-4],10)


