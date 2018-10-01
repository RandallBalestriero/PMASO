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





def doit(per_layer,randomm,mp_opt,leakiness):
    f=open(SAVE_DIR+'exp_orderEM_'+str(per_layer)+'_'+str(randomm)+'_'+str(mp_opt)+'_'+leakiness+'.pkl')
    print 'exp_resnet_'+neurons+'_'+layers+'_'+residual+'_'+sigmas
    LOSSES,reconstruction,x0,samplesclass0,samplesclass1,samples1,params=cPickle.load(f)
    f.close()
    for k in [3]:
	for i in xrange(10):
            figure(figsize=(2,2))
            imshow(samplesclass0[k][i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])
            tight_layout()
            savefig('../BASE_EXP/orderEM/samples0_'+neurons+'_'+layers+'_'+residual+'_'+sigmas+'_'+str(k)+'_'+str(i)+'.png')
	    close()
            figure(figsize=(2,2))
            imshow(samplesclass1[k][i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])
            tight_layout()
            savefig('../BASE_EXP/orderEM/samples1_'+neurons+'_'+layers+'_'+residual+'_'+sigmas+'_'+str(k)+'_'+str(i)+'.png')
	    close()
    return LOSSES


for mp_opt in [0,1,2,3]:
    for leakiness in ['None',0]:
	for per_layer in [0,1]:
	    for randomm in [0,1]:
		lo=doit(per_layer,randomm,mp_opt,leakiness)
	        print lo





