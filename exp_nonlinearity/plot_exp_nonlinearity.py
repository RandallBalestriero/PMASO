import cPickle
from pylab import *
import glob
import matplotlib as mpl
label_size = 13
mpl.rcParams['xtick.labelsize'] = label_size+10
mpl.rcParams['ytick.labelsize'] = label_size

fs=15

import os
SAVE_DIR = os.environ['SAVE_DIR']





def doit(DATASET,sigmass,leakiness,plot=1):
    f=open(SAVE_DIR+'exp_nonlinearity_'+DATASET+'_'+sigmass+'_'+leakiness+'.pkl','rb')
    LOSSES0,reconstruction0,x0,samplesclass00,samplesclass10,samples10,W0,sigmas0=cPickle.load(f)
    print LOSSES0
    f.close()
    LL = []
    if(plot==0):return LOSSES0
    for i in xrange(15):
	figure(figsize=(2,2))
        imshow(x0[i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
        xticks([])
        yticks([])
        tight_layout()
        savefig('../BASE_EXP/nonlinearity/x_'+sigmass+'_'+leakiness+'_'+str(i)+'.png')
	close()
        figure(figsize=(2,2))
        imshow(reconstruction0[i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
        xticks([])
        yticks([])
        tight_layout()
        savefig('../BASE_EXP/nonlinearity/reconstruction_'+sigmass+'_'+leakiness+'_'+str(i)+'.png')
	close()
    for k in xrange(10):
        for i in xrange(20):
	    figure(figsize=(2,2))
            imshow(samplesclass00[k][i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])
	    tight_layout()
            savefig('../BASE_EXP/nonlinearity/samples0_'+sigmass+'_'+leakiness+'_'+str(k)+'_'+str(i)+'.png')
	    close()	
            figure(figsize=(2,2))
            imshow(samplesclass10[k][i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])
            tight_layout()
            savefig('../BASE_EXP/nonlinearity/samples1_'+sigmass+'_'+leakiness+'_'+str(k)+'_'+str(i)+'.png')
	    close()
    return LOSSES0


#l1=doit('MNIST','global','-1',1)
#l1=doit('MNIST','global','0',1)
#l1=doit('MNIST','global','0.01',1)
#l1=doit('MNIST','global','None',1)

#l1=doit('flippedMNIST','global','-1',1)
l1=doit('MNIST','global','0',1)
#l1=doit('flippedMNIST','global','0.01',1)
#l1=doit('flippedMNIST','global','None',1)


print l1
awd
#l2=doit('global','0',0)
#l3=doit('global','0.01',0)
#l4=doit('global','None',0)


plot(range(len(l1)),l1,c='k')
plot(range(len(l1))[3::20],l1[3::20],linestyle='None',c='k',marker='x')

plot(range(len(l1)),l1,c='k')
plot(range(len(l1))[3::20],l1[3::20],linestyle='None',c='k',marker='o')

plot(range(len(l1)),l1,c='k')
plot(range(len(l1))[3::20],l1[3::20],linestyle='None',c='k',marker='s')

plot(range(len(l1)),l1,c='k')
#plot(range(len(l1))[3::20],l1[3::20],c='k',ls='x')

tight_layout()

savefig('../BASE_EXP/nonlinearity/losses.png')


