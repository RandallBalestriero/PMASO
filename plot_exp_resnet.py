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



def plotclasses(classes,samplesclass1):
    for i,k in zip(range(len(classes)),classes):
        for j in xrange(10):
            subplot(len(classes),10,1+i*10+j)
            imshow(samplesclass1[k][j,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])





def doit(l):
    f=open(SAVE_DIR+'exp_resnet_0_'+str(l)+'.pkl','rb')
    LOSSES0,reconstruction0,x0,samplesclass00,samplesclass10,samples10,W0,sigmas0=cPickle.load(f)
    f.close()
    f=open(SAVE_DIR+'exp_resnet_1_'+str(l)+'.pkl','rb')
    LOSSES1,reconstruction1,x1,samplesclass01,samplesclass11,samples11,W1,sigmas1=cPickle.load(f)
    f.close()
    figure(figsize=(15,3))
    for i in xrange(6):
        subplot(2,6,1+i)
        imshow(x0[i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
        xticks([])
        yticks([])
        subplot(2,6,7+i)
        imshow(reconstruction0[i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
        xticks([])
        yticks([])
    tight_layout()
    savefig('BASE_EXP/resnet/reconstruction_0_'+str(l)+'.png')
    close()
    figure(figsize=(15,3))
    for i in xrange(6):
        subplot(2,6,1+i)
        imshow(x1[i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
        xticks([])
        yticks([])
        subplot(2,6,7+i)
        imshow(reconstruction1[i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
        xticks([])
        yticks([])
    tight_layout()
    savefig('BASE_EXP/resnet/reconstruction_1_'+str(l)+'.png')
    close()

#        figure(figsize=(15,3))
#        classes=[0]
#        plotclasses(classes,samplesclass1[s])
#        tight_layout()
#        savefig('BASE_EXP/fc_threeclass1_unsup'+str(unsup)+'_neurons'+str(neurons)+'_step'+str(s)+'.png')
#        close()

    figure(figsize=(15,15))
    classes=range(10)
    plotclasses(classes,samplesclass10)
    tight_layout()
    savefig('BASE_EXP/resnet/tenclass1_0_'+str(l)+'.png')
    close()

    figure(figsize=(15,15))
    classes=range(10)
    plotclasses(classes,samplesclass11)
    tight_layout()
    savefig('BASE_EXP/resnet/tenclass1_1_'+str(l)+'.png')
    close()


#        figure(figsize=(15,3))
#        classes=[0]
#        plotclasses(classes,samplesclass0[s])
#        tight_layout()
#        savefig('BASE_EXP/fc_threeclass0_unsup'+str(unsup)+'_neurons'+str(neurons)+'_step'+str(s)+'.png')
#        close()

    figure(figsize=(15,15))
    classes=range(10)
    plotclasses(classes,samplesclass00)
    tight_layout()
    savefig('BASE_EXP/resnet/tenclass0_0_'+str(l)+'.png')
    close()
    figure(figsize=(15,15))
    classes=range(10)
    plotclasses(classes,samplesclass01)
    tight_layout()
    savefig('BASE_EXP/resnet/tenclass0_1_'+str(l)+'.png')
    close()

    figure(figsize=(15,15))
    plot(LOSSES0,c='k')
    plot(LOSSES1,c='r')
    savefig('BASE_EXP/resnet/losses_'+str(l)+'.png')
    close()



#doit(5)
doit(1)
#doit(3)
doit(2)

#doit(1,1,'local','none','none')
#doit(1,3,'local','none','none')




