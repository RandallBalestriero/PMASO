import cPickle
from pylab import *
import glob
import matplotlib as mpl
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





def doit(unsup,neurons):
    f=open('BASE_EXP/exp_fc_'+str(unsup)+'_'+str(neurons)+'.pkl','rb')
    LOSSES,reconstruction,x,samplesclass1,samples1,W,b=cPickle.load(f)
    f.close()

    figure(figsize=(15,3))
    for i in xrange(6):
        subplot(2,6,1+i)
        imshow(x[i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
        xticks([])
        yticks([])
        subplot(2,6,7+i)
        imshow(reconstruction[i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
        xticks([])
        yticks([])

    tight_layout()
    savefig('BASE_EXP/fc_reconstruction_unsup'+str(unsup)+'_neurons'+str(neurons)+'.png')
    close()

    figure(figsize=(15,3))
    classes=[0,3]
    plotclasses(classes,samplesclass1)
    tight_layout()
    savefig('BASE_EXP/fc_threeclass_unsup'+str(unsup)+'_neurons'+str(neurons)+'.png')
    close()

    figure(figsize=(15,15))
    classes=range(10)
    plotclasses(classes,samplesclass1)
    tight_layout()
    savefig('BASE_EXP/fc_tenclass_unsup'+str(unsup)+'_neurons'+str(neurons)+'.png')
    close()

    figure(figsize=(15,3))
    for i in xrange(10):
        subplot(2,10,i+1)
        imshow(reshape(W[i,0],(28,28)),aspect='auto',cmap='Greys',interpolation='nearest')
        xticks([])
        yticks([])
        subplot(2,10,i+11)
        imshow(reshape(W[i,1],(28,28)),aspect='auto',cmap='Greys',interpolation='nearest')
        xticks([])
        yticks([])

    tight_layout()
    savefig('BASE_EXP/fc_W_unsup'+str(unsup)+'_neurons'+str(neurons)+'.png')
    close()


doit(0,16)
doit(0,64)
doit(1,16)
doit(1,64)

#show()




