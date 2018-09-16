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





def doit(k):
    f=open('/mnt/project2/rb42Data/PMASO/BASE_EXP/exp_unsup_'+str(k)+'.pkl','rb')
    LOSSES0,reconstruction0,x0,samplesclass00,samplesclass10,samples10,W0,sigmas0,pred0=cPickle.load(f)
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
    savefig('BASE_EXP/unsup/reconstruction_'+str(k)+'.png')
    close()

#        figure(figsize=(15,3))
#        classes=[0]
#        plotclasses(classes,samplesclass1[s])
#        tight_layout()
#        savefig('BASE_EXP/fc_threeclass1_unsup'+str(unsup)+'_neurons'+str(neurons)+'_step'+str(s)+'.png')
#        close()

    figure(figsize=(15,15))
    classes=range(32)
    plotclasses(classes,samplesclass10)
    tight_layout()
    savefig('BASE_EXP/unsup/tenclass1_'+str(k)+'.png')
    close()


#        figure(figsize=(15,3))
#        classes=[0]
#        plotclasses(classes,samplesclass0[s])
#        tight_layout()
#        savefig('BASE_EXP/fc_threeclass0_unsup'+str(unsup)+'_neurons'+str(neurons)+'_step'+str(s)+'.png')
#        close()

    figure(figsize=(15,15))
    classes=range(32)
    plotclasses(classes,samplesclass00)
    tight_layout()
    savefig('BASE_EXP/unsup/tenclass0_'+str(k)+'.png')
    close()
    return LOSSES0




l=doit(32)


