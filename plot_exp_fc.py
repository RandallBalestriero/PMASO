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





def doit(unsup,neurons,BN,U):
    f=open('/mnt/project2/rb42Data/PMASO/BASE_EXP/exp_fc_'+str(unsup)+'_'+str(neurons)+'_'+str(BN)+'_'+str(U)+'.pkl','rb')
    LOSSES,reconstruction,x,samplesclass0,samplesclass1,samples1,W,b,sigmas=cPickle.load(f)
    f.close()

    for s in xrange(len(reconstruction)):
        figure(figsize=(15,3))
        for i in xrange(6):
            subplot(2,6,1+i)
            imshow(x[i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])
            subplot(2,6,7+i)
            imshow(reconstruction[s][i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])
        tight_layout()
        savefig('BASE_EXP/fc_reconstruction_unsup'+str(unsup)+'_neurons'+str(neurons)+'_'+str(BN)+'_'+str(U)+'_step'+str(s)+'.png')
        close()

#        figure(figsize=(15,3))
#        classes=[0]
#        plotclasses(classes,samplesclass1[s])
#        tight_layout()
#        savefig('BASE_EXP/fc_threeclass1_unsup'+str(unsup)+'_neurons'+str(neurons)+'_step'+str(s)+'.png')
#        close()

        figure(figsize=(15,15))
        classes=range(10)
        plotclasses(classes,samplesclass1[s])
        tight_layout()
        savefig('BASE_EXP/fc_tenclass1_unsup'+str(unsup)+'_neurons'+str(neurons)+'_'+str(BN)+'_'+str(U)+'_step'+str(s)+'.png')
        close()

#        figure(figsize=(15,3))
#        classes=[0]
#        plotclasses(classes,samplesclass0[s])
#        tight_layout()
#        savefig('BASE_EXP/fc_threeclass0_unsup'+str(unsup)+'_neurons'+str(neurons)+'_step'+str(s)+'.png')
#        close()

        figure(figsize=(15,15))
        classes=range(10)
        plotclasses(classes,samplesclass0[s])
        tight_layout()
        savefig('BASE_EXP/fc_tenclass0_unsup'+str(unsup)+'_neurons'+str(neurons)+'_'+str(BN)+'_'+str(U)+'_step'+str(s)+'.png')
        close()

    figure(figsize=(15,3))
    for i in xrange(10):
        subplot(2,10,i+1)
        imshow(reshape(W[0][i,0],(28,28)),vmin=W[0][i].min(),vmax=W[0][i].max(),aspect='auto',cmap='Greys',interpolation='nearest')
        xticks([])
        yticks([])
        subplot(2,10,i+11)
        imshow(reshape(W[0][i,1],(28,28)),vmin=W[0][i].min(),vmax=W[0][i].max(),aspect='auto',cmap='Greys',interpolation='nearest')
        xticks([])
        yticks([])

    tight_layout()
    savefig('BASE_EXP/fc_W_unsup'+str(unsup)+'_neurons'+str(neurons)+'_'+str(BN)+'_'+str(U)+'.png')
    close()


doit(0,2,1,0)
doit(0,2,0,1)
doit(0,2,1,1)
#doit(0,4,1,0)

#doit(0,64,0,0)
#doit(0,64,0,1)
#doit(0,64,1,1)


#doit(0,64)
#doit(1,16)
#doit(1,64)

#show()




