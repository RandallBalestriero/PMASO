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





def doit():
    f=open('/mnt/project2/rb42Data/PMASO/BASE_EXP/exp_nonlinearity_relu.pkl','rb')
    LOSSES0,reconstruction0,x0,samplesclass00,samplesclass10,samples10,W0,sigmas0=cPickle.load(f)
    f.close()
    f=open('/mnt/project2/rb42Data/PMASO/BASE_EXP/exp_nonlinearity_abs.pkl','rb')
    LOSSES1,reconstruction1,x1,samplesclass01,samplesclass11,samples11,W1,sigmas1=cPickle.load(f)
    f.close()
    f=open('/mnt/project2/rb42Data/PMASO/BASE_EXP/exp_nonlinearity_None.pkl','rb')
    LOSSES2,reconstruction2,x2,samplesclass02,samplesclass12,samples12,W2,sigmas2=cPickle.load(f)
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
    savefig('BASE_EXP/nonlinearity/reconstruction_0.png')
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
    savefig('BASE_EXP/nonlinearity/reconstruction_1.png')
    close()
    figure(figsize=(15,3))
    for i in xrange(6):
        subplot(2,6,1+i)
        imshow(x1[i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
        xticks([])
        yticks([])
        subplot(2,6,7+i)
        imshow(reconstruction2[i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
        xticks([])
        yticks([])
    tight_layout()
    savefig('BASE_EXP/nonlinearity/reconstruction_2.png')
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
    savefig('BASE_EXP/nonlinearity/tenclass1_0.png')
    close()

    figure(figsize=(15,15))
    classes=range(10)
    plotclasses(classes,samplesclass11)
    tight_layout()
    savefig('BASE_EXP/nonlinearity/tenclass1_1.png')
    close()

    figure(figsize=(15,15))
    classes=range(10)
    plotclasses(classes,samplesclass12)
    tight_layout()
    savefig('BASE_EXP/nonlinearity/tenclass1_2.png')
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
    savefig('BASE_EXP/nonlinearity/tenclass0_0.png')
    close()
    figure(figsize=(15,15))
    classes=range(10)
    plotclasses(classes,samplesclass01)
    tight_layout()
    savefig('BASE_EXP/nonlinearity/tenclass0_1.png')
    close()
    figure(figsize=(15,15))
    classes=range(10)
    plotclasses(classes,samplesclass02)
    tight_layout()
    savefig('BASE_EXP/nonlinearity/tenclass0_2.png')
    close()

    figure(figsize=(15,15))
    plot(LOSSES0,c='black')
    plot(LOSSES1,c='blue')
    plot(LOSSES2,c='red')
    savefig('BASE_EXP/nonlinearity/losses.png')
    close()



doit()


