import cPickle
from pylab import *
import glob
import matplotlib as mpl
label_size = 13
mpl.rcParams['xtick.labelsize'] = label_size+10
mpl.rcParams['ytick.labelsize'] = label_size

fs=15



def plotclasses(classes):
    for i,k in zip(range(len(classes)),classes):
        for j in xrange(10):
            subplot(len(classes),10,1+i*10+j)
            imshow(samplesclass1[k][j,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])





f=open('BASE_EXP/exp_fc.pkl','rb')
LOSSES,reconstruction,x,samplesclass1,samples1,W=cPickle.load(f)
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
savefig('BASE_EXP/fc_reconstruction.png')
close()




figure(figsize=(15,5))
classes=[0,1,3]
plotclasses(classes)
tight_layout()
savefig('BASE_EXP/fc_threeclass.png')
close()


figure(figsize=(15,15))
classes=range(10)
plotclasses(classes)
tight_layout()
savefig('BASE_EXP/fc_tenclass.png')
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
savefig('BASE_EXP/fc_W.png')
close()


##################### UNSUP

f=open('BASE_EXP/exp_fc_unsup.pkl','rb')
LOSSES,reconstruction,x,samplesclass1,samples1,W=cPickle.load(f)
f.close()

figure(figsize=(15,5))
classes=[0,1,3]
plotclasses(classes)
tight_layout()
savefig('BASE_EXP/fc_threeclass_unsup.png')
close()


figure(figsize=(15,15))
classes=range(10)
plotclasses(classes)
tight_layout()
savefig('BASE_EXP/fc_tenclass_unsup.png')
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
savefig('BASE_EXP/fc_W_unsup.png')
close()



#show()


