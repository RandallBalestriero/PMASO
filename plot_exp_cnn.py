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
            print shape(samplesclass0)
	    print shape(samplesclass0[k][j,:,:,0])
            imshow(samplesclass0[k][j,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])





f=open('BASE_EXP/exp_cnn_sup1_1.pkl','rb')
LOSSES,reconstruction,x,samplesclass1,samplesclass0,W=cPickle.load(f)
print reconstruction.max()
f.close()
print LOSSES
plot(unique(LOSSES))
tight_layout()
savefig('BASE_EXP/CNN/losses.png')
close()



for i in xrange(100):
    figure(figsize=(2,2))
    imshow(x[i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
    xticks([])
    yticks([])
    tight_layout()
    savefig('BASE_EXP/CNN/x_'+str(i)+'.png')
    close()
    figure(figsize=(2,2))
    imshow(reconstruction[i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
    xticks([])
    yticks([])
    tight_layout()
    savefig('BASE_EXP/CNN/reconstruction_'+str(i)+'.png')
    close()


for i in xrange(100):
    for j in xrange(10):
        figure(figsize=(2,2))
        imshow(samplesclass0[j][i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
        xticks([])
        yticks([])
        tight_layout()
        savefig('BASE_EXP/CNN/samples0_c'+str(j)+'_'+str(i)+'.png')
	close()
        figure(figsize=(2,2))
        imshow(samplesclass1[j][i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
        xticks([])
        yticks([])
        tight_layout()
        savefig('BASE_EXP/CNN/samples1_c'+str(j)+'_'+str(i)+'.png')
	close()


for k in xrange(4):
    figure(figsize=(20,20))
    for j in xrange(10):
	for i in xrange(10):
            subplot(10,10,i+1+j*10)
            imshow(samplesclass0[j][i+k*10,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])
    tight_layout()
    savefig('BASE_EXP/CNN/allsamples0_'+str(k)+'.png')
    close()
    figure(figsize=(20,20))
    for j in xrange(10):
        for i in xrange(10):
	    subplot(10,10,i+1+j*10)
            imshow(samplesclass1[j][i+k*10,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])
    tight_layout()
    savefig('BASE_EXP/CNN/allsamples1_'+str(k)+'.png')
    close()



figure(figsize=(15,3))
for i in xrange(len(W)):
    subplot(2,len(W),i+1)
    imshow(W[i,0,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest',vmin=W.min(),vmax=W.max())
    xticks([])
    yticks([])
    subplot(2,len(W),i+len(W)+1)
    imshow(W[i,1,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest',vmin=W.min(),vmax=W.max())
    xticks([])
    yticks([])

tight_layout()
savefig('BASE_EXP/CNN/W.png')
close()




