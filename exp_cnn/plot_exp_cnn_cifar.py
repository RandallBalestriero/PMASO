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

def normalize(x):
	return (x-x.min())/(x.max()-x.min())


def plotclasses(classes,samplesclass1):
    for i,k in zip(range(len(classes)),classes):
	print shape(samplesclass1[k])
        for j in xrange(20):
            subplot(len(classes),20,1+i*20+j)
            imshow(normalize(samplesclass1[k][j+i*20]),aspect='auto',interpolation='nearest')
            xticks([])
            yticks([])





def doit(sig,k,doplot=1):
    f=open(SAVE_DIR+'exp_cnn_cifar_'+sig+'_'+str(k)+'.pkl','rb')
    LOSSES1,reconstruction1,x1,samplesclass01,samplesclass11,W1=cPickle.load(f)
    f.close()
    if(doplot==0): return concatenate(LOSSES1)
    w = W1[0]
    for i in xrange(w.shape[0]):
        figure(figsize=(2,2))
        imshow(normalize(reshape(w[i,0],(5,5,3))),aspect='auto',interpolation='nearest')
        xticks([])
        yticks([])
        tight_layout()
        savefig('BASE_EXP/CNN_CIFAR/w_'+sig+'_c'+str(k)+'_r0_'+str(i)+'.png')
        close()
        figure(figsize=(2,2))
        imshow(normalize(reshape(w[i,1],(5,5,3))),aspect='auto',interpolation='nearest')
        xticks([])
        yticks([])
        tight_layout()
        savefig('BASE_EXP/CNN_CIFAR/w_'+sig+'_c'+str(k)+'_r1_'+str(i)+'.png')
        close()
    LLL = []
    for i in xrange(150):
        figure(figsize=(2,2))
        imshow(normalize(x1[i]),aspect='auto',interpolation='nearest')
        xticks([])
        yticks([])
        tight_layout()
        savefig('BASE_EXP/CNN_CIFAR/x_'+sig+'_c'+str(k)+'_'+str(i)+'.png')
        close()
    for i in xrange(150):
	LLL.append(((x1[i]-reconstruction1[i])**2).sum())
        figure(figsize=(2,2))
        imshow(normalize(reconstruction1[i]),aspect='auto',interpolation='nearest')
        xticks([])
        yticks([])
        tight_layout()
        savefig('BASE_EXP/CNN_CIFAR/reconstruction_x_'+sig+'_c'+str(k)+'_'+str(i)+'.png')
        close()
    print "MSE",mean(LLL),"C",k
    for i in xrange(4):
	figure(figsize=(20,2))
	for ii in xrange(10):
	    subplot(1,10,1+ii)
	    imshow(normalize(samplesclass11[-1][i*10+ii]),aspect='auto',interpolation='nearest')
            xticks([])
            yticks([])
        tight_layout()
        savefig('BASE_EXP/CNN_CIFAR/samples1_'+sig+'_c'+str(k)+'_n'+str(i)+'.png')
        close()
    for n in xrange(150):
        figure(figsize=(3,3))
        imshow(normalize(samplesclass11[-1][n]),aspect='auto',interpolation='nearest')
        xticks([])
        yticks([])
        tight_layout()
        savefig('BASE_EXP/CNN_CIFAR/tenclass1_'+sig+'_c'+str(k)+'_n'+str(n)+'.png')
        close()
#    for n in xrange(2):
#        figure(figsize=(3,3))
#        imshow(normalize(samplesclass01[-1][0][n]),aspect='auto',interpolation='nearest')
#        xticks([])
#        yticks([])
#        tight_layout()
#        savefig('BASE_EXP/cifar/tenclass0_'+sig+'_'+str(l)+'_c'+str(k)+'_n'+str(n)+'.png')
#        close()
    return LOSSES1


for c in [3,5,6,8,9]:
	loss1 = doit('channel',c,1)
	print "C",c
	print loss1[-1]
	figure(figsize=(3,3))
	try:
		plot(unique(concatenate(loss1)),c='k',ls='-',linewidth=3)
	except:
                plot(unique(loss1),c='k',ls='-',linewidth=3)
        tight_layout()
        savefig('BASE_EXP/CNN_CIFAR/loss_'+str(c)+'.png')
        close()



