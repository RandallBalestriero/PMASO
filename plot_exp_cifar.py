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





def doit(sig,l,k):
    f=open(SAVE_DIR+'exp_cifar_'+sig+'_'+str(l)+'_'+str(k)+'.pkl','rb')
    LOSSES1,reconstruction1,x1,samplesclass01,samplesclass11,W1=cPickle.load(f)
    f.close()
    figure(figsize=(20,2))
    for i in xrange(15):
	print shape(x1[i])
        subplot(1,15,1+i)
        imshow(normalize(x1[i]),aspect='auto',interpolation='nearest')
        xticks([])
        yticks([])
    tight_layout()
    savefig('BASE_EXP/cifar/reconstruction_x'+sig+'_'+str(l)+'_'+str(k)+'.png')
    close()
    for reconstruction,j in zip(reconstruction1,range(len(reconstruction1))):
        figure(figsize=(20,2))
        for i in xrange(15):
            subplot(1,15,1+i)
            imshow(normalize(reconstruction[i]),aspect='auto',interpolation='nearest')
            xticks([])
            yticks([])
        tight_layout()
        savefig('BASE_EXP/cifar/reconstruction_x'+sig+'_'+str(l)+'_'+str(j)+'_'+str(k)+'.png')
        close()
    for n in xrange(150):
        figure(figsize=(3,3))
        imshow(normalize(samplesclass11[-1][0][n]),aspect='auto',interpolation='nearest')
        xticks([])
        yticks([])
        tight_layout()
        savefig('BASE_EXP/cifar/tenclass1_'+sig+'_'+str(l)+'_c'+str(k)+'_n'+str(n)+'.png')
        close()
    for n in xrange(2):
        figure(figsize=(3,3))
        imshow(normalize(samplesclass01[-1][0][n]),aspect='auto',interpolation='nearest')
        xticks([])
        yticks([])
        tight_layout()
        savefig('BASE_EXP/cifar/tenclass0_'+sig+'_'+str(l)+'_c'+str(k)+'_n'+str(n)+'.png')
        close()
    return concatenate(LOSSES1)


for c in [7,8,9]:
	loss1 = doit('local',3,c)
	loss2 = doit('global',3,c)
        loss3 = doit('local',2,c)
        loss4 = doit('global',2,c)
	plot(unique(loss1),c='k',ls='-')
        plot(unique(loss2),c='k',ls='--')
        plot(unique(loss3),c='b',ls='-')
        plot(unique(loss4),c='b',ls='--')
	ylim([-2000,6000])
        tight_layout()
        savefig('BASE_EXP/cifar/loss_'+str(c)+'.png')
        close()



