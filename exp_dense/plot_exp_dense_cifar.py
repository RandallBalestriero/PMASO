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





def doit(sig,l,k,doplot=1):
    f=open(SAVE_DIR+'exp_cifar_'+sig+'_'+str(l)+'_'+str(k)+'.pkl','rb')
    LOSSES1,reconstruction1,x1,samplesclass01,samplesclass11,W1=cPickle.load(f)
    f.close()
    if(doplot==0): return concatenate(LOSSES1)
    w = W1[-1][0]
    LLL = []
    for i in xrange(150):
        figure(figsize=(2,2))
        imshow(normalize(x1[i]),aspect='auto',interpolation='nearest')
        xticks([])
        yticks([])
        tight_layout()
        savefig('BASE_EXP/DENSE_CIFAR/x'+sig+'_'+str(l)+'_'+str(k)+'_'+str(i)+'.png')
        close()
    for i in xrange(150):
	LLL.append(((x1[i]-reconstruction1[-1][i])**2).sum())
        figure(figsize=(2,2))
        imshow(normalize(reconstruction1[-1][i]),aspect='auto',interpolation='nearest')
        xticks([])
        yticks([])
        tight_layout()
        savefig('BASE_EXP/DENSE_CIFAR/reconstruction_x'+sig+'_'+str(l)+'_'+str(k)+'_'+str(i)+'.png')
        close()
    print mean(LLL)
#    for i in xrange(4):
#	figure(figsize=(20,2))
#	for ii in xrange(10):
#	    imshow(normalize(samplesclass11[-1][0][i*10+ii]),aspect='auto',interpolation='nearest')
#            xticks([])
#            yticks([])
#        tight_layout()
#        savefig('BASE_EXP/DENSE_CIFAR/samples1_'+sig+'_'+str(l)+'_c'+str(k)+'_n'+str(i)+'.png')
#        close()
#    for n in xrange(150):
#        figure(figsize=(3,3))
#        imshow(normalize(samplesclass11[-1][0][n]),aspect='auto',interpolation='nearest')
#        xticks([])
#        yticks([])
#        tight_layout()
#        savefig('BASE_EXP/cifar/tenclass1_'+sig+'_'+str(l)+'_c'+str(k)+'_n'+str(n)+'.png')
#        close()
#    for n in xrange(2):
#        figure(figsize=(3,3))
#        imshow(normalize(samplesclass01[-1][0][n]),aspect='auto',interpolation='nearest')
#        xticks([])
#        yticks([])
#        tight_layout()
#        savefig('BASE_EXP/cifar/tenclass0_'+sig+'_'+str(l)+'_c'+str(k)+'_n'+str(n)+'.png')
#        close()
    return concatenate(LOSSES1)


for c in [0,1,2,3,4,5,6,7,8,9]:
	print "diagonal 3"
	loss1 = doit('local',3,c,1)
	print "isotropic 3"
	loss2 = doit('global',3,c,1)
	print "diagonal 2"
        loss3 = doit('local',2,c,1)
	print "isotropic 2"
        loss4 = doit('global',2,c,1)
	print "\n\n"
#	print "C",c
#	print loss3[-1],loss1[-1]
#	print loss4[-1],loss2[-1]
#	plot(unique(loss1),c='k',ls='-')
##        plot(unique(loss2),c='k',ls='--')
##        plot(unique(loss3),c='b',ls='-')
##        plot(unique(loss4),c='b',ls='--')
#	ylim([-2000,6000])
#        tight_layout()
#        savefig('BASE_EXP/cifar/loss_'+str(c)+'.png')
#        close()



