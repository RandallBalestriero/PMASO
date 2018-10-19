import cPickle
from pylab import *
import glob
import matplotlib as mpl


import cPickle
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





def doit(DATASET,CLASS,MODEL,pl=1):
    f=open(SAVE_DIR+'exp_outlier_'+DATASET+'_'+str(CLASS)+'_'+str(MODEL)+'.pkl','rb')
    LOSSES,x,evidence=cPickle.load(f)
    x-=x.min((1,2,3),keepdims=True)
    x/=x.max((1,2,3),keepdims=True)
    f.close()
    figure(figsize=(3,3))
    hist(evidence,100)
#    xlim([6500,13500])
    xticks([])
    yticks([])
    savefig('../BASE_EXP/outlier/histo_'+DATASET+'_'+str(CLASS)+'_'+str(MODEL)+'.png')
    return 0
    S = argsort(evidence)
    if(DATASET=='MNIST'):
        for i in xrange(20):
            figure(figsize=(2,2))
            imshow(x[S[i],:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])
            tight_layout()
            savefig('../BASE_EXP/outlier/outlier_'+DATASET+'_'+str(CLASS)+'_'+str(MODEL)+'_'+str(i)+'.png')
            figure(figsize=(2,2))
            imshow(x[S[-i-1],:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])
            tight_layout()
            savefig('../BASE_EXP/outlier/nonoutlier_'+DATASET+'_'+str(CLASS)+'_'+str(MODEL)+'_'+str(i)+'.png')
    else:
        for i in xrange(20):
            figure(figsize=(2,2))
            imshow(x[S[i]],aspect='auto',interpolation='nearest')
            xticks([])
            yticks([])
            tight_layout()
            savefig('../BASE_EXP/outlier/outlier_'+DATASET+'_'+str(CLASS)+'_'+str(MODEL)+'_'+str(i)+'.png')
            figure(figsize=(2,2))
            imshow(x[S[-i-1]],aspect='auto',interpolation='nearest')
            xticks([])
            yticks([])
            tight_layout()
            savefig('../BASE_EXP/outlier/nonoutlier_'+DATASET+'_'+str(CLASS)+'_'+str(MODEL)+'_'+str(i)+'.png')
    return LOSSES[-1]



MS = ['x','o','o']
for d in ['MNIST']:
    for model in ['0','1']:
        for c in ['0']:#,'1','2','3','4','5','6','7','8','9']:
            l=doit(d,c,model,pl=1)


    

