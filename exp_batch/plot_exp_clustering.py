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





def doit(MODEL,NEURONS,pl=1):
    f=open(SAVE_DIR+'exp_clustering2_'+str(256*15)+'_256_'+MODEL+'_'+NEURONS+'.pkl','rb')
    LOSSES,reconstruction,x,y,samples0,samples1,C=cPickle.load(f)
    print C
    f.close()
    awdawd
    LLL = []
    for i in xrange(len(reconstruction)):
        LLL.append(((reconstruction[i]-x[i])**2).sum())
    MSE = mean(LLL)
    if(pl==0): return  MSE,squeeze(LOSSES)
    for i in xrange(10):
        figure(figsize=(2,2))
        imshow(x[i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
        xticks([])
        yticks([])
        tight_layout()
        savefig('../BASE_EXP/BATCH/sigmass'+'_'+MODEL+'_'+NEURONS+'_x_'+str(i)+'.png')
        close()
        figure(figsize=(2,2))
        imshow(reconstruction[i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
        xticks([])
        yticks([])
        tight_layout()
        savefig('../BASE_EXP/BATCH/sigmass'+'_'+MODEL+'_'+NEURONS+'_reconstruction_'+str(i)+'.png')
        close()
        figure(figsize=(2,2))
        imshow(samples0[i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
        xticks([])
        yticks([])
        tight_layout()
        savefig('../BASE_EXP/BATCH/sigmass'+'_'+MODEL+'_'+NEURONS+'_samples0_'+str(k)+'_'+str(i)+'.png')
#            close()
#            figure(figsize=(2,2))
#            imshow(samples1[k][i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
#            xticks([])
#            yticks([])
#            tight_layout()
#            savefig('../BASE_EXP/BATCH/sigmass'+'_'+MODEL+'_'+NEURONS+'_samples1_'+str(k)+'_'+str(i)+'.png')
#            close()

#        for k in xrange(len(samples0)):
#            figure(figsize=(2,2))
#            imshow(samples0[k][i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
#            xticks([])
#            yticks([])
#            tight_layout()
#            savefig('../BASE_EXP/BATCH/sigmass'+'_'+MODEL+'_'+NEURONS+'_samples0_'+str(k)+'_'+str(i)+'.png')
#            close()
#            figure(figsize=(2,2))
#            imshow(samples1[k][i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
#            xticks([])
#            yticks([])
#            tight_layout()
#            savefig('../BASE_EXP/BATCH/sigmass'+'_'+MODEL+'_'+NEURONS+'_samples1_'+str(k)+'_'+str(i)+'.png')
#            close()
    print LOSSES
    return MSE,squeeze(LOSSES)



#MS = ['x','o','o']
for k in ['10']:
    for model in ['1']:
        l=doit(model,k,pl=1)
#            print shape(l)[5::20],len(l),l[5::20]
#            plot(arange(len(l)),l,color='k',linewidth=2.5)
#            plot(arange(len(l))[5::20],l[5::20],linestyle='None',color='k',linewidth=1,marker=MS[int(model)],markersize=9)
#            yticks([])
#            ylim([0,3100])
#        tight_layout()
#        savefig('../BASE_EXP/FA/'+sig+'_'+k+'_loss.png')


    

