import cPickle
from pylab import *
import glob
import matplotlib as mpl
label_size = 13
mpl.rcParams['xtick.labelsize'] = label_size+10
mpl.rcParams['ytick.labelsize'] = label_size

fs=15

def plotloss(losses0,label='LIKE'):
    for j in xrange(4):
        values0 = []
        for loss0 in losses0[j]:# for each run
            values0.append(loss0[-1][0][-1][-1])# THIS IS THE LAST LIKELIHOOD VALUE
            EM   = []
            x    = []
            for i,i_ in zip(loss0,range(len(loss0))):
                if(i[1]==label):
                    x.append(i[0][0])
                    try:
                        if(loss0[i_][1]!=loss0[i_+1][1]):
                            EM.append(i[0][0][-1])
                    except:
                        0
            subplot(4,2,1+2*j)
            if(label=='LIKE'):
                plot(concatenate(x),color='k',linewidth=1)
            else:
                plot(concatenate(x),color='r',linewidth=1)
            if(j==3):
                xlabel('Update Steps',fontsize=28)
            subplot(4,2,2+2*j)
            if(label=='LIKE'):
                plot(range(1,len(EM)+1),EM,ls='-',marker='o',color='k')
            else:
                plot(range(1,len(EM)+1),EM,ls='-',marker='o',color='r')
        print label,'j=',j,mean(values0),std(values0)
    xlabel('EM Steps',fontsize=28)




def plotclasses(classes):
    for i,k in zip(range(len(classes)),classes):
        for j in xrange(10):
            subplot(len(classes),10,1+i*10+j)
            imshow(samplesclass1[k][j,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])





f=open('BASE_EXP/exp_cnn.pkl','rb')
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
savefig('BASE_EXP/cnn_reconstruction.png')
close()





figure(figsize=(15,5))
classes=[0,1,3]
plotclasses(classes)
tight_layout()
savefig('BASE_EXP/cnn_threeclass.png')
close()


figure(figsize=(15,15))
classes=range(10)
plotclasses(classes)
tight_layout()
savefig('BASE_EXP/cnn_tenclass.png')
close()


figure(figsize=(15,3))
for i in xrange(8):
    subplot(2,8,i+1)
    imshow(reshape(W[i,0],(3,3)),aspect='auto',cmap='Greys',interpolation='nearest')
    xticks([])
    yticks([])
    subplot(2,8,i+9)
    imshow(reshape(W[i,1],(3,3)),aspect='auto',cmap='Greys',interpolation='nearest')
    xticks([])
    yticks([])

tight_layout()
show()
#savefig('BASE_EXP/cnn_W.png')
#close()




