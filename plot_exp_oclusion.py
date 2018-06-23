import cPickle
from pylab import *
import glob

fs=15

def plotloss(losses0,label='LIKE'):
    for j in xrange(4):
        values0 = []
        for loss0,loss1 in zip(losses0[j],losses1[j]):# for each run
            values0.append(loss0[-1][0][-1][-1])# THIS IS THE LAST LIKELIHOOD VALUE
            EM   = []
            x    = []
            for i in loss0:
                if(i[1]==label):
                    x.append(i[0])
                    EM.append(i[0][-1])
            subplot(4,2,1+2*j)
            plot(concatenate(x),color='k',linewidth=1)
            if(j==3):
                xlabel('Update Steps',fontsize=28)
            subplot(4,2,2+2*j)
            plot(range(1,len(EM)+1),EM,ls='-o')
        print mean(values),std(values)
    xlabel('EM Steps',fontsize=28)




def plotclasses(classes):
    for i,k in zip(range(len(classes)),classes):
        for j in xrange(10):
            subplot(len(classes),10,1+i*10+j)
            imshow(samplesclass1[k][j,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])





f=open('OCLUSION_EXP/exp_fc_oclusion.pkl','rb')
reconstruction,x,xmasked=cPickle.load(f)
f.close()



figure(figsize=(15,7))
for i in xrange(10):
    subplot(3,10,1+i)
    imshow(x[i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
    xticks([])
    yticks([])
    subplot(3,10,11+i)
    imshow(xmasked[i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
    xticks([])
    yticks([])
    subplot(3,10,21+i)
    imshow(reconstruction[i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
    xticks([])
    yticks([])

tight_layout()
#savefig('BASE_EXP/fc_reconstruction.png')
#close()
show()



