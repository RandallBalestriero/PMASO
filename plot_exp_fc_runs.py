import cPickle
from pylab import *
import glob
import matplotlib as mpl
label_size = 13
mpl.rcParams['xtick.labelsize'] = label_size+10
mpl.rcParams['ytick.labelsize'] = label_size

fs=15



def format_loss(loss,label):
	EM = []
        x  = []
        for i,i_ in zip(loss,range(len(loss))):
            if(i[1]==label):
                x.append(i[0][0])
                try:
                    if(loss[i_][1]!=loss[i_+1][1]):
                        EM.append(i[0][0][-1])
                except:
                    0
	print EM
	return concatenate(x,axis=0),asarray(EM)
 

def plotloss(losses0,losses1,label='LIKE'):
    for j in xrange(len(losses0)):
        values0 = []
	values1 = []
        for loss0,loss1 in zip(losses0[j],losses1[j]):# for each run
	    x0,EM0 = format_loss(loss0,label)
            x1,EM1 = format_loss(loss1,label)
            values0.append(x0[-1])# THIS IS THE LAST LIKELIHOOD VALUE
            values1.append(x1[-1])# THIS IS THE LAST LIKELIHOOD VALUE
            subplot(4,2,1+2*j)
            if(label=='LIKE'):
                plot(x0,color='k',linewidth=1)
		plot(x1,color='b',linewidth=1)
            else:
                plot(x0,color='r',linewidth=1)
            if(j==3):
                xlabel('Update Steps',fontsize=18)
            subplot(4,2,2+2*j)
            if(label=='LIKE'):
                plot(range(1,len(EM0)+1),EM0,linewidth=1,ls='-',marker='o',color='k',markersize=3)
                plot(range(1,len(EM1)+1),EM1,linewidth=1,ls='-',marker='o',color='b',markersize=3)
            else:
                plot(range(1,len(EM0)+1),EM,ls='-',marker='o',color='r')
        print label,' noti j=',j,mean(values0),std(values0)
        print label,' init j=',j,mean(values1),std(values1)
    xlabel('EM Steps',fontsize=18)



### SUPERVISED

# INIT RANDOM
LOSSES00 = []
LOSSES01 = []
LOSSES10 = []
LOSSES11 = []

for i in [16]:#,32,48,64]:
    files = sort(glob.glob('/mnt/project2/rb42Data/PMASO/BASE_EXP/exp_fc_run1_0_0_'+str(i)+'*.pkl'))
    LOSSES00.append([])
    for fil in files:
        print fil
        f=open(fil,'rb')
        LOSSES00[-1].append(cPickle.load(f)[0])
        f.close()
    files = sort(glob.glob('/mnt/project2/rb42Data/PMASO/BASE_EXP/exp_fc_run1_0_1_'+str(i)+'*.pkl'))
    LOSSES01.append([])
    for fil in files:
        print fil
        f=open(fil,'rb')
        LOSSES01[-1].append(cPickle.load(f)[0])
        f.close()
    files = sort(glob.glob('/mnt/project2/rb42Data/PMASO/BASE_EXP/exp_fc_run1_1_0_'+str(i)+'*.pkl'))
    LOSSES10.append([])
    for fil in files:
        print fil
        f=open(fil,'rb')
        LOSSES10[-1].append(cPickle.load(f)[0])
        f.close()
    files = sort(glob.glob('/mnt/project2/rb42Data/PMASO/BASE_EXP/exp_fc_run1_1_1_'+str(i)+'*.pkl'))
    LOSSES11.append([])
    for fil in files:
        print fil
        f=open(fil,'rb')
        LOSSES11[-1].append(cPickle.load(f)[0])
        f.close()




figure(figsize=(15,10))
plotloss(LOSSES10,LOSSES00,'LIKE')
tight_layout()
savefig('BASE_EXP/fc_sup_likelihood_fixed.png')
close()

figure(figsize=(15,10))
plotloss(LOSSES11,LOSSES01,'LIKE')
tight_layout()
savefig('BASE_EXP/fc_sup_likelihood_random.png')
close()




