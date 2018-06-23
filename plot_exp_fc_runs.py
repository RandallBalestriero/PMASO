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
 

def plotloss(losses0,losses1,losses2,label='LIKE'):
    for j in xrange(len(losses0)):
        values0 = []
	values1 = []
        for loss0,loss1,loss2 in zip(losses0[j],losses1[j],losses2[j]):# for each run
	    x0,EM0 = format_loss(loss0,label)
            x1,EM1 = format_loss(loss1,label)
            x2,EM2 = format_loss(loss2,label)
            values0.append(x0[-1])# THIS IS THE LAST LIKELIHOOD VALUE
            values1.append(x1[-1])# THIS IS THE LAST LIKELIHOOD VALUE
            values2.append(x2[-1])# THIS IS THE LAST LIKELIHOOD VALUE
            subplot(4,2,1+2*j)
            if(label=='LIKE'):
                plot(x0,color='k',linewidth=1)
		plot(x1,color='b',linewidth=1)
                plot(x2,color='g',linewidth=1)
            else:
                plot(x0,color='r',linewidth=1)
            if(j==3):
                xlabel('Update Steps',fontsize=18)
            subplot(4,2,2+2*j)
            if(label=='LIKE'):
                plot(range(1,len(EM0)+1),EM0,linewidth=1,ls='-',marker='o',color='k',markersize=3)
                plot(range(1,len(EM1)+1),EM1,linewidth=1,ls='-',marker='o',color='b',markersize=3)
                plot(range(1,len(EM2)+1),EM2,linewidth=1,ls='-',marker='o',color='g',markersize=3)
            else:
                plot(range(1,len(EM0)+1),EM,ls='-',marker='o',color='r')
        print label,' noti j=',j,mean(values0),std(values0)
        print label,' init j=',j,mean(values1),std(values1)
        print label,' only j=',j,mean(values2),std(values2)
    xlabel('EM Steps',fontsize=18)



### SUPERVISED

# INIT onlythetaq RANDOM
def doit(supervised=1,random=0):
    print "SUPERVISED",supervised
    print "RANDOM",random
    LOSSES100 = []
    LOSSES000 = []
    LOSSES010 = []
    for i in [16]:#,32,48,64]:
        files = sort(glob.glob('/mnt/project2/rb42Data/PMASO/BASE_EXP/exp_fc_run'+str(supervised)+'_1_0_'+str(random)+'_'+str(i)+'*.pkl'))
        LOSSES100.append([])
        for fil in files:
            print fil
            f=open(fil,'rb')
            LOSSES100[-1].append(cPickle.load(f)[0])
            f.close()
        files = sort(glob.glob('/mnt/project2/rb42Data/PMASO/BASE_EXP/exp_fc_run'+str(supervised)+'_0_0_'+str(random)+'_'+str(i)+'*.pkl'))
        LOSSES000.append([])
        for fil in files:
            print fil
            f=open(fil,'rb')
            LOSSES000[-1].append(cPickle.load(f)[0])
            f.close()
        files = sort(glob.glob('/mnt/project2/rb42Data/PMASO/BASE_EXP/exp_fc_run'+str(supervised)+'_0_1_'+str(random)+'_'+str(i)+'*.pkl'))
        LOSSES010.append([])
        for fil in files:
            print fil
            f=open(fil,'rb')
            LOSSES010[-1].append(cPickle.load(f)[0])
            f.close()

    figure(figsize=(15,10))
    plotloss(LOSSES100,LOSSES000,LOSSES010,'LIKE')
    tight_layout()
    savefig('BASE_EXP/fc_sup_likelihood_random'+str(random)+'.png')
    close()



doit(1,0)
doit(1,1)


