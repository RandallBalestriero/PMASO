import cPickle
from pylab import *
import glob
import matplotlib as mpl
label_size = 13
mpl.rcParams['xtick.labelsize'] = label_size+2
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
	return concatenate(x,axis=0),asarray(EM)
 

def plotsuploss(SUP_NO_GIVEN,SUP_NO_RANDOM,label='LIKE'):
    for j in xrange(len(SUP_NO_GIVEN)):
        values0 = []
	values1 = []
        for loss0,loss1 in zip(SUP_NO_GIVEN[j],SUP_NO_RANDOM[j]):# for each run
	    x0,EM0 = format_loss(loss0,label)
            x1,EM1 = format_loss(loss1,label)
            values0.append(x0[-1])# THIS IS THE LAST LIKELIHOOD VALUE
            values1.append(x1[-1])# THIS IS THE LAST LIKELIHOOD VALUE
            subplot(4,2,1+2*j)
            if(label=='LIKE'):
                plot(x0,color='b',linewidth=1)
		plot(x1,color='k',linewidth=1)
            else:
                plot(x0,color='r',linewidth=1)
            if(j==3):
                xlabel('Update Steps',fontsize=22)
            subplot(4,2,2+2*j)
            if(label=='LIKE'):
                plot(range(1,len(EM0)+1),EM0,linewidth=1,ls='-',marker='o',color='b',markersize=3)
                plot(range(1,len(EM1)+1),EM1,linewidth=1,ls='-',marker='o',color='k',markersize=3)
            else:
                plot(range(1,len(EM0)+1),EM,ls='-',marker='o',color='r')
        print 'SUPERVISED',label,' SUP_NO_GIVEN j=',j,mean(values0),std(values0)
        print 'SUPERVISED',label,' SUP_NO_RANDOM j=',j,mean(values1),std(values1)
    xlabel('EM Steps',fontsize=22)

 

def plotunsuploss(UNSUP_NO_GIVEN,UNSUP_NO_RANDOM,label='LIKE'):
    for j in xrange(len(UNSUP_NO_GIVEN)):
        values0 = []
	values1 = []
        for loss0,loss1 in zip(UNSUP_NO_GIVEN[j],UNSUP_NO_RANDOM[j]):# for each run
	    x0,EM0 = format_loss(loss0,label)
            x1,EM1 = format_loss(loss1,label)
#	    print x1
            values0.append(x0[-1])# THIS IS THE LAST LIKELIHOOD VALUE
            values1.append(x1[-1])# THIS IS THE LAST LIKELIHOOD VALUE
            subplot(4,2,1+2*j)
            if(label=='LIKE'):
                plot(x0,color='b',linewidth=1)
                plot(x1,color='k',linewidth=1)
            else:
                plot(x0,color='r',linewidth=1)
            if(j==3):
                xlabel('Update Steps',fontsize=22)
            subplot(4,2,2+2*j)
            if(label=='LIKE'):
                plot(range(1,len(EM0)+1),EM0,linewidth=1,ls='-',marker='o',color='b',markersize=3)
                plot(range(1,len(EM1)+1),EM1,linewidth=1,ls='-',marker='o',color='k',markersize=3)
            else:
                plot(range(1,len(EM0)+1),EM,ls='-',marker='o',color='r')
        print 'UNSUPERVISED',label,' NO_GIVEN j=',j,mean(values0),std(values0)
        print 'UNSUPERVISED',label,' NO_RANDOM j=',j,mean(values1),std(values1)
    xlabel('EM Steps',fontsize=22)

















### SUPERVISED

# INIT onlythetaq RANDOM
def doit(random=0):
    def helper(a,b,c,d,e):
        # SUP TRAIN THETA RANDOM K
        pl=[]
        files = sort(glob.glob('/mnt/project2/rb42Data/PMASO/BASE_EXP/exp_fc_run'+str(a)+'_'+str(b)+'_'+str(c)+'_'+str(d)+'_'+str(e)+'*.pkl'))
        for fil in files:
            print fil
            f=open(fil,'rb')
            pl.append(cPickle.load(f)[0])
            f.close()
        return pl
    print "RANDOM",random
    SUP_TRAIN_GIVEN   = []# EVERYTHING
    SUP_TRAIN_RANDOM  = []
    SUP_NO_GIVEN      = []
    SUP_NO_RANDOM     = []# BASELINE
    UNSUP_NO_GIVEN    = []#EVERYTHING
    UNSUP_NO_RANDOM   = []# BASELINE
    for i in [16,32,48,64]:
        SUP_NO_GIVEN.append(helper(1,0,0,random,i))
        SUP_NO_RANDOM.append(helper(1,0,1,random,i))
        UNSUP_NO_GIVEN.append(helper(0,0,0,random,i))
        UNSUP_NO_RANDOM.append(helper(0,0,1,random,i))
    figure(figsize=(15,10))
    plotsuploss(SUP_NO_GIVEN,SUP_NO_RANDOM,'LIKE')
    tight_layout()
    savefig('BASE_EXP/fc_sup_likelihood_supervised_random'+str(random)+'.png')
    close()
    figure(figsize=(15,10))
    plotunsuploss(UNSUP_NO_GIVEN,UNSUP_NO_RANDOM,'LIKE')
    tight_layout()
    savefig('BASE_EXP/fc_sup_likelihood_unsupervised_random'+str(random)+'.png')
    close()




doit(0)
#doit(1,1)


