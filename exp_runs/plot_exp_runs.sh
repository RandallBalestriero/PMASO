import cPickle
from pylab import *
import glob
import matplotlib as mpl
label_size = 13
mpl.rcParams['xtick.labelsize'] = label_size+2
mpl.rcParams['ytick.labelsize'] = label_size

fs=15



def format_loss(loss,label):
	LOSS,GAIN = loss
	loss_,gain_ = [],[]
        for i in GAIN:
            if(i[1]==label):
                gain_.append(i[0])
	return cumsum(asarray(gain_))
 

def plotloss(A,B,label='LIKE'):
    for j in xrange(len(A)):# FOR EACH NUMBER OF NEURON CASE
        l = A[j]
	for n_run in xrange(l.shape[0]):
#            subplot(len(A),2,1+2*j)
#            if(label=='LIKE'):
#                plot(l[n_run],color='b',linewidth=1)
#            else:
#                plot(l[n_run],color='r',linewidth=1)
#            if(j==len(A)-1):
#                xlabel('Update Steps',fontsize=22)
            subplot(len(A),2,2+2*j)
            if(label=='LIKE'):
                plot(range(1,len(l[n_run])+1),l[n_run],linewidth=1,ls='-',color='b')
            else:
                plot(range(1,len(l[n_run])+1),l[n_run],ls='-',color='r')
#        print 'SUPERVISED',label,' SUP_NO_GIVEN j=',j,mean(values0),std(values0)
#        print 'SUPERVISED',label,' SUP_NO_RANDOM j=',j,mean(values1),std(values1)
    xlabel('EM Steps',fontsize=22)



### SUPERVISED

# INIT onlythetaq RANDOM
def doit():
    def helper(a,b):
        # SUP TRAIN THETA RANDOM K
        UPDATES,EM=[],[]
        files = sort(glob.glob('/mnt/project2/rb42Data/PMASO/BASE_EXP/exp_fc_run'+str(a)+'_'+str(b)+'_local_local_local_*.pkl'))
        for fil in files:
            print fil
            f=open(fil,'rb')
            b=format_loss(cPickle.load(f),'LIKE')
            f.close()
#	    UPDATES.append(a)
	    EM.append(b)
        return asarray(EM)
    print "RANDOM",random
    SUP   = [] # EVERYTHING
    UNSUP = []
    for i in [2]:
        SUP.append(helper(1,i))
#        UNSUP.append(helper(1,i))
    figure(figsize=(15,10))
    plotloss(SUP,0,'LIKE')
    tight_layout()
    savefig('BASE_EXP/fc_sup_likelihood_supervised.png')
    close()
#    figure(figsize=(15,10))
#    plotloss(UNSUP,'LIKE')
#    tight_layout()
#    savefig('BASE_EXP/fc_sup_likelihood_unsupervised.png')
#    close()




doit()
#doit(1,1)


