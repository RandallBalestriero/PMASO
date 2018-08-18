import cPickle
from pylab import *
import glob
import matplotlib as mpl
label_size = 13
mpl.rcParams['xtick.labelsize'] = label_size+2
mpl.rcParams['ytick.labelsize'] = label_size

fs=15



def format_loss(loss,label):
	EM      = []
        updates = []
        for i,i_ in zip(loss,range(len(loss))):
            if(i[1]==label):
                updates.append(i[0][0])
                try:
                    if(loss[i_][1]!=loss[i_+1][1]):
                        EM.append(i[0][0][-1])
                except:
                    0
	return concatenate(updates,axis=0),asarray(EM)
 

def plotloss(SUP,label='LIKE'):
    for j in xrange(len(SUP)):# FOR EACH NUMBER OF NEURON CASE
        UPDATES,EM = SUP[j]
	for n_run in xrange(EM.shape[0]):
            subplot(len(SUP),2,1+2*j)
            if(label=='LIKE'):
                plot(UPDATES[n_run],color='b',linewidth=1)
            else:
                plot(UPDATES[n_run],color='r',linewidth=1)
            if(j==len(SUP)-1):
                xlabel('Update Steps',fontsize=22)
            subplot(len(SUP),2,2+2*j)
            if(label=='LIKE'):
                plot(range(1,len(EM[n_run])+1),EM[n_run],linewidth=1,ls='-',color='b')
            else:
                plot(range(1,len(EM[n_run])+1),EM[n_run],ls='-',color='r')
#        print 'SUPERVISED',label,' SUP_NO_GIVEN j=',j,mean(values0),std(values0)
#        print 'SUPERVISED',label,' SUP_NO_RANDOM j=',j,mean(values1),std(values1)
    xlabel('EM Steps',fontsize=22)



### SUPERVISED

# INIT onlythetaq RANDOM
def doit():
    def helper(a,b):
        # SUP TRAIN THETA RANDOM K
        UPDATES,EM=[],[]
        files = sort(glob.glob('/mnt/project2/rb42Data/PMASO/BASE_EXP/exp_fc_run'+str(a)+'_'+str(b)+'*.pkl'))
        for fil in files:
            print fil
            f=open(fil,'rb')
            a,b=format_loss(cPickle.load(f),'LIKE')
            f.close()
	    UPDATES.append(a)
	    EM.append(b)
        return asarray(UPDATES),asarray(EM)
    print "RANDOM",random
    SUP   = [] # EVERYTHING
    UNSUP = []
    for i in [16]:
        SUP.append(helper(0,i))
        UNSUP.append(helper(1,i))
    figure(figsize=(15,10))
    plotloss(SUP,'LIKE')
    tight_layout()
    savefig('BASE_EXP/fc_sup_likelihood_supervised.png')
    close()
    figure(figsize=(15,10))
    plotloss(UNSUP,'LIKE')
    tight_layout()
    savefig('BASE_EXP/fc_sup_likelihood_unsupervised.png')
    close()




doit()
#doit(1,1)


