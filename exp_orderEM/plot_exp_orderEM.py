import cPickle
from pylab import *
import glob
import matplotlib as mpl
import os
SAVE_DIR = os.environ['SAVE_DIR']

import glob

label_size = 13
mpl.rcParams['xtick.labelsize'] = label_size+10
mpl.rcParams['ytick.labelsize'] = label_size

fs=15





def doit(per_layer,randomm,mp_opt,leakiness):
    files = glob.glob(SAVE_DIR+'exp_orderEM_'+str(per_layer)+'_'+str(randomm)+'_'+str(mp_opt)+'_'+leakiness+'_run*.pkl')
    MSEs = []
    LIKEs = []
    print files
    for name in files:
        f=open(name)
        LOSSES,reconstruction,x0,samplesclass0,samplesclass1,samples1,params=cPickle.load(f)
        f.close()
        LLL = []
        for k in xrange(1500):
	    LLL.append(((x0[k]-reconstruction[k])**2).sum())
	MSEs.append(mean(LLL))
	LIKEs.append(LOSSES[-1])
    return MSEs,LIKEs


for leakiness in ['0','None']:
    for per_layer in [0,1]:
        for randomm in [0,1]:
	    for mp_opt in [0,1,2,3]:
		try:
	            MSE,lo=doit(per_layer,randomm,mp_opt,leakiness)
	            print mean(MSE),std(MSE),mean(lo),std(lo),'\t'
		except:
	  	    print 'ERROR'
	    print '\n'





