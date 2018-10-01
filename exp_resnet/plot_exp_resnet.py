import cPickle
from pylab import *
import glob
import matplotlib as mpl
import os
SAVE_DIR = os.environ['SAVE_DIR']



label_size = 13
mpl.rcParams['xtick.labelsize'] = label_size+10
mpl.rcParams['ytick.labelsize'] = label_size

fs=15





def doit(neurons,layers,residual,sigmas,plotit=1):
    f=open(SAVE_DIR+'exp_resnet_'+neurons+'_'+layers+'_'+residual+'_'+sigmas+'.pkl','rb')
    print 'exp_resnet_'+neurons+'_'+layers+'_'+residual+'_'+sigmas
    LOSSES,reconstruction,x0,samplesclass0,samplesclass1,samples1,params=cPickle.load(f)
    f.close()
    if(plotit==0): return LOSSES
    LLL=[]
    for i in xrange(150):
	figure(figsize=(2,2))
        imshow(x0[i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
        xticks([])
        yticks([])
        tight_layout()
        savefig('../BASE_EXP/resnet/x_'+neurons+'_'+layers+'_'+residual+'_'+sigmas+'_'+str(i)+'.png')
	close()
        figure(figsize=(2,2))
        imshow(reconstruction[i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
        xticks([])
        yticks([])
        tight_layout()
        savefig('../BASE_EXP/resnet/reconstruction_'+neurons+'_'+layers+'_'+residual+'_'+sigmas+'_'+str(i)+'.png')
	close()
	LLL.append(((x0[i,:,:,0]-reconstruction[i,:,:,0])**2).sum())
    print mean(LLL),len(LOSSES),LOSSES[-1]
    for k in xrange(len(samplesclass0)):
	for i in xrange(20):
            figure(figsize=(2,2))
            imshow(samplesclass0[k][i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])
            tight_layout()
            savefig('../BASE_EXP/resnet/samples0_'+neurons+'_'+layers+'_'+residual+'_'+sigmas+'_'+str(k)+'_'+str(i)+'.png')
	    close()
            figure(figsize=(2,2))
            imshow(samplesclass1[k][i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])
            tight_layout()
            savefig('../BASE_EXP/resnet/samples1_'+neurons+'_'+layers+'_'+residual+'_'+sigmas+'_'+str(k)+'_'+str(i)+'.png')
	    close()
    return LOSSES


for l in ['2']:
#    lo = doit('32',l,'0','global')
    lo = doit('64','1','0','local',1)
#    lo = doit('64','3','1','global',1)
    print lo
#    lo = doit('64','1','1','local',1)
#    print lo

#    plot(lo)
#    savefig('../BASE_EXP/resnet/loss_32_2_1_global.png')





